# Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import asyncio
import concurrent.futures
import json
import os
import re
import uuid
from functools import lru_cache
from typing import Callable

import duckdb
import pyarrow as pa
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .data_source import DataSource
from .utils import to_parquet_bytes


def make_server(
    data_source: DataSource,
    static_path: str,
    duckdb_uri: str | None = None,
):
    """Creates a server for hosting Embedding Atlas"""

    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )

    clear_dataset_cache = mount_bytes(
        app,
        "/data/dataset.parquet",
        "application/octet-stream",
        lambda: to_parquet_bytes(data_source.dataset),
    )

    @app.get("/data/metadata.json")
    async def get_metadata():
        if duckdb_uri is None or duckdb_uri == "wasm":
            db_meta = {"database": {"type": "wasm", "load": True}}
        elif duckdb_uri == "server":
            # Point to the server itself.
            db_meta = {"database": {"type": "rest"}}
        else:
            # Point to the given uri.
            if duckdb_uri.startswith("http"):
                db_meta = {
                    "database": {"type": "rest", "uri": duckdb_uri, "load": True}
                }
            elif duckdb_uri.startswith("ws"):
                db_meta = {
                    "database": {"type": "socket", "uri": duckdb_uri, "load": True}
                }
            else:
                raise ValueError("invalid DuckDB uri")
        return data_source.metadata | db_meta

    @app.post("/data/cache/{name}")
    async def post_cache(request: Request, name: str):
        data_source.cache_set(name, await request.json())

    @app.get("/data/cache/{name}")
    async def get_cache(name: str):
        obj = data_source.cache_get(name)
        if obj is None:
            return Response(status_code=404)
        return obj

    @app.get("/data/archive.zip")
    async def make_archive():
        data = data_source.make_archive(static_path)
        return Response(content=data, media_type="application/zip")

    # Database connection

    @lru_cache(maxsize=1)
    def get_connection():
        con = duckdb.connect(":memory:")
        df = data_source.dataset
        _ = df
        con.sql("CREATE TABLE dataset AS (SELECT * FROM df)")
        return con

    def handle_query(query: dict):
        sql = query["sql"]
        command = query["type"]
        with get_connection().cursor() as cursor:
            try:
                result = cursor.execute(sql)
                if command == "exec":
                    return JSONResponse({})
                elif command == "arrow":
                    buf = arrow_to_bytes(result.arrow())
                    return Response(
                        buf, headers={"Content-Type": "application/octet-stream"}
                    )
                elif command == "json":
                    data = result.df().to_json(orient="records")
                    return Response(data, headers={"Content-Type": "application/json"})
                else:
                    raise ValueError(f"Unknown command {command}")
            except Exception as e:
                return JSONResponse({"error": str(e)}, status_code=500)

    def handle_selection(query: dict):
        predicate = query.get("predicate", None)
        format = query["format"]
        formats = {
            "json": "(FORMAT JSON, ARRAY true)",
            "jsonl": "(FORMAT JSON)",
            "csv": "(FORMAT CSV)",
            "parquet": "(FORMAT parquet)",
        }
        with get_connection().cursor() as cursor:
            filename = ".selection-" + str(uuid.uuid4()) + ".tmp"
            try:
                if predicate is not None:
                    cursor.execute(
                        f"COPY (SELECT * FROM dataset WHERE {predicate}) TO '{filename}' {formats[format]}"
                    )
                else:
                    cursor.execute(f"COPY dataset TO '{filename}' {formats[format]}")
                with open(filename, "rb") as f:
                    buffer = f.read()
                    return Response(
                        buffer, headers={"Content-Type": "application/octet-stream"}
                    )
            except Exception as e:
                return JSONResponse({"error": str(e)}, status_code=500)
            finally:
                try:
                    os.unlink(filename)
                except Exception:
                    pass

    executor = concurrent.futures.ThreadPoolExecutor()

    @app.get("/data/query")
    async def get_query(req: Request):
        data = json.loads(req.query_params["query"])
        return await asyncio.get_running_loop().run_in_executor(
            executor, lambda: handle_query(data)
        )

    @app.post("/data/query")
    async def post_query(req: Request):
        body = await req.body()
        data = json.loads(body)
        return await asyncio.get_running_loop().run_in_executor(
            executor, lambda: handle_query(data)
        )

    @app.post("/data/selection")
    async def post_selection(req: Request):
        body = await req.body()
        data = json.loads(body)
        return await asyncio.get_running_loop().run_in_executor(
            executor, lambda: handle_selection(data)
        )

    def handle_embedding(request_data: dict):
        from .projection import compute_text_projection
        
        try:
            # Extract and validate parameters
            model = request_data.get("model", "all-MiniLM-L6-v2")
            text_column = request_data.get("text_column")
            trust_remote_code = request_data.get("trust_remote_code", False)
            batch_size = request_data.get("batch_size")
            umap_args = request_data.get("umap_args", {})
            
            # Validate required parameters
            if text_column is None:
                return JSONResponse({"error": "text_column is required"}, status_code=400)
            
            if text_column not in data_source.dataset.columns:
                return JSONResponse(
                    {"error": f"Column '{text_column}' not found in dataset"}, 
                    status_code=400
                )
            
            # Validate umap_args
            if not isinstance(umap_args, dict):
                return JSONResponse({"error": "umap_args must be a dictionary"}, status_code=400)
            
            # Validate batch_size if provided
            if batch_size is not None and (not isinstance(batch_size, int) or batch_size <= 0):
                return JSONResponse({"error": "batch_size must be a positive integer"}, status_code=400)
            
            # Reuse existing projection column names or create new ones
            from .cli import find_column_name
            
            # Check if we already have projection columns to reuse
            existing_embedding = data_source.metadata.get("columns", {}).get("embedding")
            if existing_embedding and "x" in existing_embedding and "y" in existing_embedding:
                x_column = existing_embedding["x"]
                y_column = existing_embedding["y"]
            else:
                x_column = find_column_name(data_source.dataset.columns, "projection_x")
                y_column = find_column_name(data_source.dataset.columns, "projection_y")
            
            # Check for existing neighbors column
            existing_neighbors = data_source.metadata.get("columns", {}).get("neighbors")
            if existing_neighbors and existing_neighbors in data_source.dataset.columns:
                neighbors_column = existing_neighbors
            else:
                neighbors_column = find_column_name(data_source.dataset.columns, "__neighbors")
            
            # Debug: Check first row before computation
            first_row_before = data_source.dataset.iloc[0].to_dict()
            print(f"First row BEFORE embedding computation: {first_row_before}")
            
            # Compute new embeddings and projections
            compute_text_projection(
                data_source.dataset,
                text_column,
                x=x_column,
                y=y_column,
                neighbors=neighbors_column,
                model=model,
                trust_remote_code=trust_remote_code,
                batch_size=batch_size,
                umap_args=umap_args,
            )
            
            # Debug: Check first row after computation
            first_row_after = data_source.dataset.iloc[0].to_dict()
            print(f"First row AFTER embedding computation: {first_row_after}")
            
            # Update metadata
            data_source.update_embedding_metadata({
                "x": x_column,
                "y": y_column,
            }, neighbors_column, text_column)
            
            # Clear the connection cache to reflect updated dataset
            get_connection.cache_clear()
            
            # Clear the dataset.parquet cache to reflect updated dataset
            clear_dataset_cache()
            print("Cleared dataset cache")
            
            # Debug: Test that parquet bytes are actually different
            parquet_bytes = to_parquet_bytes(data_source.dataset)
            print(f"Parquet bytes length after clearing cache: {len(parquet_bytes)}")
            
            return JSONResponse({
                "success": True,
                "message": f"Embeddings computed successfully using model '{model}'",
                "columns": {
                    "x": x_column,
                    "y": y_column,
                    "neighbors": neighbors_column,
                    "text": text_column
                }
            })
            
        except ImportError as e:
            return JSONResponse(
                {"error": f"Required package not available: {str(e)}"}, 
                status_code=500
            )
        except Exception as e:
            return JSONResponse(
                {"error": f"Failed to compute embeddings: {str(e)}"}, 
                status_code=500
            )

    @app.post("/data/embedding")
    async def post_embedding(req: Request):
        body = await req.body()
        data = json.loads(body)
        return await asyncio.get_running_loop().run_in_executor(
            executor, lambda: handle_embedding(data)
        )

    # Static files for the frontend
    app.mount("/", StaticFiles(directory=static_path, html=True))

    return app


def arrow_to_bytes(arrow):
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, arrow.schema) as writer:
        writer.write(arrow)
    return sink.getvalue().to_pybytes()


def parse_range_header(request: Request, content_length: int):
    value = request.headers.get("Range")
    if value is not None:
        m = re.match(r"^ *bytes *= *([0-9]+) *- *([0-9]+) *$", value)
        if m is not None:
            r0 = int(m.group(1))
            r1 = int(m.group(2)) + 1
            if r0 < r1 and r0 <= content_length and r1 <= content_length:
                return (r0, r1)
    return None


def mount_bytes(
    app: FastAPI, url: str, media_type: str, make_content: Callable[[], bytes]
):
    @lru_cache(maxsize=1)
    def get_content() -> bytes:
        print("mount_bytes: Generating fresh content")
        content = make_content()
        print(f"mount_bytes: Generated {len(content)} bytes")
        return content

    @app.head(url)
    async def head(request: Request):
        content = get_content()
        bytes_range = parse_range_header(request, len(content))
        if bytes_range is None:
            length = len(content)
        else:
            length = bytes_range[1] - bytes_range[0]
        return Response(
            headers={
                "Content-Length": str(length),
                "Content-Type": media_type,
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
            }
        )

    @app.get(url)
    async def get(request: Request):
        content = get_content()
        bytes_range = parse_range_header(request, len(content))
        if bytes_range is None:
            return Response(
                content=content,
                headers={
                    "Cache-Control": "no-cache, no-store, must-revalidate",
                    "Pragma": "no-cache", 
                    "Expires": "0",
                }
            )
        else:
            r0, r1 = bytes_range
            result = content[r0:r1]
            return Response(
                content=result,
                headers={
                    "Content-Length": str(r1 - r0),
                    "Content-Range": f"bytes {r0}-{r1 - 1}/{len(content)}",
                    "Content-Type": media_type,
                    "Cache-Control": "no-cache, no-store, must-revalidate",
                    "Pragma": "no-cache",
                    "Expires": "0",
                },
                media_type=media_type,
                status_code=206,
            )
    
    # Return the cache clearing function
    return get_content.cache_clear
