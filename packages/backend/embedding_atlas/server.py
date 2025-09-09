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
from .cli import DEFAULT_TEXT_MODEL


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

        # --- helpers ---
        def model_slug(s: str) -> str:
            """
            Keep the model name as close as possible to original:
            - Preserve letters, digits, underscore, hyphen, and dot.
            - Replace all other characters with '_'.
            - Trim leading/trailing underscores introduced by replacements.
            This keeps `-` (hyphen) and `.` (dot) intact for readability.
            """
            s = str(s)
            s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
            return s.strip("_") or "model"

        def drop_previous_embedding_columns():
            # Drop columns referenced in metadata (if present)
            cols_meta = data_source.metadata.get("columns", {})
            emb_meta = cols_meta.get("embedding", {})
            old_x = emb_meta.get("x")
            old_y = emb_meta.get("y")
            old_neighbors = cols_meta.get("neighbors")
            to_drop = []
            for c in (old_x, old_y, old_neighbors):
                if c and c in data_source.dataset.columns:
                    to_drop.append(c)

            # Drop any residual projection/neighbor columns by prefix pattern
            for col in list(data_source.dataset.columns):
                name = str(col)
                if (
                    name.startswith("projection_x")
                    or name.startswith("projection_y")
                    or name.startswith("__neighbors")
                ):
                    if col not in to_drop:
                        to_drop.append(col)

            if to_drop:
                data_source.dataset.drop(
                    columns=[c for c in to_drop if c in data_source.dataset.columns],
                    inplace=True,
                    errors="ignore",
                )

        try:
            # Extract and validate parameters
            model = request_data.get("model", DEFAULT_TEXT_MODEL)
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
                    status_code=400,
                )

            if not isinstance(umap_args, dict):
                return JSONResponse({"error": "umap_args must be a dictionary"}, status_code=400)

            if batch_size is not None and (not isinstance(batch_size, int) or batch_size <= 0):
                return JSONResponse({"error": "batch_size must be a positive integer"}, status_code=400)

            # Always create fresh columns and drop any previous ones
            # TODO: Implement drop_previous_embedding_columns() if needed

            mslug = model_slug(model)
            x_column = f"projection_x__{mslug}"
            y_column = f"projection_y__{mslug}"
            neighbors_column = f"__neighbors__{mslug}"

            # Compute new embeddings and projections into the fresh columns
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


            # Update metadata to reflect the new columns.
            data_source.update_embedding_metadata(
                {"x": x_column, "y": y_column},
                neighbors_column,
                text_column,
            )

            # Also record the model used (since we removed UMAP params from names)
            try:
                data_source.metadata.setdefault("columns", {})
                data_source.metadata["columns"].setdefault("embedding_info", {})
                data_source.metadata["columns"]["embedding_info"].update(
                    {"model": model}
                )
            except Exception as _e:
                print(f"Warning: could not persist embedding_info into metadata: {_e}")


            # Clear the dataset.parquet cache to reflect updated dataset
            clear_dataset_cache()

            # Force a call to regenerate content immediately to flush any stale references
            try:
                # This will force the lru_cache to regenerate content with fresh data
                _ = to_parquet_bytes(data_source.dataset)
            except Exception as e:
                print(f"Warning: Error forcing fresh content: {e}")


            return JSONResponse(
                {
                    "success": True,
                    "message": f"Embeddings computed with model '{model}'",
                    "columns": {
                        "x": x_column,
                        "y": y_column,
                        "neighbors": neighbors_column,
                        "text": text_column,
                    },
                    "provenance": {
                        "model": model,
                    },
                }
            )

        except ImportError as e:
            return JSONResponse(
                {"error": f"Required package not available: {str(e)}"},
                status_code=500,
            )
        except Exception as e:
            return JSONResponse(
                {"error": f"Failed to compute embeddings: {str(e)}"},
                status_code=500,
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
    @lru_cache(maxsize=10)
    def get_content() -> bytes:
        return make_content()

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
            }
        )

    @app.get(url)
    async def get(request: Request):
        content = get_content()
        bytes_range = parse_range_header(request, len(content))
        if bytes_range is None:
            return Response(
                content=content,
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
                },
                media_type=media_type,
                status_code=206,
            )
    
    # Return the cache clearing function
    return get_content.cache_clear
