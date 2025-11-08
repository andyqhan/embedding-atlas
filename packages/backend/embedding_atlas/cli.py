# Copyright (c) 2025 Apple Inc. Licensed under MIT License.

"""Command line interface."""

# Default values for embedding/projection parameters
DEFAULT_TEXT_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_IMAGE_MODEL = "google/vit-base-patch16-384"
DEFAULT_UMAP_METRIC = "cosine"
DEFAULT_TEXT_BATCH_SIZE = 32  # What gets used internally
DEFAULT_IMAGE_BATCH_SIZE = 16

import asyncio
import logging
import pathlib
import re
import socket
from pathlib import Path

import click
import inquirer
import numpy as np
import pandas as pd
import uvicorn

from .data_source import DataSource
from .server import make_server
from .utils import Hasher, load_huggingface_data, load_pandas_data
from .version import __version__


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


def find_column_name(existing_names, candidate):
    if candidate not in existing_names:
        return candidate
    else:
        index = 1
        while True:
            s = f"{candidate}_{index}"
            if s not in existing_names:
                return s
            index += 1


def determine_and_load_data(filename: str, splits: list[str] | None = None):
    suffix = Path(filename).suffix.lower()
    hf_prefix = "hf://datasets/"

    # Override Hugging Face data if given full url
    if filename.startswith(hf_prefix):
        filename = filename.split(hf_prefix)[-1]

    # Hugging Face data
    if (len(filename.split("/")) <= 2) and (suffix == ""):
        df = load_huggingface_data(filename, splits)
    else:
        df = load_pandas_data(filename)

    return df


def load_datasets(
    inputs: list[str], splits: list[str] | None = None, sample: int | None = None
) -> pd.DataFrame:
    existing_column_names = set()
    dataframes = []
    for fn in inputs:
        print("Loading data from " + fn)
        df = determine_and_load_data(fn, splits=splits)
        dataframes.append(df)
        for c in df.columns:
            existing_column_names.add(c)

    file_name_column = find_column_name(existing_column_names, "FILE_NAME")
    for df, fn in zip(dataframes, inputs):
        df[file_name_column] = fn

    df = pd.concat(dataframes)

    if sample:
        df = df.sample(n=sample, axis=0, random_state=np.random.RandomState(42))

    return df


def prompt_for_column(df: pd.DataFrame, message: str) -> str | None:
    question = [
        inquirer.List(
            "arg",
            message=message,
            choices=sorted(["(none)"] + [str(c) for c in df.columns]),
        ),
    ]
    r = inquirer.prompt(question)
    if r is None:
        return None
    text = r["arg"]  # type: ignore
    if text == "(none)":
        text = None
    return text


def find_available_port(start_port: int, max_attempts: int = 10, host="localhost"):
    """Find the next available port starting from start_port."""
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex((host, port)) != 0:
                return port
    raise RuntimeError("No available ports found in the given range")


@click.command()
@click.argument("inputs", nargs=-1, required=True)
@click.option("--text", default=None, help="Column containing text data.")
@click.option("--image", default=None, help="Column containing image data.")
@click.option(
    "--vector", default=None, help="Column containing pre-computed vector embeddings."
)
@click.option(
    "--split",
    default=[],
    multiple=True,
    help="Dataset split name(s) to load from Hugging Face datasets. Can be specified multiple times for multiple splits.",
)
@click.option(
    "--enable-projection/--disable-projection",
    "enable_projection",
    default=True,
    help="Compute embedding projections from text/image/vector data. If disabled without pre-computed projections, the embedding view will be unavailable.",
)
@click.option(
    "--model",
    default=None,
    help="Model name for generating embeddings (e.g., 'Qwen/Qwen3-Embedding-0.6B').",
)
@click.option(
    "--trust-remote-code",
    is_flag=True,
    default=False,
    help="Allow execution of remote code when loading models from Hugging Face Hub.",
)
@click.option(
    "--batch-size",
    type=int,
    default=None,
    help="Batch size for processing embeddings (default: 32 for text, 16 for images). Larger values use more memory but may be faster.",
)
@click.option(
    "--x",
    "x_column",
    help="Column containing pre-computed X coordinates for the embedding view.",
)
@click.option(
    "--y",
    "y_column",
    help="Column containing pre-computed Y coordinates for the embedding view.",
)
@click.option(
    "--neighbors",
    "neighbors_column",
    help='Column containing pre-computed nearest neighbors in format: {"ids": [n1, n2, ...], "distances": [d1, d2, ...]}. IDs should be zero-based row indices.',
)
@click.option(
    "--sample",
    default=None,
    type=int,
    help="Number of random samples to draw from the dataset. Useful for large datasets.",
)
@click.option(
    "--umap-n-neighbors",
    type=int,
    help="Number of neighbors to consider for UMAP dimensionality reduction (default: 15).",
)
@click.option(
    "--umap-min-dist",
    type=float,
    help="The min_dist parameter for UMAP.",
)
@click.option(
    "--umap-metric",
    default=DEFAULT_UMAP_METRIC,
    help="Distance metric for UMAP computation (default: 'cosine').",
)
@click.option(
    "--umap-random-state", type=int, help="Random seed for reproducible UMAP results."
)
@click.option(
    "--duckdb",
    type=str,
    default="wasm",
    help="DuckDB connection mode: 'wasm' (run in browser), 'server' (run on this server), or URI (e.g., 'ws://localhost:3000').",
)
@click.option(
    "--host",
    default="localhost",
    help="Host address for the web server (default: localhost).",
)
@click.option(
    "--port", default=5055, help="Port number for the web server (default: 5055)."
)
@click.option(
    "--auto-port/--no-auto-port",
    "enable_auto_port",
    default=True,
    help="Automatically find an available port if the specified port is in use.",
)
@click.option(
    "--static", type=str, help="Custom path to frontend static files directory."
)
@click.option(
    "--export-application",
    type=str,
    help="Export the visualization as a standalone web application to the specified ZIP file and exit.",
)
@click.version_option(version=__version__, package_name="embedding_atlas")
def main(
    inputs,
    text: str | None,
    image: str | None,
    vector: str | None,
    split: list[str] | None,
    enable_projection: bool,
    model: str | None,
    trust_remote_code: bool,
    batch_size: int | None,
    x_column: str | None,
    y_column: str | None,
    neighbors_column: str | None,
    sample: int | None,
    umap_n_neighbors: int | None,
    umap_min_dist: int | None,
    umap_metric: str | None,
    umap_random_state: int | None,
    static: str | None,
    duckdb: str,
    host: str,
    port: int,
    enable_auto_port: bool,
    export_application: str | None,
):
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: (%(name)s) %(message)s",
    )

    df = load_datasets(inputs, splits=split, sample=sample)

    print(df)

    if enable_projection and (x_column is None or y_column is None):
        # No x, y column selected, first see if text/image/vectors column is specified, if not, ask for it
        if text is None and image is None and vector is None:
            text = prompt_for_column(
                df, "Select a column you want to run the embedding on"
            )
        umap_args = {}
        if umap_min_dist is not None:
            umap_args["min_dist"] = umap_min_dist
        if umap_n_neighbors is not None:
            umap_args["n_neighbors"] = umap_n_neighbors
        if umap_random_state is not None:
            umap_args["random_state"] = umap_random_state
        if umap_metric is not None:
            umap_args["metric"] = umap_metric
        # Use consistent default model and create model-specific column names
        if text is not None:
            effective_model = model if model is not None else DEFAULT_TEXT_MODEL
        elif image is not None:
            effective_model = model if model is not None else DEFAULT_IMAGE_MODEL
        else:
            effective_model = model if model is not None else DEFAULT_TEXT_MODEL
        
        # Run embedding and projection
        if text is not None or image is not None or vector is not None:
            from .projection import (
                compute_image_projection,
                compute_text_projection,
                compute_vector_projection,
            )

            mslug = model_slug(effective_model)
            
            x_column = find_column_name(df.columns, f"projection_x__{mslug}")
            y_column = find_column_name(df.columns, f"projection_y__{mslug}")
            if neighbors_column is None:
                neighbors_column = find_column_name(df.columns, f"__neighbors__{mslug}")
                new_neighbors_column = neighbors_column
            else:
                # If neighbors_column is already specified, don't overwrite it.
                new_neighbors_column = None
            if vector is not None:
                compute_vector_projection(
                    df,
                    vector,
                    x=x_column,
                    y=y_column,
                    neighbors=new_neighbors_column,
                    umap_args=umap_args,
                )
            elif text is not None:
                compute_text_projection(
                    df,
                    text,
                    x=x_column,
                    y=y_column,
                    neighbors=new_neighbors_column,
                    model=effective_model,
                    trust_remote_code=trust_remote_code,
                    batch_size=batch_size,
                    umap_args=umap_args,
                )
            elif image is not None:
                compute_image_projection(
                    df,
                    image,
                    x=x_column,
                    y=y_column,
                    neighbors=new_neighbors_column,
                    model=effective_model,
                    trust_remote_code=trust_remote_code,
                    batch_size=batch_size,
                    umap_args=umap_args,
                )
            else:
                raise RuntimeError("unreachable")

    # Define effective_model for cases where we didn't compute embeddings but still need it
    if 'effective_model' not in locals():
        effective_model = model if model is not None else DEFAULT_TEXT_MODEL

    id_column = find_column_name(df.columns, "_row_index")
    df[id_column] = range(df.shape[0])

    metadata = {
        "columns": {
            "id": id_column,
            "text": text,
        },
    }

    if x_column is not None and y_column is not None:
        metadata["columns"]["embedding"] = {
            "x": x_column,
            "y": y_column,
        }
        # Add embedding info like server does
        if text is not None or image is not None:  # Only add if we computed embeddings
            metadata["columns"]["embedding_info"] = {
                "model": effective_model
            }
    if neighbors_column is not None:
        metadata["columns"]["neighbors"] = neighbors_column

    hasher = Hasher()
    hasher.update(inputs)
    hasher.update(metadata)
    identifier = hasher.hexdigest()

    dataset = DataSource(identifier, df, metadata)

    if static is None:
        static = str((pathlib.Path(__file__).parent / "static").resolve())

    if export_application is not None:
        with open(export_application, "wb") as f:
            f.write(dataset.make_archive(static))
        exit(0)

    app = make_server(dataset, static_path=static, duckdb_uri=duckdb)

    if enable_auto_port:
        new_port = find_available_port(port, max_attempts=10, host=host)
        if new_port != port:
            logging.info(f"Port {port} is not available, using {new_port}")
    else:
        new_port = port
    uvicorn.run(app, port=new_port, host=host, access_log=False)


if __name__ == "__main__":
    main()
