# Embedding Atlas Backend API Documentation

## Overview
The Embedding Atlas backend provides a FastAPI server that serves embedding data and supports interactive visualization through REST API endpoints. The server handles data querying, caching, export functionality, and serves the frontend application.

## Running the Backend

To start the backend server:

```bash
cd packages/backend
uv run embedding-atlas <dataset.parquet> [options]
```

**Example:**
```bash
cd packages/backend
uv run embedding-atlas ~/data/my_dataset.parquet --text content
```

**Common Options:**
- `--text <column>`: Column containing text data to embed
- `--image <column>`: Column containing image data to embed
- `--model <model_name>`: Specify embedding model (default: Qwen3-Embedding-0.6B for text)
- `--port <port>`: Server port (default: 5055)
- `--host <host>`: Server host (default: localhost)

For development with the frontend, run this in a separate terminal from the frontend dev server.

## API Endpoints

### Data Endpoints

#### `GET /data/dataset.parquet`
**Description**: Downloads the dataset in Parquet format  
**Response**: Binary Parquet file containing the complete dataset  
**Content-Type**: `application/octet-stream`  
**Features**: Supports HTTP Range requests for partial downloads

#### `GET /data/metadata.json`
**Description**: Retrieves dataset metadata including column information and database configuration  
**Response**: JSON object containing:
- Dataset metadata (columns, identifiers, etc.)
- Database configuration (`type`: "wasm", "rest", or "socket")
- Connection URIs for external DuckDB instances

**Example Response**:
```json
{
  "columns": {
    "id": "_row_index",
    "text": "content",
    "embedding": {
      "x": "projection_x",
      "y": "projection_y"
    },
    "neighbors": "__neighbors"
  },
  "database": {
    "type": "wasm",
    "load": true
  }
}
```

### Caching Endpoints

#### `POST /data/cache/{name}`
**Description**: Stores data in server-side cache with specified name  
**Parameters**:
- `name` (path): Cache key identifier
**Body**: JSON data to cache  
**Response**: Empty response on success

#### `GET /data/cache/{name}`
**Description**: Retrieves cached data by name  
**Parameters**:
- `name` (path): Cache key identifier
**Response**: JSON object if found, 404 if not found

### Query Endpoints

#### `GET /data/query`
**Description**: Executes SQL queries on the dataset (GET method with query parameter)  
**Parameters**:
- `query` (query param): URL-encoded JSON query object
**Query Object Format**:
```json
{
  "sql": "SELECT * FROM dataset LIMIT 10",
  "type": "arrow|json|exec"
}
```
**Response**: 
- `arrow`: Binary Arrow format (`application/octet-stream`)
- `json`: JSON array of records (`application/json`)
- `exec`: Empty response for execution-only queries

#### `POST /data/query`
**Description**: Executes SQL queries on the dataset (POST method with JSON body)  
**Body**: Query object (same format as GET version)  
**Response**: Same as GET version

#### `POST /data/selection`
**Description**: Exports selected data based on predicate filtering  
**Body**:
```json
{
  "predicate": "column_name > 100", // Optional SQL WHERE clause
  "format": "json|jsonl|csv|parquet"
}
```
**Response**: Binary data in requested format (`application/octet-stream`)

#### `POST /data/embedding`
**Description**: Computes new embeddings using a different model and updates the dataset with new projections  
**Body**:
```json
{
  "model": "Qwen3-Embedding-0.6B",
  "text_column": "content",
  "trust_remote_code": false,
  "batch_size": 32,
  "umap_args": {
    "n_neighbors": 15,
    "min_dist": 0.1,
    "metric": "cosine",
    "random_state": 42
  }
}
```
**Response**: 
- Success (200): JSON with new column information
- Error (400/500): JSON error object

**Example Success Response**:
```json
{
  "success": true,
  "message": "Embeddings computed successfully using model 'Qwen3-Embedding-0.6B'",
  "columns": {
    "x": "projection_x",
    "y": "projection_y",
    "neighbors": "__neighbors",
    "text": "content"
  }
}
```

**Features**:
- Uses HuggingFace SentenceTransformers models
- Leverages existing caching system for performance
- Updates dataset and metadata in real-time
- Validates model and parameter inputs
- Runs asynchronously for large datasets

### Export Endpoints

#### `GET /data/archive.zip`
**Description**: Creates a downloadable ZIP archive containing the complete visualization application with data  
**Response**: ZIP file containing static frontend files and embedded dataset  
**Content-Type**: `application/zip`  
**Use Case**: Standalone deployment of the visualization

### Static File Serving

#### `GET /` (and all other paths)
**Description**: Serves the frontend application static files  
**Response**: HTML, CSS, JS, and other static assets for the web interface  
**Features**: Single Page Application (SPA) routing support

## Database Integration

The server creates an in-memory DuckDB instance and loads the dataset into a table named `dataset`. This enables:
- SQL querying capabilities
- Efficient data filtering and aggregation
- Support for complex analytical operations
- Arrow format output for high-performance data transfer

## Configuration Options

The server can be configured with different DuckDB modes:
- **WASM Mode** (`duckdb=wasm`): DuckDB runs in the browser
- **Server Mode** (`duckdb=server`): DuckDB runs on the backend server
- **Remote Mode** (`duckdb=ws://...` or `duckdb=http://...`): Connect to external DuckDB instance

## Error Handling

All query endpoints return error responses in the format:
```json
{
  "error": "Error message description"
}
```
with HTTP status code 500 for execution errors.

## Performance Features

- **Concurrent Query Execution**: Queries run in a thread pool executor
- **Content Caching**: LRU cache for dataset and content generation
- **Range Request Support**: Efficient partial file downloads
- **Connection Pooling**: Reused DuckDB connections for query performance