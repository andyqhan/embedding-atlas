# Embedding Atlas - Codebase Summary

## Overview
Embedding Atlas is an interactive visualization tool for large embeddings, providing visualizations, cross-filtering, and search capabilities for embeddings and metadata. Built by Apple Inc., it supports up to millions of data points with WebGPU/WebGL rendering.

## Architecture

### Multi-Package Monorepo Structure
- **Root**: Workspace manager with build scripts and formatting tools
- **Frontend Packages**: TypeScript/Svelte components with WebGPU/WebGL renderers
- **Backend Package**: Python CLI tool and server
- **Documentation Package**: Static site with demos and guides

### Core Technologies
**Frontend:**
- Svelte 5+ for reactive UI components
- WebGPU (with WebGL2 fallback) for high-performance rendering
- Mosaic (UW Data Lab) for data coordination and SQL queries
- DuckDB WASM for client-side data processing
- Vite for build tooling

**Backend:**
- Python 3.10+ with FastAPI server
- Pandas for data manipulation
- UMAP for dimensionality reduction
- Sentence Transformers for embeddings
- DuckDB for data queries

**Algorithms:**
- Rust-based density clustering (WebAssembly)
- C++ UMAP implementation (WebAssembly)
- Kernel density estimation for contour visualization

## Key Components

### packages/component/
Core visualization components including EmbeddingView and rendering engines.
- **EmbeddingView**: Main scatter plot component with density visualization
- **WebGPU/WebGL Renderers**: High-performance point and density rendering
- **Density Clustering**: Automatic data clustering and labeling

### packages/viewer/
Complete application with data management, search, and multi-view coordination.
- **EmbeddingAtlas**: Full-featured viewer component
- **Data Management**: File upload, DuckDB integration, column inference
- **Search System**: Real-time embedding similarity search
- **Plot System**: Coordinated histogram, box plots, and other visualizations

### packages/table/
High-performance virtual scrolling table component with custom cell renderers.

### packages/backend/
Python package providing CLI tool and Jupyter widget.
- **CLI**: `embedding-atlas <dataset.parquet>` command
- **Server**: FastAPI backend for data serving
- **Widget**: Jupyter notebook integration

## Key Features
- Interactive embedding visualization with up to millions of points
- Automatic clustering and labeling of data regions
- Kernel density estimation with contour visualization
- Real-time search and nearest neighbor queries
- Multi-coordinated views for metadata exploration
- Order-independent transparency rendering
- Support for various data formats (Parquet, CSV, HuggingFace datasets)
- Python CLI tool, Jupyter widget, and npm package availability

## Application Modes

### Database Modes
- **WASM Mode** (`type: "wasm"`): DuckDB runs in browser via WebAssembly
- **REST Mode** (`type: "rest"`): Connect to remote DuckDB server via HTTP
- **WebSocket Mode** (`type: "socket"`): Real-time connection to DuckDB server

### Data Source Modes
Supports both text and images. The default text model is Qwen/Qwen3-Embedding-0.6B, and the default image model is google/vit-base-patch16-384.
- **Backend Data Source**: Loads from FastAPI server endpoints
- **Test Data Source**: Generates synthetic data for development
- **File Upload Mode**: Direct CSV/Parquet upload interface

### Rendering Modes
- **WebGPU Renderer**: GPU-accelerated rendering for modern browsers
- **WebGL2 Renderer**: Fallback for broader compatibility

### UMAP Execution Modes
- **WebAssembly UMAP**: Client-side C++ implementation compiled to WASM
- **Server-side UMAP**: Python-based computation on backend

### Deployment Modes
- **CLI Server Mode**: `embedding-atlas <dataset.parquet>` launches FastAPI server
- **Jupyter Widget Mode**: Embedded widget via anywidget
- **Standalone Export**: Self-contained ZIP with embedded data
- **Development Mode**: Hot reload with `npm run dev`

## Development Setup
- Monorepo managed with npm workspaces
- Build script: `./scripts/build.sh`
- Format checking: `prettier -c .`
- Individual packages have their own dev/build scripts

### Running in Development

**To start the backend server:**
```bash
cd packages/backend
uv run embedding-atlas <dataset.parquet> [options]
```

**To start the frontend development server:**
```bash
cd packages/viewer
npm run dev
```

The backend and frontend need to be run in separate terminal sessions. The backend serves the API and data, while the frontend development server provides hot reload for UI development.

