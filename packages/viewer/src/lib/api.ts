// Copyright (c) 2025 Apple Inc. Licensed under MIT License.

// The component API for embedding viewer.

import type { CustomCell } from "@embedding-atlas/table";
import type { Coordinator } from "@uwdata/mosaic-core";
import { createClassComponent } from "svelte/legacy";

import Component from "./EmbeddingAtlas.svelte";

import cssCode from "../app.css?inline";

export interface EmbeddingAtlasProps {
  /** The Mosaic coordinator */
  coordinator: Coordinator;

  /** The name of the data table */
  table: string;

  /** The column for unique row identifiers */
  idColumn: string;

  /** The X and Y columns for the embedding projection view */
  projectionColumns?: { x: string; y: string } | null;

  /** The column for pre-computed nearest neighbors.
   * Each value in the column should be a dictionary with the format: `{ "ids": [id1, id2, ...], "distances": [distance1, distance2, ...] }`.
   * `"ids"` should be an array of row ids (as given by the `idColumn`) of the neighbors, sorted by distance.
   * `"distances"` should contain the corresponding distances to each neighbor.
   * Note that if `searcher.nearestNeighbors` is specified, the UI will use the searcher instead.
   */
  neighborsColumn?: string | null;

  /** The column for text. The text will be used as content for the tooltip and search features. */
  textColumn?: string | null;

  /** The color scheme. */
  colorScheme?: "light" | "dark" | null;

  /** The initial viewer state */
  initialState?: EmbeddingAtlasState | null;

  /** An object that provides search functionalities, including full text search, vector search, and nearest neighbor queries.
   * If not specified (undefined), a default full-text search with the text column will be used.
   * If set to null, search will be disabled. */
  searcher?: Searcher | null;

  /** Set to true to enable automatic labels for the embedding */
  automaticLabels?: boolean | null;

  /** A cache to speed up initialization of the viewer */
  cache?: Cache | null;

  /** Custom cell renderers for the table view */
  tableCellRenderers?: Record<string, CustomCell | "markdown">;

  /** A callback to export the currently selected points */
  onExportSelection?:
    | ((predicate: string | null, format: "json" | "jsonl" | "csv" | "parquet") => Promise<void>)
    | null;

  /** A callback to download the application as archive */
  onExportApplication?: (() => Promise<void>) | null;

  /** The currently selected embedding model */
  selectedModel?: string;

  /** A callback to compute new embeddings with a different model */
  onComputeEmbeddings?: ((model: string) => Promise<void>) | null;

  /** A callback when the state of the viewer changes. You may serialize the state to JSON and load it back. */
  onStateChange?: ((state: EmbeddingAtlasState) => void) | null;
}

export interface EmbeddingAtlasState {
  /** The version of Embedding Atlas that created this state */
  version: string;
  /** UNIX timestamp when this was created */
  timestamp: number;
  /** The view configuration and state of the embedding view */
  view?: any;
  /** The list of plots */
  plots?: { id: string; title: string; spec: any }[];
  /** The state of all plots */
  plotStates?: Record<string, any>;
  /** The selection predicate (SQL expression) */
  predicate?: string | null;
  /** The currently selected embedding model */
  selectedModel?: string;
}

export interface Cache {
  /** Gets an object from the cache with the given key. Returns `null` if the entry is not found. */
  get(key: string): Promise<any | null>;

  /** Sets an object to the cache with the given key */
  set(key: string, value: any): Promise<void>;
}

export interface Searcher {
  /** Perform a full text search with the given query */
  fullTextSearch?(
    query: string,
    options?: { limit: number; predicate: string | null; onStatus: (status: string) => void },
  ): Promise<{ id: any }[]>;

  /** Perform a vector search with the given query */
  vectorSearch?(
    query: string,
    options?: { limit: number; predicate: string | null; onStatus: (status: string) => void },
  ): Promise<{ id: any; distance?: number }[]>;

  /** Find nearest neighbors of the row of the given id */
  nearestNeighbors?(
    id: any,
    options?: { limit: number; predicate: string | null; onStatus: (status: string) => void },
  ): Promise<{ id: any; distance?: number }[]>;
}

export class EmbeddingAtlas {
  private component: any;
  private container: HTMLDivElement;
  private currentProps: EmbeddingAtlasProps;

  constructor(target: HTMLElement, props: EmbeddingAtlasProps) {
    this.currentProps = { ...props };

    // Container element
    this.container = document.createElement("div");
    this.container.style.display = "flex";
    this.container.style.width = "100%";
    this.container.style.height = "100%";
    target.appendChild(this.container);

    // Shadow root on container
    let shadowRoot = this.container.attachShadow({ mode: "open" });
    let style = document.createElement("style");
    style.innerText = cssCode;
    shadowRoot.appendChild(style);

    // Inner container element
    let innerContainer = document.createElement("div");
    innerContainer.style.display = "flex";
    innerContainer.style.width = "100%";
    innerContainer.style.height = "100%";
    shadowRoot.appendChild(innerContainer);

    // The Svelte component
    this.component = createClassComponent({ component: Component, target: innerContainer, props: props });
  }

  update(props: Partial<EmbeddingAtlasProps>) {
    let updates: Partial<EmbeddingAtlasProps> = {};
    for (let key in props) {
      if ((props as any)[key] !== (this.currentProps as any)[key]) {
        (updates as any)[key] = (props as any)[key];
        (this.currentProps as any)[key] = (props as any)[key];
      }
    }
    this.component.$set(updates);
  }

  destroy() {
    this.component.$destroy();
    this.container.remove();
  }
}
