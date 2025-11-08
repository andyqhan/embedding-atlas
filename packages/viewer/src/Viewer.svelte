<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import { coordinator as defaultCoordinator, Priority } from "@uwdata/mosaic-core";
  import { onMount } from "svelte";

  import EmbeddingAtlas from "./lib/EmbeddingAtlas.svelte";
  import Spinner from "./lib/Spinner.svelte";

  import type { DataColumns, DataSource } from "./data_source.js";
  import type { EmbeddingAtlasState } from "./lib/api.js";
  import { systemDarkMode } from "./lib/dark_mode_store.js";
  import { type ExportFormat } from "./lib/mosaic_exporter.js";
  import { debounce } from "./lib/utils.js";
  import { getQueryPayload, setQueryPayload } from "./query_payload.js";

  const coordinator = defaultCoordinator();

  interface Props {
    dataSource: DataSource;
  }

  // Minimal timeout wrapper - crashes loudly if operation takes too long
  function withTimeout<T>(promise: Promise<T>, timeoutMs: number, operation: string): Promise<T> {
    return Promise.race([
      promise,
      new Promise<never>((_, reject) =>
        setTimeout(() => reject(new Error(`${operation} timed out after ${timeoutMs}ms`)), timeoutMs)
      )
    ]);
  }

  let { dataSource }: Props = $props();

  let ready = $state(false);
  let error = $state(false);
  let status = $state("Loading...");
  let initialState: any | null = $state(null);
  let currentState: any | null = $state(null);
  let columns: DataColumns | null = $state(null);
  let rebuilding = $state(false);
  let selectedModel = $state("Qwen/Qwen3-Embedding-0.6B");

  let projectionKey = $derived.by(() => {
    const key = columns ? `${columns.embedding?.x ?? ""}:${columns.embedding?.y ?? ""}` : "";
    return key;
  });

  onMount(async () => {
    try {
      initialState = await getQueryPayload();
      
      // Load selectedModel from initialState if available
      if (initialState?.selectedModel) {
        selectedModel = initialState.selectedModel;
      }
      
      status = "Initializing database...";
      columns = await dataSource.initializeCoordinator(coordinator, "dataset", (s) => {
        status = s;
      });
      ready = true;

    } catch (e: any) {
      error = true;
      status = e.message;
      return;
    }
  });

  async function onExportSelection(predicate: string | null, format: ExportFormat) {
    if (dataSource.downloadSelection) {
      await dataSource.downloadSelection(predicate, format);
    }
  }

  async function onDownloadArchive() {
    if (dataSource.downloadArchive) {
      await dataSource.downloadArchive();
    }
  }

  async function onModelChange(model: string) {
    if (!dataSource.computeEmbeddings || !columns?.text) return;
    
    selectedModel = model;
    
    await onComputeEmbeddings(model, columns.text);
  }

  async function onComputeEmbeddings(model: string, textColumn: string) {
    if (!dataSource.computeEmbeddings) return;

    try {
      rebuilding = true;
      status = "Recomputing embeddings...";
      
      await dataSource.computeEmbeddings(model, textColumn);

      // Refresh the metadata so we pick up the new column names (`embedding.x/y`, `neighbors`)
      if ("metadata" in dataSource) {
        status = "Updating metadata...";
        const metadata = await (dataSource as any).metadata();
        columns = metadata.columns;
      }

      // Rebuild the DuckDB `dataset` table from fresh parquet so the new columns exist in the DB
      if ("serverUrl" in dataSource) {
        status = "Rebuilding database table...";
        const serverUrl = (dataSource as any).serverUrl as string;
        const ts = Date.now();
        const datasetUrl =
          serverUrl + (serverUrl.endsWith("/") ? "" : "/") + `dataset.parquet?t=${ts}`;

        await withTimeout(
          coordinator.exec(
            `
            CREATE OR REPLACE TABLE dataset AS
            (SELECT * FROM read_parquet('${datasetUrl}'))
            `,
            { priority: Priority.High }
          ),
          30000, // 30 second timeout
          "Table rebuild operation"
        );

      }

      // Clear caches so downstream queries/plots see the new schema and data
      status = "Refreshing components...";
      coordinator.clear();
      
      rebuilding = false;
      status = "Ready";
    } catch (error) {
      rebuilding = false;
      throw error;
    }
  }

  function onStateChange(state: EmbeddingAtlasState) {
    // Create a copy of the state without the embedding viewport to avoid persisting 
    // coordinates that are meaningless when the projection/model changes
    const stateWithoutEmbeddingViewport = {
      ...state,
      plotStates: state.plotStates ? {
        ...state.plotStates,
        "embedding-view": state.plotStates["embedding-view"] ? {
          ...state.plotStates["embedding-view"],
          viewportState: null  // Reset viewport when model changes
        } : undefined
      } : undefined
    };
    
    currentState = stateWithoutEmbeddingViewport;  // used for persisting UI state when model changes
    setQueryPayload({ ...state, predicate: undefined, selectedModel: selectedModel });
  }

</script>

<div class="fixed left-0 right-0 top-0 bottom-0">
  {#if ready && columns != null && !rebuilding}
    {#key projectionKey}
      <EmbeddingAtlas
        coordinator={coordinator}
        table="dataset"
        initialState={currentState || initialState}
        idColumn={columns.id}
        textColumn={columns.text}
        projectionColumns={columns.embedding}
        neighborsColumn={columns.neighbors}
        cache={dataSource.cache}
        automaticLabels={true}
        selectedModel={selectedModel}
        onExportApplication={dataSource.downloadArchive ? onDownloadArchive : null}
        onExportSelection={dataSource.downloadSelection ? onExportSelection : null}
        onComputeEmbeddings={dataSource.computeEmbeddings ? onModelChange : null}
        onStateChange={debounce(onStateChange, 200)}
      />
    {/key}
  {:else}
    <div
      class="w-full h-full grid place-content-center select-none text-slate-800 bg-slate-200 dark:text-slate-200 dark:bg-slate-800"
      class:dark={$systemDarkMode}
    >
      {#if error}
        <div class="text-red-600" style:max-width="36rem">{status}</div>
      {:else}
        <div class="w-72">
          <Spinner status={status} />
        </div>
      {/if}
    </div>
  {/if}
</div>
