<!-- Copyright (c) 2025 Apple Inc. Licensed under MIT License. -->
<script lang="ts">
  import { coordinator as defaultCoordinator } from "@uwdata/mosaic-core";
  import { onMount } from "svelte";

  import EmbeddingAtlas from "./lib/EmbeddingAtlas.svelte";
  import Spinner from "./lib/Spinner.svelte";

  import type { DataColumns, DataSource } from "./data_source.js";
  import type { EmbeddingAtlasState } from "./lib/api.js";
  import { systemDarkMode } from "./lib/dark_mode_store.js";
  import { type ExportFormat } from "./lib/mosaic_exporter.js";
  import { debounce } from "./lib/utils.js";
  import { getQueryPayload, setQueryPayload } from "./query_payload.js";

  import * as SQL from "@uwdata/mosaic-sql";

  const coordinator = defaultCoordinator();

  interface Props {
    dataSource: DataSource;
  }

  let { dataSource }: Props = $props();

  let ready = $state(false);
  let error = $state(false);
  let status = $state("Loading...");
  let initialState: any | null = $state.raw(null);
  let columns: DataColumns | null = $state.raw(null);
  let dataRefreshKey = $state(0);

  onMount(async () => {
    try {
      initialState = await getQueryPayload();
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

  async function onComputeEmbeddings(model: string, textColumn: string) {
    if (dataSource.computeEmbeddings && 'metadata' in dataSource) {
      console.log(`Starting embedding computation with model: ${model}`);
      
      await dataSource.computeEmbeddings(model, textColumn);
      console.log("Backend embedding computation completed");
      
      // Refresh the metadata to get updated column information
      const metadata = await (dataSource as any).metadata();
      console.log("Old columns:", columns);
      columns = metadata.columns;
      console.log("New columns:", columns);
      console.log("Updated metadata:", metadata);
      
      // Reload the dataset to reflect the new embeddings
      const oldFirstRow = await coordinator.query(SQL.Query.from("dataset").select("*").limit(1));
      const oFR = oldFirstRow.get(0);
      console.log("First row of OLD dataset:", {
        projection_x: oFR.projection_x,
        projection_y: oFR.projection_y
      });

      if ('serverUrl' in dataSource) {
        const serverUrl = (dataSource as any).serverUrl;
        const timestamp = Date.now();
        const datasetUrl = serverUrl + (serverUrl.endsWith('/') ? '' : '/') + `dataset.parquet?_t=${timestamp}`;
        
        console.log(`About to reload dataset from: ${datasetUrl}`);
        console.log(`Timestamp: ${new Date().toISOString()}`);
        
        // Test: Fetch the parquet data directly via JavaScript to verify it's different
        console.log("Fetching parquet data directly via fetch() to verify...");
        const response = await fetch(datasetUrl, {
          cache: 'no-store',
          headers: {
            'Cache-Control': 'no-cache'
          }
        });
        const arrayBuffer = await response.arrayBuffer();
        console.log(`Fetched parquet data: ${arrayBuffer.byteLength} bytes`);
        console.log(`Response headers:`, Object.fromEntries(response.headers.entries()));
        
        // Hash the first 100 bytes to see if content actually differs
        const first100Bytes = new Uint8Array(arrayBuffer.slice(0, 100));
        const hashArray = Array.from(first100Bytes.slice(0, 20));
        console.log(`First 20 bytes: [${hashArray.join(', ')}]`);
        
        // Add a small delay to ensure backend cache is cleared
        await new Promise(resolve => setTimeout(resolve, 100));
        
        // Simple approach: Just reload the table
        console.log("Dropping and recreating table...");
        await coordinator.exec(`DROP TABLE IF EXISTS dataset;`);
        await coordinator.exec(`CREATE TABLE dataset AS (SELECT * FROM read_parquet('${datasetUrl}'));`);
        
        console.log("Dataset reloaded, checking first row...");
        const newFirstRow = await coordinator.query(SQL.Query.from("dataset").select("*").limit(1));
        const nFR = newFirstRow.get(0);
        console.log("First row of NEW dataset:", {
          projection_x: nFR.projection_x,
          projection_y: nFR.projection_y
        });
        
        // Check if data actually changed by sampling a few more rows
        const sampleRows = await coordinator.query(SQL.Query.from("dataset").select("*").limit(5));
        console.log("Sample of NEW data (5 rows):");
        for (let i = 0; i < Math.min(5, sampleRows.numRows); i++) {
          const row = sampleRows.get(i);
          console.log(`Row ${i}: projection_x=${row.projection_x}, projection_y=${row.projection_y}`);
        }
        
        // Just clear coordinator state without forcing component unmount
        console.log("Clearing coordinator state...");
        console.log("columns:", columns);
        console.log("projectionColumns:", columns?.embedding);
        
        // Clear any cached state in the coordinator/database without forcing component refresh
        if (coordinator.clear) {
          await coordinator.clear();
        }
        
        console.log("Coordinator clear completed");
        
        // Try to force refresh by briefly modifying projectionColumns to trigger re-render
        const originalEmbedding = columns?.embedding;
        const originalId = columns?.id;
        if (columns && columns.embedding) {
          // Refresh both embedding view and table by temporarily nulling key columns
          columns = { ...columns, embedding: null, id: null };
          await new Promise(resolve => setTimeout(resolve, 10));
          columns = { ...columns, embedding: originalEmbedding, id: originalId };
          console.log("Forced projectionColumns and table refresh");
        }
        
        // Test: Wait a bit longer and check if the visualization eventually updates
        console.log("Waiting 2 seconds to see if visualization eventually updates...");
        await new Promise(resolve => setTimeout(resolve, 2000));
        
        // Check the data one more time to confirm it's still correct
        try {
          const finalCheck = await coordinator.query(SQL.Query.from("dataset").select("*").limit(1));
          const finalRow = finalCheck.get(0);
          console.log("Final data check after 2 seconds:", {
            projection_x: finalRow.projection_x,
            projection_y: finalRow.projection_y
          });
        } catch (finalCheckError) {
          console.error("Error during final data check:", finalCheckError);
        }
        
        console.log("Embedding update completed!");
      }
    }
  }

  function onStateChange(state: EmbeddingAtlasState) {
    setQueryPayload({ ...state, predicate: undefined });
  }
</script>

<div class="fixed left-0 right-0 top-0 bottom-0">
  {#if ready && columns != null}
    <EmbeddingAtlas
      coordinator={coordinator}
      table="dataset"
      initialState={initialState}
      idColumn={columns.id}
      textColumn={columns.text}
      projectionColumns={columns.embedding}
      neighborsColumn={columns.neighbors}
      cache={dataSource.cache}
      automaticLabels={true}
      onExportApplication={dataSource.downloadArchive ? onDownloadArchive : null}
      onExportSelection={dataSource.downloadSelection ? onExportSelection : null}
      onComputeEmbeddings={dataSource.computeEmbeddings ? onComputeEmbeddings : null}
      onStateChange={debounce(onStateChange, 200)}
    />
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
