# Ragged Paged Attention (RPA)

This document explains the Ragged Paged Attention mechanism in Simply, covering
its core components, the lifecycle of a query, and how outputs are generated.

## Overview

Ragged Paged Attention is a memory-efficient attention mechanism designed for
high-throughput serving of Large Language Models (LLMs). It manages Key-Value
(KV) caches using non-contiguous memory "pages" (similar to OS virtual memory),
allowing flexible allocation and handling of ragged (variable-length) batches.

## Key Benefits

*   **Memory Efficiency**: Eliminates memory fragmentation by allocating KV
    cache in non-contiguous pages, similar to OS virtual memory.
*   **High Throughput**: Enables continuous batching and efficient
    variable-length processing (ragged attention), maximizing TPU utilization.
*   **Mixed Prefill and Decode**: Seamlessly batches new prompts (prefill) with
    ongoing generations (decode) in the same forward pass, reducing scheduling
    latency and improving overall system efficiency.
*   **Flexible Scaling**: Decouples logical sequence length from physical memory
    allocation, allowing for longer contexts and dynamic generation.

## Key Components

### 1. `DecodeState` (Physical Memory Management)

Defined in `utils/ragged_paged_attention.py`.

*   **Role**: Manages the physical KV cache memory.
*   **Structure**:
    *   `pages`: A large pre-allocated tensor storing all KV blocks
        `[total_num_pages, page_size, n_kv_heads * 2, head_dim]`.
    *   `page_indices`: Maps logical sequence positions to physical page indices
        `[batch_size, max_num_pages_per_seq]`.
    *   `available_page_indices`: A stack of free page indices.
*   **Key Operations**:
    *   `allocate`: Assigns new pages to sequences that need more space.
    *   `insert`: Writes new K/V embeddings into the allocated pages.
    *   `release`: Frees pages when a sequence finishes.
    *   `update_decode_state_and_compute_attn`: Updates decode state and
        computes attention.

### 2. `SamplingState` (Logical Batch Management)

Defined in `utils/ragged_paged_attention.py`.

*   **Role**: Manages the logical state of the current batch of requests.
*   **Structure**:
    *   `tokens`: Stores the generated tokens for each request.
    *   `position`: Current token position for each request.
    *   `input_lens`: Length of the prompt/prefix for each request.
    *   `rank`: Queue system to determine which requests get processed.
*   **Key Operations**:
    *   `push`: Adds new requests to the batch.
    *   `get`: Retrieves generated tokens and scores.
    *   `release`: Removes finished requests from the batch.

### 3. `Batcher` & `SimplyService` (Orchestration)

Defined in `serving/page_server.py`.

*   **Role**: Handles the server loop, gRPC interface, and coordination between
    the model and the RPA state.
*   **Structure**:
    *   `request_queue`: Buffers incoming user queries.
    *   `loop`: The main background thread that continuously runs the decoding
        steps.

## Life of a Query

Here describes how a query travels through the system from request to
completion.

### 1. Request Arrival

A client sends a `Run` request via gRPC to `SimplyService`.

*   The request is put into `Batcher.request_queue`.
*   A `Future` is created to await the result.

### 2. Batch Scheduling & Ingestion

The `Batcher.loop` continuously checks for new requests and free slots in the
`SamplingState`.

*   **Check Availability**: If there are free slots (i.e., `is_pad_seq` is true
    for some batch indices), new requests are popped from the queue.
*   **Tokenization**: The prompt text is tokenized (via `input_processor`).
*   **Push to State**: `SamplingState.push` assigns the request to a specific
    batch index (slot).
    *   It initializes `tokens`, `input_lens`, and resets `position` to 0.
    *   The request enters the decoding queue.

### 3. Execution Phase (The Decoding Loop)

The Loop runs `compiled_decode_fn` (wrapping `model.apply` and RPA logic)
repeatedly.

*   **Token Selection (`issue_lens`)**:

    *   Before running the model, `SamplingState.issue_lens` determines how many
        tokens to process for each sequence.
    *   **Priority**: It respects the `rank` (requests with smaller rank indices
        are prioritized).
    *   **Capacity**: It fills the available batch capacity (e.g.,
        `max_num_issue_tokens`) greedily.
    *   **Mixed Batching**: It automatically handles both prefill (issuing all
        prompt tokens) and decoding (issuing 1 token) in the same step.
    *   **Ragged Gathering**: The selected tokens are gathered into a ragged
        tensor via `ragged_issue_tokens`, forming the actual input for the
        forward pass.

*   **KV Allocation**:

    *   `DecodeState.allocate` checks if sequences need new pages for the next
        token(s) and grabs them from `available_page_indices`.

*   **Model Forward Pass**:

    *   The model computes Query, Key, and Value embeddings.
    *   **KV Insertion**: `DecodeState.insert` writes the new K/V data into the
        physical pages mapped by `page_indices`.
    *   **Attention**: `ragged_paged_attention` kernel computes attention using
        the paged KV cache.

*   **Sampling**:

    *   The model outputs logits.
    *   New tokens are sampled (greedy, top-k, etc.).
    *   `SamplingState` updates the `tokens` array with the new token.
    *   `position` is incremented.

### 4. Output Generation & Completion

After each step, the `Batcher` checks for completion.

*   **Check Status**: `completed_mask` identifies sequences that have hit the
    EOS token or max length.
*   **Retrieval**: `SamplingState.get(completed_mask)` extracts the full
    generated sequence:
    *   It slices `tokens` based on `input_len` and current `position`.
    *   Decodes token IDs back to text.
*   **Response**: The result text is set on the `Future`, completing the gRPC
    call for the client.

### 5. Cleanup

*   **Release**: `SamplingState.release` (and `DecodeState.release`) is called
    for completed sequences.
    *   Physical pages are returned to `available_page_indices`.
    *   The batch slot is marked as padding (`is_pad_seq=True`), making it
        available for a new request.
