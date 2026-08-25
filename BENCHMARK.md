
## Benchmark

### SW raid0 including 7 NVMe PCIe 4 disks (56GB/s)

| Model                         | safetensors | fastsafetensors | instanttensor (URING) | instanttensor (AIO) | Notes                                                                                                                                                                                                                                                                      |
|-------------------------------|-------------|-----------------|-----------------------|---------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Qwen3-32B                     | 14.78s      | 15.53s          | 2.16s                 | 2.41s               |
| Gemma-4-26B                   | 11.73s      | OOM             | 1.78s                 | 2.36s               | See below for OOM analysis                                                                                                                                                                                                                                                 |
| Gemma-4-26B (TP=2)            | 10.47s      | 5.23s           | 1.76s                 | 2.31s               |                                                                                                                                                                                                                                                                            |
| GPT-oss-120b (TP=4)           | 8.74s       | 2.54s           | 2.45s                 | 2.87s               |
| DeepSeek-v4-Flash-0731 (TP=4) | 15.36s      | 17.01s          | 10.21s                | 11.01s              |                                                                                                                                                                                                                                                                            |
| Inkling-Small-FP8-Dynamic     | N/A         | N/A             | N/A                   | N/A                 | Using upstream vLLM 0.27.1 - [Errored](test/models/inkling-small-fp8-dynamic/vllm.log) - ValueError: All MoE projections need to have same quantization scheme but found multiple                                                                                          |
| Inkling-NVFP4 (TP=8)          | 213.18s     | 44.07s          | OOM                   | OOM                 | Using upstream vLLM 0.27.1 - OOM after loading weights - H100 also does not support NVFP4 - InstantTensor [OOM while allocating buffer](test/models/inkling-nvfp4/instanttensor-default.log)                                                                               |
| Inkling-Small                 | 146.34s     | OOM             | OOM                   | OOM                 | Using upstream vLLM 0.27.1. [Fasttensors fails like for Gemma](test/models/inkling-small/fasttensors.log). The shard size is bigger than the available memory after pre-allocation. InstantTensor also fails with OOM when allocating its buffer similar to Inkling-NVFP4. |


### Single NVMe PCIe 4 disk (8GB/s)

| Model                         | safetensors | fastsafetensors | instanttensor (URING) | instanttensor (AIO) | Notes                      |
|-------------------------------|-------------|-----------------|-----------------------|---------------------|----------------------------|
| Gemma-4-26B (TP=2)            | 10.28s      | 5.24s           | 7.45s                 | 7.45s               |                            |
| GPT-oss-120b (TP=4)           | 8.78s       | 2.75s           | 10.52s                | 10.53s              |
| DeepSeek-v4-Flash-0731 (TP=4) | 15.36s      | 15.80s          | 24.59s                | 25.70s              |                            |

# Gemma-4-26B-A4B-it Loading Failure with `fastsafetensors`

## Summary

Loading `Gemma-4-26B-A4B-it` in BF16 on a single 80 GB NVIDIA H100 fails when vLLM is configured with `--load-format fastsafetensors`. The model itself is small enough to fit in GPU memory, but the `fastsafetensors` loading path introduces a large temporary GPU allocation tied to the size of an individual checkpoint shard.

The observed failure is therefore a **peak-memory problem during model loading**, rather than an insufficient steady-state memory capacity for the model.

The `fastsafetensors` implementation loading granularity is an entire `.safetensors` file.

For each checkpoint shard assigned to a rank, the loader creates a device buffer large enough to contain the complete file. The tensors contained in that file can then be exposed, distributed or copied to their final destinations.

This means that temporary GPU memory consumption depends directly on **checkpoint file size**.

The `fastsafetensors` maintainers describe the current behavior as requiring each rank to load an entire safetensors shard into a GPU device buffer before proceeding. Consequently, peak additional GPU memory is at least the size of one checkpoint shard, with potentially further temporary allocations depending on the operation being performed.

For models whose checkpoints are split into relatively small files, such as 2–5 GB each, this overhead is normally manageable. It becomes problematic when a model contains a very large individual shard.

## Observed Behaviour

The deployment uses vLLM `0.24.0+rhaiv.9`, BF16, tensor parallelism of 1, and explicitly selects `load_format='fastsafetensors'`.

During startup, vLLM reaches:

```text
Starting to load model /mnt/models...
Loading fastsafetensors checkpoint shards: 0/2
```

and then immediately fails with:

```text
CUDA out of memory. Tried to allocate 46.48 GiB.
GPU 0 has a total capacity of 79.18 GiB
of which 29.81 GiB is free.
...
48.65 GiB is allocated by PyTorch
```

The `0/2` progress indicator does not mean that no GPU memory has been allocated. At this point vLLM has already constructed the model and allocated approximately 48.65 GiB of GPU memory for its parameters. It then begins loading the checkpoint data into those destination parameters.

## Root Cause

The Gemma checkpoint consists of two very unevenly sized weight files. The first is approximately 49.9 GB while the second is only about 1.7 GB.

A 49.9 GB file corresponds to approximately:

```text
49.9 GB ≈ 46.5 GiB
```

This almost exactly matches the failed CUDA allocation:

```text
Tried to allocate 46.48 GiB
```

The resulting loading sequence is therefore approximately:

```text
1. vLLM creates Gemma model
                         GPU usage ≈ 48.65 GiB

2. fastsafetensors starts shard 1

3. Allocate device buffer for shard
                         requested ≈ 46.48 GiB

4. Required peak:
   48.65 + 46.48 + overhead
                         ≈ 95.8 GiB

5. H100 capacity
                         ≈ 79.18 GiB

6. CUDA OOM
```

The logs show the allocation failing immediately after `fastsafetensors` begins processing the first checkpoint file.

This behavior is consistent with the known file-level granularity of the current `fastsafetensors` implementation.

The issue is therefore not that Gemma requires 95 GiB during normal inference. Rather, the loading path temporarily requires both:

```text
resident model parameters
            +
one complete checkpoint-file buffer
```

For this checkpoint, the first file is almost as large as the entire model.

## Why `fastsafetensors` Normally Helps

The extra temporary memory is a trade-off for faster loading.

Traditional loaders can process tensors more incrementally, which limits temporary GPU allocations but may not fully utilize very fast local NVMe or distributed storage.

`fastsafetensors` instead attempts to maximize I/O throughput through asynchronous transfers, GPU-side operations, and, where available, GPUDirect Storage. The project reports substantial model-loading speed improvements in appropriate storage and hardware configurations.

Therefore, `fastsafetensors` itself is not inherently unsuitable for an 80 GB H100. The particular problem here is the combination of:

```text
~49 GiB BF16 model
+
~46.5 GiB individual checkpoint shard
+
80 GiB GPU
```

If the same weights were distributed across significantly smaller files, `fastsafetensors` could retain its faster-loading architecture without generating such a large transient allocation.

## Mitigation

The preferred mitigation while retaining `fastsafetensors` is to **reshard the checkpoint into smaller safetensors files**, for example 4–8 GB per shard.

Instead of:

```text
Shard 1                         ~46.5 GiB
Shard 2                          ~1.6 GiB
```

a resharded checkpoint might have:

```text
Shard 1                           ~4 GiB
Shard 2                           ~4 GiB
...
Shard N                           ~4 GiB
```

The approximate peak would then become:

```text
Resident model                  ~48.7 GiB
fastsafetensors device buffer    ~4.0 GiB
other overhead                    ~1 GiB
                                ----------
Peak                            ~54 GiB
```

This preserves the model weights and inference characteristics while directly addressing the cause of the loading OOM.

Other alternatives include:

* using multiple GPUs with tensor parallelism;
* CPU weight offloading;
* using a pre-quantized checkpoint;

# Inkling-NVFP4: InstantTensor loading failure

Loading `Inkling-nvfp4` with vLLM 0.27.1 using `--load-format instanttensor` fails during InstantTensor initialization, before the model weights are loaded.

The deployment uses 8-way tensor parallelism on H100 GPUs. InstantTensor selects the `URING` backend on all ranks and attempts to allocate an internal GPU I/O buffer.

## Failure

InstantTensor initially requests an I/O depth of 64 but reduces it to 49 because of available GPU memory:

```text
Shrink io_depth from 64 to 49 due to memory limit.
```

Immediately afterwards, all ranks fail while allocating the loader's device buffer:

```text
cudaMalloc(&this->device_buffer, this->buffer_size) -> 2:out of memory
```

Unlike the safetensors run, no checkpoint shards are successfully loaded. The failure therefore occurs in InstantTensor's GPU staging-buffer allocation rather than during model inference or KV-cache initialization.

