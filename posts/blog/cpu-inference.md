---
title: Making a CPU image classifier 2.4x faster
description: Instrumenting an image tagging service stage by stage, and why dynamic batching and INT8 quantization both lost to the plain baseline.
date: 2026-08-28
---

# Intro

I run MobileNetV2 behind a FastAPI endpoint. You POST an image, you get the top
3 labels back. The model runs on 2 CPU cores with a 2 GiB memory limit.

The load generator is k6 (which is a tool to test your services under load), open-loop with Poisson arrivals, so the arrival rate
does not slow down when the service does. All numbers below use that.

Basically, I got the pipeline from 87 ms to 36 ms per image, but also applied
two well-known optimizations that made things worse.

# The starting point

The first honest run was bad.

| metric | value |
|---|---|
| offered load | 8 RPS |
| end-to-end p99 | 3 587 ms |
| server-side p99 | 907 ms |

The gap between those two numbers is queueing. Requests spent most of their
life waiting, not computing. The service could not serve the load it was given.

The first useful step was not an optimization. It was instrumentation: a
Prometheus histogram per pipeline stage. It showed where the time went, per
image:

| stage | time | share |
|---|---|---|
| JPEG decode | 40.6 ms | 47% |
| preprocess | 6.0 ms | 7% |
| model inference | 40.3 ms | 46% |

Two findings here. Inference was only half the problem. And
no model optimization would ever touch the other half.

# Step 1: move storage writes off the request path

Every request wrote the image and its predictions to object storage before
answering. Two synchronous writes, both inside the latency budget. Moving them
to a background task removed them from the response time.

Super cheap without tradeoff.

# Step 2: ONNX Runtime

I exported the model to ONNX and served it with ONNX Runtime instead of just PyTorch.
Same weights and outputs. I verified label-for-label at several
batch sizes before trusting it.

There are 2 reasons why it worked.

Python leaves here. PyTorch eager returns to the interpreter between
every one of the ~150 operations in the graph. ONNX Runtime runs the whole
graph in C++.

The runtime also rewrites the graph at load time. It fuses `Conv + BatchNorm +
ReLU` into single kernels, and folds the batch-norm layers into the conv
weights entirely. Batch-norm at inference is "multiply by A, add B", and a
convolution is the same, so the two collapse into one.
It was written by smart people so we can trust it, especially since we compared it with the previous results.

Inference went from 43 ms to 15 ms per image. That is 2.8x, for free, with
identical predictions.

# Step 3: fix the input, not the decoder

Decode was still 40 ms. I tried faster decoders. The best alternative was 18%
faster on average and slower on large files.

JPEG decode cost tracks the compressed file size,
because every decoder must Huffman-decode the whole stream before it
can output a pixel. A 1.6 MB photo costs about 270 ms to decode no matter what
you decode it with. Scaling tricks help only a little: decoding at 1/8
resolution saved 38%, and the rest is irreducible.

The model only needs 224x224 pixels. Sending 12-megapixel photos to it is
waste that the server cannot undo. So the fix was changing the input policy. I
sized the workload like a client-side-resized upload, 0.2 to 1.1 megapixels.

I updated the k6s example images accordingly and decode dropped from 40.6 ms to 14.3 ms.

After steps 1 to 3:

| | before | after |
|---|---|---|
| pipeline per image | 87 ms | 36 ms |
| end-to-end p99 at the same load | 3 587 ms | 64 ms |

# What failed: dynamic batching

The idea is standard. Hold incoming requests for a few milliseconds, run them
through the model as one batch, split the results back to the callers. I built
it with an asyncio queue, one consumer task, and the built-in futures.

**The first attempt made everything worse.** p99 went from 64 ms to over
200 ms. The logs showed why. At the offered load, requests arrived about 140 ms
apart, and the batching window was 30 ms. Nearly every "batch" contained one
image.

Expected batch size is roughly `1 + RPS * window`. Every request paid the wait.
Almost none got a batch.

**The second attempt exposed a better bug.** I had put JPEG decode inside the
batch worker. That serialized every decode in the service through a single
thread, while a second core sat idle.

Decode had to move back into the request path, where each request decodes in
parallel. This works because PIL releases the GIL during decode, so it is real
parallelism and not turn-taking. That single fact is worth knowing: a C
extension can release the GIL while it works, and PIL, NumPy, PyTorch and ONNX
Runtime all do. Threading only helps if the work releases it.

The batch worker also needed its own dedicated thread. It shared a thread pool
with the decode calls, and under load the many producers starved the one
consumer. The queue grew, nothing drained it, and every request timed out at
once.

With all of that fixed and measured properly at 20 RPS, batches still averaged
under 2 images, and p99 was no better than the unbatched service.

The actual implmentation was okay. Batching is best when a batch of N costs
much less than N single calls. On a GPU that is true, on a CPU it is mostly false: 
batched inference scales near-linearly with
batch size, so there is nothing to optimize.

A small model makes it worse. MobileNetV2 fits in the CPU cache, and published
measurements show batch size 1 is often optimal for models this size.

# What failed numero dos: INT8 quantization

INT8 quantization stores weights and activations as 8-bit integers instead of
floats. A quarter of the memory traffic, and some CPUs have instructions built
for exactly this.

I quantized the ONNX graph and added an accuracy check to the build.
Quantization never raises an error, it just returns different answers, so
without a check a broken model ships silently.

The check flagged it straight away: only 81% of images kept the same top-1
label. That turned out to be a bad measure. On ordinary photos this model is
barely confident, with top scores around 0.03 to 0.15, so the top two labels
are within noise of each other and swap easily. Checking whether the original
winner stayed anywhere in the quantized top-3, which is what the API returns
anyway, was stable.

But it was not faster. Latency was the same and less stable. The quantized
graph wraps each operator in conversion nodes, and the runtime has to fuse
those into real int8 kernels. Where it cannot, you pay float math plus the
conversions on top. Without the right CPU instructions there is not much to win
in the first place.

So I removed it. The plain ONNX model is simpler to build, has stable latency
and exact accuracy.

# Where it landed

| | start | end |
|---|---|---|
| pipeline per image | 87 ms | 36 ms |
| inference per image | 43 ms | 15 ms |
| decode per image | 41 ms | 14 ms |
| sustainable load | ~11 RPS | ~35 RPS |

The final configuration is kinda boring. ONNX Runtime with graph optimizations,
sanely sized inputs, storage writes off the request path, thread counts matched
to the CPU limit.

The two clever techniques, batching and quantization, are the ones that did not
survive measurement.

# What I would tell someone starting something similar

**Instrument per stage before optimizing anything.** The stage histogram is
what showed that half the budget was JPEG decode, which no model work could
fix. Without it I would have spent the whole time on the model and moved
end-to-end latency by a third of what I expected.

**Measure with an open-loop load generator**, and treat dropped iterations as a
validity check. A closed-loop test quietly slows down along with the service
and reports flattering percentiles.

**Do the arithmetic first.** 2 CPU cores is 2 000 CPU-milliseconds per second.
Divide by the CPU cost of one request and that is your ceiling. No technique
moves you past it. Techniques only lower the cost per request.

**GPU advice does not transfer to CPUs by default.** Batching and INT8 are both
correct, standard and widely documented, and both lost to the boring baseline
on this hardware. The only way to know was to build them and measure.
