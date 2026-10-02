# Input gradients with frozen ALICE

`estimate_mi_fast(..., differentiable=True)` returns a scalar tensor in nats.
With `return_std=True`, it returns two scalar tensors: MI and Monte Carlo
standard error. The latter measures Monte Carlo sampling variability, not
estimator bias, accuracy, posterior uncertainty, or coverage. Its derivative
is defined as zero when the estimated variance is nonpositive.

Call `model.eval().requires_grad_(False)` first. Evaluation mode disables
training behavior such as dropout; freezing parameters prevents weight-gradient
computation. Neither disables derivatives with respect to inputs. The estimator
validates these conditions without changing model flags, calling backward, or
writing parameter `.grad` fields. The caller owns optimization.

Rebuild simulator samples and call the estimator again on every optimization
step. Each invocation builds its own support cache and graph, which ordinary
backward consumes. No `retain_graph=True` is needed. Do not mutate model state,
inputs, or buffers between forward and backward. Fixed context is supported;
its cache is built outside inference mode so query autograd can save it safely.

## Gradient modes and caching

An enclosing `torch.no_grad()` or `torch.inference_mode()` raises `ValueError`
for differentiable calls. An enclosing `torch.enable_grad()` can override
`no_grad()`, but it cannot override `inference_mode()`. Samples created under
inference mode are also rejected; construct simulator inputs outside it.
A graph detached by the caller before estimation cannot be restored.
Normal calls still run under inference mode and return Python floats.

For the recognized `InducedGroupICLDenoiserModel` / `QueryDecoder` architecture,
ALICE calls its existing `_encode_support`, `_encode_latents`, `_encode_query`,
and `_decode_query` components directly. This preserves the relation graph,
support states, latents and query derivatives even when the checkpoint's cache
entry points carry `@torch.no_grad()`. **No checkpoint-code update is needed**
for this architecture. Projection caching uses the existing attention weights
and retains gradients through projected keys and values.

Other architectures fall back to ordinary `forward`, even when `use_cache=True`.
A checkpoint author can opt into caching by setting
`supports_input_grad_cache = True`, guaranteeing that both `_support_cache`
and `predict_query_logits_cached` preserve context and query autograd without
`no_grad`, inference mode, or detachment. Such checkpoints may need a code
revision; pin and review it. A detached query-logit result raises an error when
inputs require gradients. This check cannot prove the absence of partial
internal detachment; the checkpoint's autograd contract remains necessary.

`prepare_mi_model` adapters can be reused, but their context caches are rebuilt
for each call. Compiled forward/backward support depends on the compiler and
device. Full-graph capture is tested with the `eager` backend; Inductor input
backward is not certified by that test.

## Memory and recomputation

`checkpoint_queries=True` uses [PyTorch non-reentrant checkpointing](https://docs.pytorch.org/docs/2.14/checkpoint.html)
for each query chunk, including all masked velocities, zero-time corrections
and the integrand. Context and cache objects are explicit checkpoint inputs.
Row splitting, times, and shared Gaussian noise are sampled before these
regions; backward reuses the same draws and does not rebuild or mutate caches.

With caching enabled, support encodings, relation graphs, projected keys/values
and their encoding graphs remain resident. Query intermediate activations are
recomputed. Support encoding itself is not checkpointed. With `use_cache=False`,
support computation occurs inside each query checkpoint and is recomputed too.
The normalized samples, noised evaluation rows and integration times also
remain resident. Smaller `chunk` alone limits forward working memory, but does
not discard activations retained by autograd across chunks. Turning off query
fusion can further reduce each chunk's working set.

## Normalization and numerical precision

The disjoint context/evaluation split is unchanged. With `normalize=True`,
`differentiable=True` uses a smooth Gaussian-kernel CDF fitted only on context
rows. For context column variance `v` (population variance), the bandwidth is
`h = sqrt(0.2**2 * v + 0.001**2)`. Each value `x` maps to
`Phi^-1(1e-4 + (1 - 2e-4) * mean_i Phi((x - context_i) / h))`.
Gradients flow through context, evaluation values and the fitted bandwidth;
there are no detached fits or straight-through derivatives. The positive
bandwidth floor handles constant columns and singleton contexts. Probability
shrinkage keeps quantiles finite, although numerical tails can still saturate.

Normalization preserves FP64 and otherwise computes and returns FP32. Pairwise
CDF tensors are computed in chunks of 128 query rows and checkpointed whenever
inputs require gradients, independently of `checkpoint_queries`. This bounds
retained activations but still requires quadratic work in the context size.

Ordinary inference keeps hard empirical Gaussian-copula normalization. Smooth
normalization changes the transform used during pretraining, so normalized MI
values can differ between differentiable and inference calls; smoothness alone
does not establish estimator accuracy or optimization quality. Use
`normalize=False` to bypass both transforms and supply your own preprocessing.

Model operations retain the model dtype. Differentiable integration/reduction
uses FP32 for FP16/BF16 samples and otherwise preserves the sample dtype.
FP32 samples and design variables are useful even with low-precision weights;
internal low-precision backward can still underflow or overflow. The caller
may scale the objective before backward and unscale the resulting gradients.
There is no automatic gradient scaling. Normal inference's reduction dtype is
unchanged. The MI formula, integration interval, shared noise and random draw
order are unchanged. Seeded comparisons require the same generator state,
model, dtype, device and estimator options; cross-device bitwise equivalence
is not promised.

Validation on PyTorch 2.14.0 exercises CPU FP64, FP32, FP16 and BF16, and
MPS FP32, FP16 and BF16. MPS does not support FP64. CUDA tests are included
but were skipped because CUDA hardware was unavailable.
Higher-order derivatives and parameter training are outside this API contract.

The pinned small-checkpoint tests use `atol=rtol=1e-4` for FP32 values and
`2e-4` for gradients with matched fused layouts. Changing fusion changes GEMM
row counts; its FP32 gradient comparison uses relative L2 error below `1e-3`.
An FP64 fused/unfused comparison uses `atol=rtol=1e-10`. Low-precision cached
versus uncached gradients use relative L2 error below four machine epsilons
(about `0.0039` for FP16 and `0.0313` for BF16). Cache sharing sums adjoints
before support backward, while uncached evaluation sums after it, changing
low-precision rounding. These are regression tolerances, not guarantees of
gradient accuracy on arbitrary data.
