# device_utils

Safe torch-device resolution that survives GPU compute-capability
mismatches instead of crashing on the first kernel launch.

## Why this module exists

`torch.cuda.is_available()` only checks that CUDA is compiled in and
that *some* CUDA-capable device is visible. It does **not** check that
PyTorch was built with kernels for the device's compute capability.
A rig with a GTX 1060 (sm_61) running a modern torch build that only
ships kernels for sm_75+ will:

1. Report `torch.cuda.is_available() == True`.
2. Accept `tensor.cuda()` without complaint.
3. Crash on the first forward pass with
   `torch.AcceleratorError: no kernel image is available for execution
   on the device`.

Benchmarks that had been running for half an hour then died when the
first embedding call hit the GPU. `device_utils.safe_device()` closes
that trap up front.

## Public surface

```python
from device_utils import safe_device, clear_cache, torch_device_env_disabled
```

### `safe_device(index: int | None = None, *, warn: bool = True) -> str`

Returns one of `"cpu"`, `"cuda"`, or `"cuda:{index}"`. The decision
tree:

1. **CUDA unavailable** → `"cpu"`. No warning (expected on CPU-only
   hosts).
2. **Index explicitly requested** → probe that device's compute
   capability; if not present in `torch.cuda.get_arch_list`, warn once
   to stderr (gated by `warn=False` for tests) and fall back to `"cpu"`.
3. **Default device requested** → probe device 0 with the same
   capability check. Mismatch → warn and fall back.
4. **All checks pass** → return `"cuda"` (or `"cuda:{index}"`).

The result is cached per (index, torch version) tuple, so repeat callers
never re-probe.

```python
device = safe_device()
model.to(device)         # never crashes on sm-mismatch
```

### `torch_device_env_disabled() -> bool`

Honour `REMANENTIA_DISABLE_CUDA=1` for bench runs where a user wants to
force CPU even though the GPU is healthy. Returns `True` when the env
var is set to any truthy value.

### `clear_cache() -> None`

Reset the internal memoisation. Test-only; never call in production
(the probe invokes `torch.zeros(...).cuda()` as a smoke test, which is
wasted work on repeated calls).

## Invariants

- **Never crashes**: every negative path ends in `"cpu"`, never raises.
- **Single warning per process**: capability mismatches warn once per
  device index; repeated calls stay silent.
- **Respects `CUDA_VISIBLE_DEVICES`**: when the env var hides a device,
  torch itself reports `is_available() == False`; the module does not
  second-guess that.
- **No state mutation**: the probe reads `get_device_capability` and
  `get_arch_list`; it never changes any torch setting.

## Usage pattern

```python
# At every entry point that will call torch on an unknown host:
device = safe_device()
logger.info(f"Remanentia running on {device}")

# Before an embedding build:
if device == "cpu":
    logger.warning("No usable GPU; embedding build will be slow")
```

## See also

- `arcane_retriever.py` — primary consumer via `_get_cross_encoder`.
- `bench_longmemeval.py` — uses `safe_device` before loading any model.
