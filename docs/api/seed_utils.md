# seed_utils

One-call-per-entry-point seed pinning across Python `random`, NumPy,
PyTorch (CPU + CUDA), and `PYTHONHASHSEED`.

## Why this module exists

Before 2026-04-17 no Remanentia entry point set a seed. Every bench
run produced a slightly different number, and the "±10 q noise
envelope" quoted in the LongMemEval docs was a guess rather than a
measurement. Reviewers rightly asked whether the 3.5-point lift from
R10 to R11 was real or noise.

`seed_utils.seed_everything` is the single call every reproducible
entry point now makes. The bench refuses to start without it
(`bench_longmemeval.py` raises if `--seed` is unset and no
`REMANENTIA_SEED` env var is found).

## Public surface

```python
from seed_utils import seed_everything, seed_from_env
```

### `seed_everything(seed: int | None = None, *, torch_cuda_deterministic: bool = False) -> int`

Pin every randomness source atomically. Returns the seed actually
applied so the caller can log it.

- `seed=None` → a fresh seed is drawn from `secrets.randbits(32)` and
  returned. Useful for "don't care about reproducibility, do care about
  logging exactly what ran" callers.
- `torch_cuda_deterministic=True` → also sets
  `torch.backends.cudnn.deterministic = True` and
  `torch.backends.cudnn.benchmark = False`. Slower but byte-identical
  across runs on the same hardware. Off by default — the bench only
  needs q-level reproducibility, not bit-level.

What gets pinned, in order:

1. `random.seed(seed)` — Python stdlib RNG.
2. `numpy.random.seed(seed)` + `numpy.random.default_rng(seed)` —
   legacy global RNG **and** a fresh per-call generator.
3. `torch.manual_seed(seed)` — CPU.
4. `torch.cuda.manual_seed_all(seed)` — all visible CUDA devices.
5. `os.environ["PYTHONHASHSEED"] = str(seed)` — for any child processes
   spawned *after* the call. (It does not affect the current process
   because the interpreter has already read the var.)

### `seed_from_env(var: str = "REMANENTIA_SEED", default: int = 42) -> int`

Read an integer seed from the environment. Raises `ValueError` when
the env var is set but not a base-10 integer; returns `default` when
absent. Combine with `seed_everything` at the top of a bench entry
point:

```python
from seed_utils import seed_everything, seed_from_env

seed = seed_everything(seed_from_env())
logger.info(f"Seed: {seed}")
```

## Invariants

- **Single call is enough.** No sub-module needs to re-seed; every
  library we care about reads its state from the globals set here.
- **No silent no-ops.** When `torch` is importable but CUDA is absent,
  the CUDA seed call is skipped without complaint. When `torch` is not
  importable at all, the call emits a one-line stderr warning and
  continues (`seed_everything` is best-effort; a missing torch is not
  a fatal error for reproducibility of non-torch code paths).
- **PYTHONHASHSEED caveat documented.** The env var affects *child*
  processes only. If a bench spawns workers via `multiprocessing` with
  `spawn` context, they inherit the var; with `fork`, they inherit
  the parent's already-set hash state.

## What this module does NOT pin

- **`sentence-transformers` / `transformers` internal dropout** —
  honours `torch.manual_seed` already; no extra call needed.
- **LLM provider RNG** (OpenAI, Anthropic) — controlled by
  `temperature` and `seed` request parameters on each call, not library
  state. The bench passes `seed=…` to the provider when it set one
  itself.
- **Rust crate RNG** — deterministic by construction. No Rust Remanentia
  crate contains a non-deterministic code path today.

## Usage pattern

```python
# bench_longmemeval.py head
import argparse
from seed_utils import seed_everything

p = argparse.ArgumentParser()
p.add_argument("--seed", type=int, default=None)
args = p.parse_args()

seed = seed_everything(args.seed)
print(f"Running with seed={seed}")
```

## See also

- `bench_longmemeval.py` — primary consumer.
- `run_exp/locomo.py`, `run_exp/bench_notebook.py` — also call it.
