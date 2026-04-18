# Notebook Walkthrough

The `notebooks/` directory holds runnable Jupyter notebooks that
show the pipeline end-to-end. Start with
[`01_load_oracle_and_query.ipynb`](https://github.com/anulum/remanentia/blob/main/notebooks/01_load_oracle_and_query.ipynb)
— the shortest path from a fresh install to a working retrieval
query.

## What the notebook covers

1. Decompose two synthetic conversation sessions into
   `AtomicFact` records via `fact_decomposer.decompose_sessions`.
2. Build an `ArcaneRetriever` over those facts.
3. Run a `retrieve()` query and print the ranked results.
4. Extract a rule-based answer via `answer_extractor.extract_answer`.
5. **Optional:** repeat the pipeline on a randomly-picked item
   from `data/longmemeval_oracle.json` if it is present locally.

The notebook runs **offline** — no hosted-LLM API keys are needed
for the rule-based path. The optional oracle cell is
guarded; it prints a pointer when the oracle is not installed.

## Running it

```bash
pip install -e ".[all]"
pip install jupyter
jupyter lab notebooks/
```

Or headless:

```bash
pip install nbclient
jupyter execute notebooks/01_load_oracle_and_query.ipynb
```

## Next steps

- `docs/benchmarks/LongMemEval.md` — full benchmark methodology.
- `bench_longmemeval.py --help` — run the bench end-to-end.
- `docs/models/` — model cards for the five trained components.
- `docs/adr/` — architectural decision records.
