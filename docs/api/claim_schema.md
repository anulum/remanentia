# Claim Schema

`claim_schema.py` exports the shared claim-axis vocabulary as deterministic JSON
Schema for cross-repository consumers. It reads the live constants from
`claim_axes.py`, preserves the falsified/reference-validated rejection invariant,
and writes the committed schema artefact at
`docs/schema/remanentia_claim_axes.schema.json`.

Use the CLI to refresh or verify the artefact:

```bash
remanentia claim-schema --output docs/schema/remanentia_claim_axes.schema.json
remanentia claim-schema --output docs/schema/remanentia_claim_axes.schema.json --check
```

::: claim_schema
