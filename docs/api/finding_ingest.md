# Finding Ingest

`finding_ingest.py` pulls authored findings from a SYNAPSE hub event store,
runs them through the real finding parser and admission gate, and persists only
admitted findings to the Markdown finding sink. The cursor advances across every
scanned event, including rejected events, so malformed claims do not loop
forever on the next ingest pass.

Remanentia also enforces the claim-axis invariant required by the MS.3 claim
model pull-forward: a finding with `evidence_kind=falsified` and
`claim_status=reference_validated` is rejected before persistence. A negative
finding must enter memory as a refuted claim, never as a reference-validated
claim that only gets corrected later at render time.

::: finding_ingest
