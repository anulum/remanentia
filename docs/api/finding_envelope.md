# Finding Envelope

`finding_envelope.py` seals Synapse-style findings with the SCPN Studio honesty
envelope and verifies the rendered recall-gate grade by recomputing it from the
signed finding unit. A platform-valid envelope is still downgraded to
`UNGRADED` when the signed finding is outside its validity interval, no longer
active, voided by digest, or superseded by a lineage closure entry.

Use `LineageClosureEntry` for bi-temporal supersession records that need to
carry both the retired content digest and the audit reason for retiring it:

```python
from finding_envelope import LineageClosureEntry, verify_finding

verdict = verify_finding(
    envelope.to_dict(),
    rendered_grade,
    keyring=keyring,
    as_of=15.0,
    supersession_closure=(
        LineageClosureEntry(
            content_digest=envelope.content_digest,
            reason="newer source measurement superseded this finding",
            superseded_at=21.0,
            successor_digest="sha256:newer",
        ),
    ),
)
```

Malformed closure metadata fails closed: after the platform signature verifies,
Remanentia returns `UNGRADED` rather than rendering a stale signed finding as
current.

::: finding_envelope
