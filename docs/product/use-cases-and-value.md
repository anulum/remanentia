# Use Cases and Value

Remanentia is infrastructure for software that must retain useful context beyond
one prompt or process lifetime. It is most valuable when memory must remain
searchable, inspectable, and under the operator's control.

## Who it is for

- Teams building assistants that revisit decisions across many sessions.
- Engineering, research, and operations groups with large Markdown evidence
  trails and recurring handovers.
- Developers who need an MCP or HTTP memory service without committing the
  source corpus to a hosted vector database.
- Organisations that need explicit provenance, correction, and no-egress
  evaluation surfaces.

## Application patterns

### Agent continuity

Store important decisions and observations, then let an agent retrieve them at
the next task boundary. MCP exposes recall, write, graph, status, feedback, and
correctness tools through one stdio process.

**Value driver:** less repeated discovery and fewer contradictions caused by
context-window turnover.

### Engineering knowledge and handovers

Index session logs, ADRs, incident findings, and selected source documents.
Date-aware search and entity relations help recover both the current conclusion
and the evidence that led to it.

**Value driver:** shorter time-to-context for maintainers and a more auditable
trail than unstructured chat history.

### Research memory

Keep observations, hypotheses, and corrections as evidence-linked notes.
Benchmark and provenance modules make the retrieval path measurable rather than
relying on demonstrations alone.

**Value driver:** repeatable retrieval evaluation and clearer separation between
measured findings and experimental ideas.

### Private and sovereign deployments

Run BM25, graphs, temporal logic, and rule-based extraction locally. Embeddings,
local models, or hosted models are optional layers chosen by the operator.

**Value driver:** control over corpus location, model egress, retention, and
deployment topology.

### Multi-agent coordination

Expose one selected memory store to multiple tools over MCP or authenticated
HTTP. Feedback and correctness records can support later recall calibration.

**Value driver:** shared context with explicit storage and evaluation boundaries.

## What makes it different

| Design choice | Practical consequence |
|---|---|
| Local-first core | Useful recall does not require a managed service or hosted LLM |
| Text and JSON artefacts | Memories and evidence remain inspectable and portable |
| Hybrid retrieval | Lexical precision can be combined with optional semantic ranking |
| Temporal and graph context | The system can recover relationships and changes, not only similar chunks |
| Multiple interfaces | The same memory can serve shell users, Python code, agents, and services |
| Published evaluation surfaces | Operators can inspect accuracy, no-egress, abstention, and lineage evidence |

## Readiness boundaries

The core retrieval and integration surfaces are functional and tested. Some
capabilities remain conditional or experimental:

- Embedding and model-based layers add dependencies and must be enabled.
- Hosted synthesis can send selected context to a provider; it is not part of
  the no-egress core path.
- Temporal SNN memory is research governed by preregistered experiments, not a
  production dependency.
- Remanentia does not replace application authentication, backup policy, records
  governance, human review, or domain-specific safety controls.
- Benchmark results describe committed configurations and datasets; they do not
  guarantee accuracy on a new corpus.

## Adoption path

1. Run the [Quick Start](../tutorials/quickstart.md) on a synthetic or disposable
   corpus.
2. Choose an integration in the [Interface Guide](../guides/choose-an-interface.md).
3. Define source roots, data retention, and authentication before production use.
4. Measure recall on representative questions and inspect failure categories.
5. Add embeddings or synthesis only when the measured benefit justifies their
   compute, privacy, and operational cost.

## Commercial path

The repository is available under AGPL-3.0-or-later. A commercial licence is
available for products or deployments that need different distribution terms.
Commercial value comes from integration, operational control, and reduced
context-recovery work; this documentation makes no speculative valuation or
market-size claim.

For licensing or deployment discussions, contact
[remanentia@anulum.li](mailto:remanentia@anulum.li).
