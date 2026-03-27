# Remanentia

**Persistent AI memory with SNN-orchestrated consolidation, entity graphs, and deep contextual recall.**

BM25+embedding hybrid retrieval with RRF | 11 typed entity relation types | temporal reasoning | async consolidation | thread-safe MCP server

Part of the [ANULUM](https://www.anulum.li) scientific computing ecosystem — the memory layer connecting SC-NeuroCore, Director-AI, SCPN-Fusion-Core, and other projects via cross-project retrieval.

## Quick Start

```bash
pip install remanentia
remanentia init
remanentia consolidate --force
remanentia search "what did we decide about authentication"
```

## Architecture

```
Query
  |
  v
BM25 (real TF + inverted index) .............. first-pass retrieval
  |
  v
Bi-encoder rerank (MiniLM-L6-v2) ............. semantic similarity
  |
  v
Reciprocal Rank Fusion ........................ scale-invariant score fusion
  |
  v
Cross-encoder rerank (MiniLM-L-6-v2) ......... fine-grained re-scoring
  |
  v
Entity graph boost ............................ 11 typed relation types
  |
  v
Temporal graph + date arithmetic .............. TReMu code execution
  |
  v
Answer extraction ............................. query-proximity scoring
  |
  v
Knowledge store (multi-hop graph search) ...... Zettelkasten + prospective queries
```

## Memory Types

| Type | Storage | Example |
|------|---------|---------|
| Episodic | `reasoning_traces/*.md` | Raw session decisions |
| Semantic | `memory/semantic/**/*.md` | Consolidated facts with YAML frontmatter |
| Procedural | `skills/*.json` | Extracted skills and workflows |
| Graph | `memory/graph/*.jsonl` | Entity-entity relations with evidence |

## MCP Server

```json
{
  "mcpServers": {
    "remanentia": {
      "command": "python",
      "args": ["mcp_server.py"]
    }
  }
}
```

Tools: `remanentia_recall`, `remanentia_remember`, `remanentia_status`, `remanentia_graph`

## License

AGPL-3.0-or-later | Commercial license available

**Contact:** [remanentia@anulum.li](mailto:remanentia@anulum.li) | [protoscience@anulum.li](mailto:protoscience@anulum.li) |
[GitHub Discussions](https://github.com/anulum/remanentia/discussions) |
[www.remanentia.com](https://www.remanentia.com) | [www.anulum.li](https://www.anulum.li)

---

<p align="center">
  <a href="https://www.anulum.li">
    <img src="assets/anulum_logo_company.jpg" width="180" alt="ANULUM">
  </a>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://www.anulum.li">
    <img src="assets/fortis_studio_logo.jpg" width="180" alt="Fortis Studio">
  </a>
  <br>
  <em>Developed by <a href="https://www.anulum.li">ANULUM</a> / Fortis Studio</em>
</p>
