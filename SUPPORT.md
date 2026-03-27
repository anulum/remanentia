# Support

## Getting Help

- **Documentation:** [remanentia.com/docs](https://remanentia.com/docs)
- **GitHub Issues:** [github.com/anulum/remanentia/issues](https://github.com/anulum/remanentia/issues)
- **GitHub Discussions:** [github.com/anulum/remanentia/discussions](https://github.com/anulum/remanentia/discussions)
- **Email:** [protoscience@anulum.li](mailto:protoscience@anulum.li)

## Security Vulnerabilities

Report security vulnerabilities via [GitHub Security Advisories](https://github.com/anulum/remanentia/security/advisories/new) — **not** public issues. See [SECURITY.md](SECURITY.md) for details.

## Commercial Licensing

Remanentia is AGPL-3.0-or-later. For commercial licensing enquiries: [protoscience@anulum.li](mailto:protoscience@anulum.li)

## FAQ

**Q: Do I need a GPU?**
No. Core retrieval runs on CPU with numpy only. GPU accelerates embedding rerank (sentence-transformers) and SNN simulation (PyTorch CUDA).

**Q: Do I need an LLM API key?**
No. Core retrieval, consolidation, and entity extraction work without any LLM. The `ANTHROPIC_API_KEY` is optional for answer synthesis.

**Q: How large a corpus can Remanentia handle?**
Tested on ~20K paragraphs (~2000 documents). Rust BM25 activates at 50K+ paragraphs for scaled performance.
