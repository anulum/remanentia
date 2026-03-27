# Governance

## Project Lead

**Miroslav Šotek** ([ORCID: 0009-0009-3560-0851](https://orcid.org/0009-0009-3560-0851))
- Final authority on architecture, roadmap, and releases
- Contact: [remanentia@anulum.li](mailto:remanentia@anulum.li) | [protoscience@anulum.li](mailto:protoscience@anulum.li)

## Decision Making

This is a single-maintainer project with AI-assisted development.

- **Architecture decisions:** project lead
- **Feature proposals:** GitHub Issues or Discussions
- **Bug fixes:** PR with tests, reviewed by project lead
- **Breaking changes:** documented in CHANGELOG, announced in Discussions

## AI Collaboration

Development is assisted by AI agents (Claude, Codex, Gemini) operating
under strict protocols defined in CLAUDE.md, CODEX_INSTRUCTIONS.md, and
GEMINI_INSTRUCTIONS.md. All AI-generated code is reviewed by the project
lead before merge. AI agents do not have push access.

## Releases

- Versions follow [Semantic Versioning](https://semver.org/)
- Release process documented in the Enterprise Hardening Checklist
- All releases are tagged and published to PyPI via OIDC trusted publisher

## License

AGPL-3.0-or-later with commercial license available.
See [NOTICE.md](NOTICE.md) for use-case-to-license mapping.
