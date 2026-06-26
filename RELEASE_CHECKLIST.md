# Release Checklist

One-line summary: **every release emits sdist + wheel + SBOM +
sigstore signatures + SLSA provenance, and the checklist below is
the operator's way to prove we actually did that**.

## Before tagging

- [ ] CHANGELOG moved from `[Unreleased]` to the target version with
      the release date.
- [ ] Version bumped in `pyproject.toml` (`version = "X.Y.Z"`).
- [ ] `CITATION.cff` `version:` and `date-released:` updated.
- [ ] README test badge / count matches `pytest --collect-only -q`.
- [ ] `docs/benchmarks/*` reflect the committed result files, not a
      superseded number.
- [ ] `python tools/public_leak_audit.py` passes. This scans tracked public
      text surfaces for private workspace paths, private workspace labels, and
      agent-identity labels before packaging or publication.
- [ ] ADR log updated if this release makes an architectural change.
- [ ] `git status` clean; no untracked source files committed.

## Verify the automated pipeline

- [ ] `.github/workflows/release.yml` permissions block contains
      `contents: write`, `id-token: write`, `attestations: write`.
- [ ] `python tools/check_release_integrity.py` passes. This verifies
      pinned workflow actions, CycloneDX SBOM generation from the installed
      release artefact, SHA-256 digest generation, sigstore signing and
      verification, SLSA provenance, and GitHub Release upload of `.sigstore`
      bundles.
- [ ] Action SHAs are pinned (not `@v3`, not `@main`). When bumping,
      fetch the SHA live:

      ```bash
      curl -sSL https://api.github.com/repos/sigstore/gh-action-sigstore-python/git/refs/tags/vX.Y.Z \
        | jq -r '.object.sha'
      ```

- [ ] PyPI trusted-publisher configuration still points at this repo
      + workflow name.

## Push the tag

```bash
git tag vX.Y.Z -m "vX.Y.Z"
git push origin vX.Y.Z
```

The workflow builds sdist + wheel, installs into a throwaway venv to
emit an accurate CycloneDX SBOM, signs with sigstore, attaches a
SLSA attestation, and uploads everything to the GitHub Release.

## After the workflow lands

- [ ] `gh release view vX.Y.Z` shows sdist, wheel, `sbom.cyclonedx.json`,
      `sha256sums.txt` **and** sigstore `.sigstore` bundles.
- [ ] `gh attestation verify <wheel> --repo anulum/remanentia` returns
      a valid provenance.
- [ ] `sigstore verify identity --cert-identity
      'https://github.com/anulum/remanentia/.github/workflows/release.yml@refs/tags/vX.Y.Z'
      --cert-oidc-issuer 'https://token.actions.githubusercontent.com'
      remanentia-X.Y.Z.tar.gz` returns OK.
- [ ] `pip install remanentia==X.Y.Z` in a fresh venv succeeds and
      `python -c "import remanentia; print(remanentia.__version__)"`
      returns `X.Y.Z`.
- [ ] Quick smoke of the API: `remanentia` CLI + `api_server.py`
      boot, `/health` 200.

## Communicate

- [ ] GitHub Discussions announcement (link to CHANGELOG + release
      assets).
- [ ] README "latest version" badge refreshes automatically via
      shields.io; no manual step required.
- [ ] If security-relevant, mention the change in `SECURITY.md`
      § "Verifying a Released Artefact".

## Rollback

If something is broken within 48 h of release:

1. Yank PyPI: `pip install twine && twine yank remanentia==X.Y.Z`
   (leaves the file, blocks default installation).
2. Mark the GitHub Release as pre-release or delete the release
   (**not** the tag — leave the tag so the provenance chain stays
   coherent).
3. Announce in Discussions with the rollback reason and expected
   re-release window.

Last change: 2026-04-17.
