# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""Auto-generate SNN stimuli from git commit history across all repos.

Scans configured repository directories for recent commits and creates
stimulus files for the SNN daemon to ingest. Each commit message becomes
a stimulus tagged with the repo name.

Usage::

    # One-shot scan (run from monorepo root)
    python 04_ARCANE_SAPIENCE/git_stimulus.py

    # Continuous watch (every 5 minutes)
    python 04_ARCANE_SAPIENCE/git_stimulus.py --watch --interval 300

    # Scan specific repo only
    python 04_ARCANE_SAPIENCE/git_stimulus.py --repo 03_CODE/scpn-control
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [GitStim] %(message)s")
logger = logging.getLogger("ArcSap.GitStim")

BASE_DIR = Path(__file__).parent
MONOREPO_ROOT = BASE_DIR.parent
STIMULUS_DIR = BASE_DIR / "snn_stimuli"
STATE_PATH = BASE_DIR / "snn_state" / "git_stimulus_state.json"

REPO_DIRS = [
    "03_CODE/scpn-control",
    "03_CODE/sc-neurocore",
    "03_CODE/SCPN-Fusion-Core",
    "03_CODE/scpn-phase-orchestrator",
    "03_CODE/scpn-quantum-control",
    "03_CODE/DIRECTOR_AI",
    "03_CODE/CCW_Standalone",
]


def _load_state() -> dict[str, str]:
    """Load last-seen SHA per repo."""
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_state(state: dict[str, str]) -> None:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2) + "\n")


def _git_log_since(repo_path: Path, since_sha: str | None, max_commits: int = 20) -> list[dict]:
    """Get recent commits from a git repo."""
    if not (repo_path / ".git").exists():
        return []

    cmd = ["git", "-C", str(repo_path), "log", "--format=%H|%s|%an|%at", f"-{max_commits}"]
    if since_sha:
        cmd = ["git", "-C", str(repo_path), "log", "--format=%H|%s|%an|%at", f"{since_sha}..HEAD"]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            return []
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return []

    commits = []
    for line in result.stdout.strip().split("\n"):
        if not line.strip():
            continue
        parts = line.split("|", 3)
        if len(parts) < 4:
            continue
        commits.append({
            "sha": parts[0],
            "message": parts[1],
            "author": parts[2],
            "timestamp": int(parts[3]),
        })
    return commits


def _drop_stimulus(text: str, source: str, project: str, timestamp: int) -> str:
    """Write a stimulus file for the daemon."""
    STIMULUS_DIR.mkdir(parents=True, exist_ok=True)
    safe_source = source.replace("/", "_").replace("\\", "_")
    path = STIMULUS_DIR / f"git_{safe_source}_{timestamp}.json"
    if path.exists():
        path = STIMULUS_DIR / f"git_{safe_source}_{timestamp}_{hash(text) % 10000}.json"

    data = {
        "text": text,
        "source": f"git-{source}",
        "project": project,
        "timestamp": timestamp,
    }
    path.write_text(json.dumps(data, indent=2) + "\n")
    return str(path)


def scan_repos(repo_dirs: list[str] | None = None) -> int:
    """Scan all repos for new commits and generate stimuli.

    Returns the number of new stimuli generated.
    """
    state = _load_state()
    dirs = repo_dirs or REPO_DIRS
    total_new = 0

    for rel_dir in dirs:
        repo_path = MONOREPO_ROOT / rel_dir
        if not repo_path.exists():
            continue

        repo_name = rel_dir.split("/")[-1]
        last_sha = state.get(repo_name)
        commits = _git_log_since(repo_path, last_sha)

        if not commits:
            continue

        # Newest commit first — update state to newest SHA
        state[repo_name] = commits[0]["sha"]

        for commit in commits:
            text = f"{repo_name}: {commit['message']} (by {commit['author']})"
            _drop_stimulus(text, repo_name, repo_name, commit["timestamp"])
            total_new += 1

        logger.info("%s: %d new commits → stimuli", repo_name, len(commits))

    _save_state(state)
    return total_new


def main():
    parser = argparse.ArgumentParser(description="Git commit auto-stimulus for Arcane Sapience SNN")
    parser.add_argument("--repo", type=str, help="Scan a specific repo directory only")
    parser.add_argument("--watch", action="store_true", help="Continuous watch mode")
    parser.add_argument("--interval", type=int, default=300, help="Watch interval in seconds")
    parser.add_argument("--reset", action="store_true", help="Reset state (re-scan all)")
    args = parser.parse_args()

    if args.reset and STATE_PATH.exists():
        STATE_PATH.unlink()
        logger.info("State reset — will re-scan all repos")

    repos = [args.repo] if args.repo else None

    if args.watch:
        logger.info("Watch mode: scanning every %ds", args.interval)
        while True:
            n = scan_repos(repos)
            if n > 0:
                logger.info("Generated %d stimuli this cycle", n)
            time.sleep(args.interval)
    else:
        n = scan_repos(repos)
        logger.info("Done: %d new stimuli generated", n)


if __name__ == "__main__":
    main()
