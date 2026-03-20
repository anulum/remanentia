#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""Register a heartbeat for the Arcane Sapience monitor dashboard.

Called automatically by Claude Code SessionStart hook, or manually::

    python 04_ARCANE_SAPIENCE/heartbeat_register.py claude scpn-control active "physics deepening"
    python 04_ARCANE_SAPIENCE/heartbeat_register.py codex director-ai active "lint fixes"
    python 04_ARCANE_SAPIENCE/heartbeat_register.py gemini sc-neurocore idle
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

HEARTBEAT_DIR = Path(__file__).parent / "heartbeats"


def register(agent: str, project: str = "", status: str = "active", detail: str = "") -> str:
    HEARTBEAT_DIR.mkdir(parents=True, exist_ok=True)
    safe = agent.replace("/", "_").replace("\\", "_").replace(" ", "_")
    path = HEARTBEAT_DIR / f"{safe}.json"
    data = {
        "agent": agent,
        "project": project,
        "status": status,
        "detail": detail,
        "pid": os.getpid(),
        "timestamp": time.time(),
    }
    path.write_text(json.dumps(data, indent=2) + "\n")
    return str(path)


if __name__ == "__main__":
    args = sys.argv[1:]
    agent = args[0] if len(args) > 0 else "unknown"
    project = args[1] if len(args) > 1 else ""
    status = args[2] if len(args) > 2 else "active"
    detail = args[3] if len(args) > 3 else ""
    path = register(agent, project, status, detail)
    print(f"Heartbeat: {agent} -> {path}")
