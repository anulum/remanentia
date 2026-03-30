# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Example: MCP server configuration

"""Generate and test MCP server configuration.

The MCP server provides 4 tools for AI agents:
- remanentia_recall: search memory with filters
- remanentia_remember: persist a new memory (triggers background consolidation)
- remanentia_status: system statistics
- remanentia_graph: entity relationship query

Usage::

    cd remanentia/
    python examples/mcp_config.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp_server import TOOLS, handle_recall, handle_status, handle_graph

# Show available tools and their schemas
print("=== MCP Tools ===\n")
for tool in TOOLS:
    print(f"Tool: {tool['name']}")
    print(f"  Description: {tool['description'][:100]}")
    props = tool["inputSchema"].get("properties", {})
    required = tool["inputSchema"].get("required", [])
    for pname, pdef in props.items():
        req = " (required)" if pname in required else ""
        print(f"  Param: {pname}{req} — {pdef.get('description', pdef.get('type', ''))}")
    print()

# Generate .mcp.json config
remanentia_root = str(Path(__file__).resolve().parent.parent)
config = {
    "mcpServers": {
        "remanentia": {
            "command": "python",
            "args": [str(Path(remanentia_root) / "mcp_server.py")],
            "env": {
                "REMANENTIA_BASE": remanentia_root,
            },
        }
    }
}
print("=== .mcp.json config ===\n")
print(json.dumps(config, indent=2))

# Test the tools directly (same functions the MCP server calls)
print("\n=== handle_status() ===\n")
status = handle_status()
print(status)

print("\n=== handle_graph(top=3) ===\n")
graph = handle_graph(top=3)
print(graph)
