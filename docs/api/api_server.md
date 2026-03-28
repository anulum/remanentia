# api_server

Lightweight stdlib HTTP API server for cross-service integration (SYNAPSE bridge, SPO, etc). No FastAPI dependency.

Endpoints: GET `/health`, `/status`; POST `/recall`, `/consolidate`, `/remember`.

::: api_server.RemanentiaHandler
    options:
      show_source: true
      members_order: source

::: api_server._json_default
