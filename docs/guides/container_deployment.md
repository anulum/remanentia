<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
-->

# Container Deployment

Remanentia ships a Dockerfile and Compose file for the REST API surface.
The container runs `remanentia serve`, stores mutable memory under `/data`,
and probes `GET /health` for runtime health.

## Build

```bash
docker build -t remanentia:local .
```

The image installs the `api` extra only, runs as a non-root `remanentia` user,
and leaves memory data outside the image in the `/data` volume.

## Secret

Create the bearer token file used by Compose:

```bash
mkdir -p secrets
printf '%s\n' '<replace-with-a-random-token>' > secrets/remanentia_api_token
chmod 600 secrets/remanentia_api_token
```

The `secrets/` directory is excluded from Docker build context and git.

## Run

```bash
docker compose up --build
```

Compose binds the REST API to `127.0.0.1:8001`, mounts a named
`remanentia-data` volume at `/data`, and passes the token through
`/run/secrets/remanentia_api_token`.

## Health

Both Dockerfile and Compose use the same health probe:

```text
GET http://127.0.0.1:8001/health
```

The health endpoint stays public so the container can report readiness without
exposing the bearer token. Private endpoints still require
`Authorization: Bearer <token>`.

## Direct Docker Run

For direct `docker run`, pass `REMANENTIA_API_TOKEN` or mount a token file:

```bash
docker run --rm \
  -p 127.0.0.1:8001:8001 \
  -e REMANENTIA_API_TOKEN='<replace-with-a-random-token>' \
  -v remanentia-data:/data \
  remanentia:local
```
