# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Tests for no_egress_audit

from __future__ import annotations

from no_egress_audit import (
    EgressMonitor,
    audit_endpoints,
    classify_endpoint,
)


class TestClassifyEndpoint:
    def test_loopback_and_runtimes_are_local(self) -> None:
        for ep in (
            "http://localhost:11434/v1",
            "http://127.0.0.1:8080",
            "http://[::1]:8080/v1",
            "http://0.0.0.0:8000",
            "localhost:11434",  # scheme-less loopback descriptor
            "ollama://gemma3:4b",
            "file:///models/qwen.gguf",
            "in-process",
            "null",
        ):
            assert classify_endpoint(ep) == "local"

    def test_cloud_hosts_are_cloud(self) -> None:
        assert classify_endpoint("https://api.openai.com/v1") == "cloud"
        assert classify_endpoint("https://api.anthropic.com") == "cloud"

    def test_runtime_name_cannot_rescue_a_network_url(self) -> None:
        """Regression pin: a runtime substring must not make a remote URL local.

        ``ollama.com`` is Ollama's hosted cloud service and a proxy path
        containing ``/ollama`` still leaves the machine — both are egress.
        The pre-fix substring match classified them local.
        """
        assert classify_endpoint("https://ollama.com/v1") == "cloud"
        assert classify_endpoint("https://api.ollama.com") == "cloud"
        assert classify_endpoint("https://proxy.example.com/ollama/v1") == "cloud"
        assert classify_endpoint("wss://llamacpp.example.com/stream") == "cloud"

    def test_loopback_substring_in_dns_name_is_cloud(self) -> None:
        # a DNS name is not an IP literal — it proves nothing about locality
        assert classify_endpoint("https://localhost.evil.com") == "cloud"
        assert classify_endpoint("http://127.0.0.1.evil.com/v1") == "cloud"

    def test_empty_or_unknown_is_cloud(self) -> None:
        # prove-local, not assume-local
        assert classify_endpoint("") == "cloud"
        assert classify_endpoint(None) == "cloud"
        assert classify_endpoint("mystery-gateway") == "cloud"


class TestEgressMonitor:
    def test_pure_local_run(self) -> None:
        m = EgressMonitor()
        assert m.record("http://localhost:11434/v1") == "local"
        assert m.record("ollama://qwen") == "local"
        v = m.verdict()
        assert v.pure_local is True
        assert bool(v) is True
        assert v.total_calls == 2
        assert v.cloud_calls == 0
        assert v.cloud_endpoints == ()

    def test_one_cloud_call_fails_sovereignty(self) -> None:
        m = EgressMonitor()
        m.record("http://localhost:11434/v1")
        assert m.record("https://api.openai.com/v1") == "cloud"
        v = m.verdict()
        assert v.pure_local is False
        assert bool(v) is False
        assert v.total_calls == 2
        assert v.cloud_calls == 1
        assert v.cloud_endpoints == ("https://api.openai.com/v1",)
        assert v.by_endpoint["http://localhost:11434/v1"] == 1

    def test_empty_endpoint_recorded_as_cloud(self) -> None:
        m = EgressMonitor()
        assert m.record("") == "cloud"
        v = m.verdict()
        assert v.pure_local is False
        assert v.by_endpoint["(empty)"] == 1


class TestAuditEndpoints:
    def test_audit_mixed(self) -> None:
        v = audit_endpoints(
            ["http://localhost:11434", "https://api.openai.com", "http://localhost:11434"]
        )
        assert v.total_calls == 3
        assert v.cloud_calls == 1
        assert v.pure_local is False
        assert v.by_endpoint["http://localhost:11434"] == 2

    def test_audit_empty_is_vacuously_local(self) -> None:
        v = audit_endpoints([])
        assert v.pure_local is True  # no cloud call made
        assert v.total_calls == 0
