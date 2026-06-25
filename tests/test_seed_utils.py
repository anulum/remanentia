# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Tests for seed_utils

from __future__ import annotations

import os
import random

import numpy as np
import pytest

import seed_utils
from seed_utils import build_reproducibility_manifest, seed_everything, seed_from_env


class TestSeedEverything:
    def test_default_seed_is_42(self):
        assert seed_everything() == 42

    def test_returns_provided_seed(self):
        assert seed_everything(7) == 7

    def test_python_random_is_deterministic(self):
        seed_everything(123)
        a = [random.random() for _ in range(5)]
        seed_everything(123)
        b = [random.random() for _ in range(5)]
        assert a == b

    def test_numpy_random_is_deterministic(self):
        seed_everything(123)
        a = np.random.rand(5).tolist()
        seed_everything(123)
        b = np.random.rand(5).tolist()
        assert a == b

    def test_numpy_default_rng_independent_between_seeds(self):
        seed_everything(1)
        a = np.random.rand(3).tolist()
        seed_everything(2)
        b = np.random.rand(3).tolist()
        assert a != b

    def test_pythonhashseed_set(self):
        seed_everything(99)
        assert os.environ["PYTHONHASHSEED"] == "99"

    def test_torch_determinism(self):
        try:
            import torch
        except ImportError:  # pragma: no cover — torch is a hard dep
            pytest.skip("torch not installed")
        seed_everything(321)
        t1 = torch.randn(5)
        seed_everything(321)
        t2 = torch.randn(5)
        assert torch.equal(t1, t2)


class TestSeedFromEnv:
    def test_default_when_unset(self, monkeypatch):
        monkeypatch.delenv("REMANENTIA_SEED", raising=False)
        assert seed_from_env() == 42

    def test_reads_int(self, monkeypatch):
        monkeypatch.setenv("REMANENTIA_SEED", "777")
        assert seed_from_env() == 777

    def test_custom_default(self, monkeypatch):
        monkeypatch.delenv("REMANENTIA_SEED", raising=False)
        assert seed_from_env(default=13) == 13

    def test_non_integer_falls_back(self, monkeypatch):
        monkeypatch.setenv("REMANENTIA_SEED", "not-a-number")
        assert seed_from_env(default=5) == 5

    def test_custom_var_name(self, monkeypatch):
        monkeypatch.setenv("MY_SEED", "11")
        monkeypatch.delenv("REMANENTIA_SEED", raising=False)
        assert seed_from_env("MY_SEED") == 11


class TestReproducibilityManifest:
    def test_manifest_records_seed_workload_and_parameters(self):
        manifest = build_reproducibility_manifest(
            seed=123,
            workload="performance_benchmark",
            parameters={"iterations": 5, "query": "memory"},
        )

        assert manifest["seed"] == 123
        assert manifest["workload"] == "performance_benchmark"
        assert manifest["parameters"] == {"iterations": 5, "query": "memory"}
        assert manifest["python_version"]
        assert manifest["platform"]
        assert "packages" in manifest
        assert manifest["packages"]["numpy"] == np.__version__
        assert "timestamp_utc" in manifest

    def test_manifest_is_json_serialisable(self):
        manifest = build_reproducibility_manifest(
            seed=42,
            workload="unit-test",
            parameters={"alpha": 0.1},
        )

        import json

        encoded = json.dumps(manifest, sort_keys=True)
        assert '"seed": 42' in encoded
        assert '"alpha": 0.1' in encoded

    def test_manifest_records_blank_version_for_missing_optional_package(self, monkeypatch):
        real_version = seed_utils.importlib.metadata.version

        def version(name: str) -> str:
            if name == "sentence-transformers":
                raise seed_utils.importlib.metadata.PackageNotFoundError(name)
            return real_version(name)

        monkeypatch.setattr(seed_utils.importlib.metadata, "version", version)

        manifest = build_reproducibility_manifest(seed=42, workload="unit-test")

        assert manifest["packages"]["sentence-transformers"] == ""


class TestEndToEnd:
    def test_random_and_numpy_in_lockstep(self):
        """Both backends reset together when seed_everything is re-called."""
        seed_everything(42)
        py_first = [random.random() for _ in range(3)]
        np_first = np.random.rand(3).tolist()

        seed_everything(42)
        py_second = [random.random() for _ in range(3)]
        np_second = np.random.rand(3).tolist()

        assert py_first == py_second
        assert np_first == np_second
