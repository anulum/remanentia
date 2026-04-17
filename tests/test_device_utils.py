# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Tests for safe-device selection

from __future__ import annotations

from unittest.mock import patch

import pytest

import device_utils


@pytest.fixture(autouse=True)
def _reset_cache():
    device_utils.clear_cache()
    yield
    device_utils.clear_cache()


class TestSafeDeviceNoCuda:
    def test_returns_cpu_when_cuda_unavailable(self):
        with patch("torch.cuda.is_available", return_value=False):
            assert device_utils.safe_device() == "cpu"

    def test_caches_per_index(self):
        with patch("torch.cuda.is_available", return_value=False):
            device_utils.safe_device(index=0)
            device_utils.safe_device(index=1)
        # Calling with different index still resolves without re-patching
        assert "cpu" in [device_utils.safe_device(0), device_utils.safe_device(1)]


class TestSafeDeviceSupportedGPU:
    def test_returns_cuda_when_arch_matches(self):
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_capability", return_value=(8, 0)),
            patch("torch.cuda.get_arch_list", return_value=["sm_80", "sm_90"]),
        ):
            assert device_utils.safe_device() == "cuda"

    def test_returns_cuda_indexed(self):
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_capability", return_value=(8, 6)),
            patch("torch.cuda.get_arch_list", return_value=["sm_86"]),
        ):
            assert device_utils.safe_device(index=2) == "cuda:2"

    def test_ptx_fallback_accepted(self):
        # Arch list sometimes encodes PTX forward-compat: sm_86+PTX
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_capability", return_value=(8, 6)),
            patch("torch.cuda.get_arch_list", return_value=["sm_86+PTX"]),
        ):
            assert device_utils.safe_device() == "cuda"


class TestSafeDeviceUnsupportedGPU:
    def test_gtx1060_sm61_falls_back_to_cpu(self, capsys):
        # Reproduces the exact 2026-04-17 CI failure: torch build has
        # sm_70+ only, GPU reports sm_61.
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_capability", return_value=(6, 1)),
            patch(
                "torch.cuda.get_arch_list",
                return_value=["sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120"],
            ),
            patch("torch.cuda.get_device_name", return_value="NVIDIA GeForce GTX 1060 6GB"),
        ):
            result = device_utils.safe_device()
        assert result == "cpu"
        err = capsys.readouterr().err
        assert "sm_61" in err
        assert "GTX 1060" in err

    def test_warn_false_suppresses_stderr(self, capsys):
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_capability", return_value=(6, 1)),
            patch("torch.cuda.get_arch_list", return_value=["sm_80"]),
            patch("torch.cuda.get_device_name", return_value="old card"),
        ):
            device_utils.safe_device(warn=False)
        assert capsys.readouterr().err == ""

    def test_empty_arch_list_treated_as_compatible(self):
        # Older torch may not expose get_arch_list; don't block on missing info.
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_capability", return_value=(6, 1)),
            patch("torch.cuda.get_arch_list", return_value=[]),
        ):
            assert device_utils.safe_device() == "cuda"


class TestEnvDisabled:
    def test_cuda_visible_devices_empty(self, monkeypatch):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
        assert device_utils.torch_device_env_disabled() is True

    def test_cuda_visible_devices_set(self, monkeypatch):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
        assert device_utils.torch_device_env_disabled() is False

    def test_cuda_visible_devices_unset(self, monkeypatch):
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        assert device_utils.torch_device_env_disabled() is False


class TestCapToArch:
    def test_two_digit(self):
        assert device_utils._cap_to_arch((8, 0)) == "sm_80"
        assert device_utils._cap_to_arch((8, 6)) == "sm_86"

    def test_one_digit(self):
        assert device_utils._cap_to_arch((6, 1)) == "sm_61"
        assert device_utils._cap_to_arch((7, 5)) == "sm_75"


class TestGetCapabilityFailure:
    def test_exception_falls_back_to_cpu(self):
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.get_device_capability", side_effect=RuntimeError("boom")),
        ):
            assert device_utils.safe_device() == "cpu"
