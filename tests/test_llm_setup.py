# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# Copyright (C) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (C) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Project: Remanentia — persistent AI memory
# Repository: https://github.com/anulum/remanentia
"""Tests for llm_setup — hardware detection, model recommendation, config."""

from __future__ import annotations

from llm_setup import ModelConfig, recommend_model, write_config


class TestRecommendModel:
    def test_high_vram(self):
        cfg = recommend_model(vram=8, ram=16)
        assert cfg is not None
        assert "7b" in cfg.name
        assert cfg.device == "gpu"

    def test_medium_vram(self):
        cfg = recommend_model(vram=4, ram=16)
        assert cfg is not None
        assert "3b" in cfg.name
        assert cfg.device == "gpu"

    def test_cpu_16gb(self):
        cfg = recommend_model(vram=0, ram=16)
        assert cfg is not None
        assert "3b" in cfg.name
        assert cfg.device == "cpu"

    def test_cpu_8gb(self):
        cfg = recommend_model(vram=0, ram=8)
        assert cfg is not None
        assert "1.5b" in cfg.name
        assert cfg.device == "cpu"

    def test_insufficient_hardware(self):
        cfg = recommend_model(vram=0, ram=4)
        assert cfg is None

    def test_very_high_vram(self):
        cfg = recommend_model(vram=24, ram=32)
        assert cfg is not None
        assert "7b" in cfg.name


class TestWriteConfig:
    def test_writes_toml(self, tmp_path):
        path = tmp_path / "llm.toml"
        result = write_config(
            backend="local",
            local_url="http://gpu:9090/v1",
            local_model="qwen2.5-3b",
            path=path,
        )
        assert result == path
        text = path.read_text()
        assert 'backend = "local"' in text
        assert "gpu:9090" in text
        assert "qwen2.5-3b" in text
        assert "extract = 100" in text

    def test_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "deep" / "nested" / "llm.toml"
        write_config(path=path)
        assert path.exists()

    def test_default_values(self, tmp_path):
        path = tmp_path / "llm.toml"
        write_config(path=path)
        text = path.read_text()
        assert 'backend = "local"' in text
        assert "localhost:8080" in text

    def test_default_path(self, tmp_path):
        import llm_setup

        orig = llm_setup._DEFAULT_CONFIG_DIR
        llm_setup._DEFAULT_CONFIG_DIR = tmp_path
        try:
            result = write_config()
            assert result == tmp_path / "llm.toml"
            assert result.exists()
        finally:
            llm_setup._DEFAULT_CONFIG_DIR = orig


class TestModelConfig:
    def test_dataclass(self):
        cfg = ModelConfig("test-model", "q4_k_m", "gpu", 4.5, 5.0)
        assert cfg.name == "test-model"
        assert cfg.quant == "q4_k_m"
        assert cfg.device == "gpu"
        assert cfg.download_gb == 4.5
        assert cfg.vram_gb == 5.0
