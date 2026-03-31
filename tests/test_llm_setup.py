# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# Remanentia — Tests for LLM setup

"""Tests for llm_setup — hardware detection, model recommendation, config writing.

Multi-angle coverage: boundary tiers, edge cases, invalid inputs, config
round-trip, file system errors, idempotency, and content validation.
"""

from __future__ import annotations


from llm_setup import ModelConfig, recommend_model, write_config


# ── ModelConfig dataclass ─────────────────────────────────────


class TestModelConfig:
    def test_fields(self):
        cfg = ModelConfig("test", "q4", "gpu", 4.5, 5.0)
        assert cfg.name == "test"
        assert cfg.quant == "q4"
        assert cfg.device == "gpu"
        assert cfg.download_gb == 4.5
        assert cfg.vram_gb == 5.0

    def test_equality(self):
        a = ModelConfig("m", "q4", "gpu", 1.0, 2.0)
        b = ModelConfig("m", "q4", "gpu", 1.0, 2.0)
        assert a == b

    def test_inequality(self):
        a = ModelConfig("m1", "q4", "gpu", 1.0, 2.0)
        b = ModelConfig("m2", "q4", "gpu", 1.0, 2.0)
        assert a != b

    def test_repr(self):
        cfg = ModelConfig("test", "q4", "gpu", 1.0, 2.0)
        r = repr(cfg)
        assert "test" in r
        assert "q4" in r


# ── recommend_model — boundary tiers ──────────────────────────


class TestRecommendModel:
    """Every hardware tier boundary is tested both sides."""

    def test_high_vram_returns_7b(self):
        cfg = recommend_model(vram=8, ram=16)
        assert cfg is not None
        assert "7b" in cfg.name
        assert cfg.device == "gpu"

    def test_vram_exactly_8(self):
        cfg = recommend_model(vram=8.0, ram=0)
        assert cfg is not None
        assert "7b" in cfg.name

    def test_vram_just_below_8(self):
        cfg = recommend_model(vram=7.9, ram=0)
        assert cfg is not None
        assert "3b" in cfg.name
        assert cfg.device == "gpu"

    def test_medium_vram_returns_3b_gpu(self):
        cfg = recommend_model(vram=4, ram=16)
        assert cfg is not None
        assert "3b" in cfg.name
        assert cfg.device == "gpu"

    def test_vram_exactly_4(self):
        cfg = recommend_model(vram=4.0, ram=0)
        assert cfg is not None
        assert "3b" in cfg.name

    def test_vram_just_below_4(self):
        cfg = recommend_model(vram=3.9, ram=16)
        assert cfg is not None
        assert cfg.device == "cpu"

    def test_cpu_16gb_returns_3b_cpu(self):
        cfg = recommend_model(vram=0, ram=16)
        assert cfg is not None
        assert "3b" in cfg.name
        assert cfg.device == "cpu"

    def test_ram_exactly_16(self):
        cfg = recommend_model(vram=0, ram=16.0)
        assert cfg is not None
        assert "3b" in cfg.name

    def test_ram_just_below_16(self):
        cfg = recommend_model(vram=0, ram=15.9)
        assert cfg is not None
        assert "1.5b" in cfg.name

    def test_cpu_8gb_returns_1_5b(self):
        cfg = recommend_model(vram=0, ram=8)
        assert cfg is not None
        assert "1.5b" in cfg.name
        assert cfg.device == "cpu"

    def test_ram_exactly_8(self):
        cfg = recommend_model(vram=0, ram=8.0)
        assert cfg is not None
        assert "1.5b" in cfg.name

    def test_insufficient_returns_none(self):
        assert recommend_model(vram=0, ram=4) is None

    def test_zero_everything(self):
        assert recommend_model(vram=0, ram=0) is None

    def test_negative_values(self):
        assert recommend_model(vram=-1, ram=-1) is None

    def test_very_high_vram(self):
        cfg = recommend_model(vram=80, ram=128)
        assert cfg is not None
        assert "7b" in cfg.name

    def test_gpu_preferred_over_cpu(self):
        """With both VRAM and RAM, GPU path wins."""
        cfg = recommend_model(vram=6, ram=32)
        assert cfg is not None
        assert cfg.device == "gpu"

    def test_all_configs_have_quant(self):
        """Every returned config has q4_k_m quantisation."""
        for vram, ram in [(8, 16), (4, 8), (0, 16), (0, 8)]:
            cfg = recommend_model(vram=vram, ram=ram)
            assert cfg is not None
            assert cfg.quant == "q4_k_m"

    def test_all_configs_have_positive_download(self):
        for vram, ram in [(8, 16), (4, 8), (0, 16), (0, 8)]:
            cfg = recommend_model(vram=vram, ram=ram)
            assert cfg.download_gb > 0


# ── write_config ──────────────────────────────────────────────


class TestWriteConfig:
    def test_writes_valid_toml(self, tmp_path):
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
        assert "[llm.tokens]" in text

    def test_token_defaults(self, tmp_path):
        path = tmp_path / "llm.toml"
        write_config(path=path)
        text = path.read_text()
        assert "extract = 100" in text
        assert "generate = 200" in text
        assert "synthesise = 200" in text

    def test_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "deep" / "nested" / "dir" / "llm.toml"
        write_config(path=path)
        assert path.exists()

    def test_default_values(self, tmp_path):
        path = tmp_path / "llm.toml"
        write_config(path=path)
        text = path.read_text()
        assert 'backend = "local"' in text
        assert "localhost:8080" in text
        assert "qwen2.5-7b" in text

    def test_default_path_uses_config_dir(self, tmp_path):
        import llm_setup

        orig = llm_setup._DEFAULT_CONFIG_DIR
        llm_setup._DEFAULT_CONFIG_DIR = tmp_path
        try:
            result = write_config()
            assert result == tmp_path / "llm.toml"
            assert result.exists()
        finally:
            llm_setup._DEFAULT_CONFIG_DIR = orig

    def test_overwrites_existing(self, tmp_path):
        path = tmp_path / "llm.toml"
        path.write_text("old content", encoding="utf-8")
        write_config(backend="anthropic", path=path)
        text = path.read_text()
        assert "old content" not in text
        assert 'backend = "anthropic"' in text

    def test_utf8_encoding(self, tmp_path):
        path = tmp_path / "llm.toml"
        write_config(path=path)
        raw = path.read_bytes()
        assert b"\xff" not in raw  # no BOM or non-UTF8

    def test_special_chars_in_url(self, tmp_path):
        path = tmp_path / "llm.toml"
        write_config(local_url="http://host:8080/v1?key=val&x=1", path=path)
        text = path.read_text()
        assert "key=val&x=1" in text

    def test_roundtrip_with_load_config(self, tmp_path):
        """Written config can be loaded back correctly."""
        from llm_backend import load_config

        path = tmp_path / "llm.toml"
        write_config(
            backend="local",
            local_url="http://myhost:1234/v1",
            local_model="test-model",
            path=path,
        )
        cfg = load_config(path)
        assert cfg.backend == "local"
        assert cfg.local_url == "http://myhost:1234/v1"
        assert cfg.local_model == "test-model"
        assert cfg.max_tokens_extract == 100
        assert cfg.max_tokens_generate == 200
        assert cfg.max_tokens_synthesise == 200

    def test_idempotent(self, tmp_path):
        """Writing twice produces identical files."""
        p1 = tmp_path / "a.toml"
        p2 = tmp_path / "b.toml"
        write_config(backend="local", local_model="m", path=p1)
        write_config(backend="local", local_model="m", path=p2)
        assert p1.read_text() == p2.read_text()

    def test_returns_path_object(self, tmp_path):
        result = write_config(path=tmp_path / "test.toml")
        assert isinstance(result, type(tmp_path / "x"))


# ── _DEFAULT_MODEL_DIR ────────────────────────────────────────


class TestModelDir:
    def test_points_to_repo_models(self):
        import llm_setup

        assert llm_setup._DEFAULT_MODEL_DIR.name == "models"
        assert llm_setup._DEFAULT_MODEL_DIR.parent == llm_setup._REPO_DIR


# ── Missing patterns: negative, pipeline ──────────────────────


class TestLLMSetupEdgeCases:
    def test_negative_vram(self):
        assert recommend_model(vram=-10, ram=-10) is None

    def test_float_boundaries(self):
        assert recommend_model(vram=7.999, ram=0) is not None
        assert (
            recommend_model(vram=3.999, ram=0) is None
            or recommend_model(vram=3.999, ram=16) is not None
        )

    def test_write_config_feeds_load_config(self, tmp_path):
        """Pipeline: write_config → load_config roundtrip."""
        from llm_backend import load_config

        path = tmp_path / "test.toml"
        write_config(
            backend="local", local_url="http://test:9090/v1", local_model="test-3b", path=path
        )
        cfg = load_config(path)
        assert cfg.backend == "local"
        assert cfg.local_url == "http://test:9090/v1"
        assert cfg.local_model == "test-3b"


class TestLLMSetupNegativeCases:
    def test_not_enough_ram(self):
        """Insufficient hardware returns None, not a broken config."""
        result = recommend_model(vram=0, ram=2)
        assert result is None

    def test_not_enough_anything(self):
        assert recommend_model(vram=0, ram=0) is None

    def test_negative_values_not_crash(self):
        assert recommend_model(vram=-100, ram=-50) is None
