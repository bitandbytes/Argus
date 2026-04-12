"""
Unit tests for PluginRegistry (src/plugins/registry.py).

Tests cover:
  - Registering and retrieving all four plugin types
  - KeyError on missing plugin names
  - list_available() and get_all_*() helpers
  - get_filters_by_stage() filtering
  - discover_plugins() loads active-only entries from a YAML config

Note: Task 1.4 indicator implementations do not exist yet.  All tests use
stub concrete classes defined in this file; monkeypatching is used for the
discover_plugins tests to bypass actual module imports.
"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd
import pytest
import yaml

from src.plugins.base import (
    DataEnricher,
    IndicatorPlugin,
    ParamSpec,
    SignalFilter,
    SmoothResult,
    SmoothingPlugin,
)
from src.plugins.registry import PluginRegistry


# ============================================================================
# Minimal concrete stub implementations
# ============================================================================


class StubIndicator(IndicatorPlugin):
    name = "stub_indicator"
    category = "trend"
    version = "1.0.0"

    def compute(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        return df

    def normalize(self, values: pd.Series) -> pd.Series:
        return values.clip(-1, 1)

    def get_tunable_params(self) -> Dict[str, ParamSpec]:
        return {}

    def get_default_params(self) -> Dict[str, Any]:
        return {}


class AnotherStubIndicator(IndicatorPlugin):
    """A second indicator used to test get_all_indicators with multiple entries."""

    name = "another_stub_indicator"
    category = "momentum"
    version = "1.0.0"

    def compute(self, df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        return df

    def normalize(self, values: pd.Series) -> pd.Series:
        return values.clip(-1, 1)

    def get_tunable_params(self) -> Dict[str, ParamSpec]:
        return {}

    def get_default_params(self) -> Dict[str, Any]:
        return {}


class StubSmoother(SmoothingPlugin):
    name = "stub_smoother"
    version = "1.0.0"

    def smooth(self, series: pd.Series, params: Dict[str, Any]) -> SmoothResult:
        zeros = pd.Series(0.0, index=series.index)
        return SmoothResult(
            smoothed=series,
            trend=series,
            velocity=zeros,
            noise_estimate=zeros,
            confidence=pd.Series(1.0, index=series.index),
        )

    def get_tunable_params(self) -> Dict[str, ParamSpec]:
        return {}


class StubEnricher(DataEnricher):
    name = "stub_enricher"
    data_type = "sentiment"
    version = "1.0.0"

    def enrich(self, ticker: str, features: Any) -> Dict[str, float]:
        return {}


class StubFilterPostMeta(SignalFilter):
    name = "stub_filter_post_meta"
    stage = "post_meta"
    version = "1.0.0"

    def filter(self, signal: Any, context: Dict[str, Any]) -> Any:
        return signal


class StubFilterPreQuant(SignalFilter):
    name = "stub_filter_pre_quant"
    stage = "pre_quant"
    version = "1.0.0"

    def filter(self, signal: Any, context: Dict[str, Any]) -> Any:
        return signal


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture()
def registry() -> PluginRegistry:
    """Return a fresh, empty PluginRegistry."""
    return PluginRegistry()


# ============================================================================
# Register & Retrieve
# ============================================================================


def test_get_indicator_returns_registered(registry: PluginRegistry) -> None:
    stub = StubIndicator()
    registry._indicators["stub_indicator"] = stub
    assert registry.get_indicator("stub_indicator") is stub


def test_get_smoother_returns_registered(registry: PluginRegistry) -> None:
    stub = StubSmoother()
    registry._smoothers["stub_smoother"] = stub
    assert registry.get_smoother("stub_smoother") is stub


def test_get_enricher_returns_registered(registry: PluginRegistry) -> None:
    stub = StubEnricher()
    registry._enrichers["stub_enricher"] = stub
    assert registry.get_enricher("stub_enricher") is stub


def test_get_filter_returns_registered(registry: PluginRegistry) -> None:
    stub = StubFilterPostMeta()
    registry._filters["stub_filter"] = stub
    assert registry.get_filter("stub_filter") is stub


# ============================================================================
# Error on missing
# ============================================================================


def test_get_indicator_missing_raises_keyerror(registry: PluginRegistry) -> None:
    with pytest.raises(KeyError, match="No indicator plugin"):
        registry.get_indicator("nonexistent")


def test_get_smoother_missing_raises_keyerror(registry: PluginRegistry) -> None:
    with pytest.raises(KeyError, match="No smoother plugin"):
        registry.get_smoother("nonexistent")


def test_get_enricher_missing_raises_keyerror(registry: PluginRegistry) -> None:
    with pytest.raises(KeyError, match="No enricher plugin"):
        registry.get_enricher("nonexistent")


def test_get_filter_missing_raises_keyerror(registry: PluginRegistry) -> None:
    with pytest.raises(KeyError, match="No filter plugin"):
        registry.get_filter("nonexistent")


# ============================================================================
# List
# ============================================================================


def test_list_available_empty(registry: PluginRegistry) -> None:
    result = registry.list_available()
    assert result == {
        "indicators": [],
        "smoothers": [],
        "enrichers": [],
        "filters": [],
    }


def test_list_available_populated(registry: PluginRegistry) -> None:
    registry._indicators["ind1"] = StubIndicator()
    registry._smoothers["sm1"] = StubSmoother()
    registry._enrichers["enc1"] = StubEnricher()
    registry._filters["flt1"] = StubFilterPostMeta()

    result = registry.list_available()

    assert "ind1" in result["indicators"]
    assert "sm1" in result["smoothers"]
    assert "enc1" in result["enrichers"]
    assert "flt1" in result["filters"]


def test_get_all_indicators(registry: PluginRegistry) -> None:
    ind1 = StubIndicator()
    ind2 = AnotherStubIndicator()
    registry._indicators["ind1"] = ind1
    registry._indicators["ind2"] = ind2

    all_inds = registry.get_all_indicators()

    assert len(all_inds) == 2
    assert ind1 in all_inds
    assert ind2 in all_inds


def test_get_filters_by_stage(registry: PluginRegistry) -> None:
    post_meta = StubFilterPostMeta()
    pre_quant = StubFilterPreQuant()
    registry._filters["post_meta"] = post_meta
    registry._filters["pre_quant"] = pre_quant

    result = registry.get_filters_by_stage("post_meta")

    assert post_meta in result
    assert pre_quant not in result


# ============================================================================
# Discover from config (monkeypatched _instantiate)
# ============================================================================


def _make_plugins_yaml(tmp_path, active: bool) -> str:
    """Write a minimal plugins YAML and return the file path."""
    config = {
        "indicators": {
            "enabled": [
                {
                    "name": "stub_indicator",
                    "class": "stub.StubIndicator",
                    "active": active,
                }
            ]
        },
        "smoothers": {"enabled": []},
        "enrichers": {"enabled": []},
        "filters": {"enabled": []},
    }
    path = tmp_path / "plugins.yaml"
    path.write_text(yaml.dump(config))
    return str(path)


def test_discover_plugins_loads_active(registry: PluginRegistry, tmp_path, monkeypatch) -> None:
    config_path = _make_plugins_yaml(tmp_path, active=True)

    stub_instance = StubIndicator()
    monkeypatch.setattr(registry, "_instantiate", lambda class_path: stub_instance)

    registry.discover_plugins(config_path)

    assert "stub_indicator" in registry.list_available()["indicators"]
    assert registry.get_indicator("stub_indicator") is stub_instance


def test_discover_plugins_skips_inactive(registry: PluginRegistry, tmp_path, monkeypatch) -> None:
    config_path = _make_plugins_yaml(tmp_path, active=False)

    monkeypatch.setattr(registry, "_instantiate", lambda class_path: StubIndicator())

    registry.discover_plugins(config_path)

    assert registry.list_available()["indicators"] == []
