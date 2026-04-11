"""
Plugin registry for discovering and loading plugins from config.

The registry is the only component that directly imports plugin classes.
Core pipeline code accesses plugins via registry.get_indicator(name), etc.

See .claude/skills/plugin-author/SKILL.md for plugin development guidance.
"""

from importlib import import_module
from pathlib import Path
from typing import Dict, List

import yaml

from src.plugins.base import DataEnricher, IndicatorPlugin, SignalFilter, SmoothingPlugin


class PluginRegistry:
    """
    Discovers and stores plugin instances loaded from config/plugins.yaml.

    Plugins are loaded lazily on first access. Configuration determines which
    plugins are active — disabled plugins are ignored.

    Usage:
        registry = PluginRegistry()
        registry.discover_plugins("config/plugins.yaml")
        rsi = registry.get_indicator("rsi")
        all_indicators = registry.get_all_indicators()
    """

    def __init__(self) -> None:
        self._indicators: Dict[str, IndicatorPlugin] = {}
        self._smoothers: Dict[str, SmoothingPlugin] = {}
        self._enrichers: Dict[str, DataEnricher] = {}
        self._filters: Dict[str, SignalFilter] = {}

    def discover_plugins(self, config_path: str) -> None:
        """
        Load all active plugins from the config file.

        Args:
            config_path: Path to config/plugins.yaml.
        """
        with open(config_path) as f:
            config = yaml.safe_load(f)

        for entry in config.get("indicators", {}).get("enabled", []):
            if entry.get("active", False):
                plugin = self._instantiate(entry["class"])
                self._indicators[entry["name"]] = plugin

        for entry in config.get("smoothers", {}).get("enabled", []):
            if entry.get("active", False):
                plugin = self._instantiate(entry["class"])
                self._smoothers[entry["name"]] = plugin

        for entry in config.get("enrichers", {}).get("enabled", []):
            if entry.get("active", False):
                plugin = self._instantiate(entry["class"])
                self._enrichers[entry["name"]] = plugin

        for entry in config.get("filters", {}).get("enabled", []):
            if entry.get("active", False):
                plugin = self._instantiate(entry["class"])
                self._filters[entry["name"]] = plugin

    def _instantiate(self, class_path: str) -> object:
        """Import and instantiate a plugin class from its fully-qualified path."""
        module_path, class_name = class_path.rsplit(".", 1)
        module = import_module(module_path)
        cls = getattr(module, class_name)
        return cls()

    def get_indicator(self, name: str) -> IndicatorPlugin:
        if name not in self._indicators:
            raise KeyError(f"No indicator plugin registered with name '{name}'")
        return self._indicators[name]

    def get_smoother(self, name: str) -> SmoothingPlugin:
        if name not in self._smoothers:
            raise KeyError(f"No smoother plugin registered with name '{name}'")
        return self._smoothers[name]

    def get_enricher(self, name: str) -> DataEnricher:
        if name not in self._enrichers:
            raise KeyError(f"No enricher plugin registered with name '{name}'")
        return self._enrichers[name]

    def get_filter(self, name: str) -> SignalFilter:
        if name not in self._filters:
            raise KeyError(f"No filter plugin registered with name '{name}'")
        return self._filters[name]

    def get_all_indicators(self) -> List[IndicatorPlugin]:
        return list(self._indicators.values())

    def get_all_smoothers(self) -> List[SmoothingPlugin]:
        return list(self._smoothers.values())

    def get_all_enrichers(self) -> List[DataEnricher]:
        return list(self._enrichers.values())

    def get_filters_by_stage(self, stage: str) -> List[SignalFilter]:
        """Return filters registered for a specific pipeline stage."""
        return [f for f in self._filters.values() if f.stage == stage]

    def list_available(self) -> Dict[str, List[str]]:
        """Return a summary of all registered plugins by type."""
        return {
            "indicators": list(self._indicators.keys()),
            "smoothers": list(self._smoothers.keys()),
            "enrichers": list(self._enrichers.keys()),
            "filters": list(self._filters.keys()),
        }
