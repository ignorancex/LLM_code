#!/usr/bin/env python3
import functools
import os
import re
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

from .common_functions import Cosmology, FigureHandler

_CONFIG_PATHS = [
    Path(__file__).parent.parent / "settings.yaml",  # Original location
    Path(__file__).parent / "settings.yaml",  # Package-included
    Path(os.getenv("GCGMICS_CONFIG", "settings.yaml")),  # Env-specified
]

for path in _CONFIG_PATHS:
    if path and path.exists():
        _CONFIG_PATH = path
        break


def find_placeholders_with_path(data):
    """Find placeholder locations using full path support"""
    results = defaultdict(list)

    def traverse(obj, path):
        if isinstance(obj, str):
            matches = re.findall(r"\{([^\}]+)\}", obj)
            if matches:
                results[path].extend(matches)
        elif isinstance(obj, dict):
            for key, value in obj.items():
                new_path = f"{path}.{key}" if path else key
                traverse(value, new_path)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                new_path = f"{path}[{i}]"
                traverse(item, new_path)

    traverse(data, "")
    return dict(results)


def substitute_placeholders(data, context):
    """Substitute placeholders using context with full path support"""
    if isinstance(data, str):

        def replace(match):
            key = match.group(1)
            return str(context.get(key, match.group(0)))

        return re.sub(r"\{([^\}]+)\}", replace, data)
    elif isinstance(data, dict):
        return {k: substitute_placeholders(v, context) for k, v in data.items()}
    elif isinstance(data, list):
        return [substitute_placeholders(item, context) for item in data]
    return data


def extract_all_anchors(data):
    """Extract anchors with full paths as keys"""
    anchors = {}

    def traverse(obj, path):
        if isinstance(obj, dict):
            for key, value in obj.items():
                new_path = f"{path}.{key}" if path else key
                if not isinstance(value, (dict, list)):
                    anchors[new_path] = value
                traverse(value, new_path)
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                new_path = f"{path}[{i}]"
                if not isinstance(item, (dict, list)):
                    anchors[new_path] = item
                traverse(item, new_path)

    traverse(data, "")
    return anchors


def dict_to_namespace(data):
    """Convert dictionary to nested SimpleNamespace"""
    if isinstance(data, dict):
        return SimpleNamespace(**{k: v for k, v in data.items()})
        # return data
    elif isinstance(data, list):
        return [dict_to_namespace(item) for item in data]
    return data


@functools.lru_cache(maxsize=1)
def load_settings():
    """Load and process settings with caching"""
    try:
        with open(_CONFIG_PATH, "r") as f:
            data = yaml.safe_load(f)

        context = extract_all_anchors(data)
        processed_data = substitute_placeholders(data, context)

        # Analysis settings
        default_analysis_dict = {
            "cosmology_parameters": {
                "omega_M": 0.3156,
                "omega_lambda": 0.6844,
                "h": 0.6727,
            }  # PLANCK 2015
        }
        if "Analysis" not in processed_data:
            processed_data.update({"Analysis": default_analysis_dict})
        else:
            analysis_keys = processed_data["Analysis"].keys()
            for key, value in default_analysis_dict.items():
                if key not in analysis_keys:
                    processed_data["Analysis"].update({key: value})
                else:
                    pass
        cosmology_object = Cosmology(processed_data["Analysis"]["cosmology_parameters"])
        processed_data["Analysis"].update({"cosmology_object": cosmology_object})

        # Plotting settings
        default_plotting_dict = {
            "tlb_lim": [0, 12.0],
            "z_lim": [0.0, 8.0],
            "fine_spacing": 0.2,
            "z_cut": 6.5,
            "coarse_spacing": 1.0,
        }
        if "Plotting" not in processed_data:
            processed_data.update({"Plotting": default_plotting_dict})
        else:
            plotting_keys = processed_data["Plotting"].keys()
            for key, value in default_plotting_dict.items():
                if key not in plotting_keys:
                    processed_data["Plotting"].update({key: value})
                else:
                    pass

        all_redshifts = np.arange(
            *processed_data["Plotting"]["z_lim"],
            processed_data["Plotting"]["fine_spacing"],
        )
        major_redshifts = all_redshifts[~(all_redshifts % 1).astype(bool)]
        minor_redshifts = all_redshifts[(all_redshifts % 1).astype(bool)]
        minor_redshifts = np.concatenate(
            (
                minor_redshifts[minor_redshifts < processed_data["Plotting"]["z_cut"]],
                np.arange(
                    processed_data["Plotting"]["z_cut"],
                    processed_data["Plotting"]["z_lim"][-1],
                    processed_data["Plotting"]["coarse_spacing"],
                ),
            )
        )

        axis_rescale = np.abs(np.diff(processed_data["Plotting"]["tlb_lim"])[0])

        supplementary_plotting_dict = {
            "all_redshifts": all_redshifts,
            "major_redshifts": major_redshifts,
            "minor_redshifts": minor_redshifts,
            "axis_rescale": axis_rescale,
        }
        plotting_keys = processed_data["Plotting"].keys()
        for key, value in supplementary_plotting_dict.items():
            if key not in plotting_keys:
                processed_data["Plotting"].update({key: value})
            else:
                pass

        figure_handler = FigureHandler(
            major_redshifts=processed_data["Plotting"]["major_redshifts"],
            minor_redshifts=processed_data["Plotting"]["minor_redshifts"],
            tlb_lim=processed_data["Plotting"]["tlb_lim"],
            cos_obj=cosmology_object,
        )
        processed_data["Plotting"].update({"figure_handler": figure_handler})

        return dict_to_namespace(processed_data)
    except Exception as e:
        raise RuntimeError(f"Failed to load settings from {_CONFIG_PATH}: {e}") from e


def get(path, default=None):
    """Get a setting by dot path with list support"""
    settings = load_settings()
    current = settings
    parts = path.split(".")

    try:
        for part in parts:
            if "[" in part and "]" in part:
                # Handle list indices: "items[0]"
                key, index_str = part.split("[", 1)
                index = int(index_str.rstrip("]"))
                current = getattr(current, key)
                try:
                    current = current[index]
                except IndexError:
                    return default
            else:
                try:
                    current = getattr(current, part)
                except AttributeError:
                    return default
        return current
    except (TypeError, ValueError):
        return default


def __getattr__(name):
    """Retrieve top-level settings attributes dynamically."""
    s = load_settings()
    try:
        return getattr(s, name)
    except AttributeError:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'") from None


def __dir__():
    """Include top-level settings keys in module directory listing."""
    base_attrs = set(globals().keys())
    s = load_settings()
    settings_attrs = set(vars(s).keys())
    return sorted(base_attrs | settings_attrs)


# Load settings immediately to populate __dir__ (optional but helps introspection)
try:
    s = load_settings()
    for key in vars(s):
        if key not in globals():
            globals()[key] = getattr(s, key)
except Exception:
    pass  # Graceful degradation if loading fails
