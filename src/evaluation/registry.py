"""
src/evaluation/registry.py
Dataset metadata registry for evaluation.
    
Centralises resolution and capacity lookups so that evaluate.py and any
future scripts share a single source of truth. Adding a new dataset only
requires one entry in each dict below.
"""

# Temporal resolution of each dataset in minutes.
DATASET_RESOLUTION: dict[str, int] = {
    "30651":           5,
    "43458":           5,
    "75354":           5,
    "76016":           5,
    "csg_wind_5":     15,
    "gefc12_wind_7":  60,
    "gefc14_wind_10": 60,
    "penmanshiel_15": 10,
}

# Installed capacity of each dataset in MW.
DATASET_CAPACITY: dict[str, float] = {
    "30651":           16.0,
    "43458":           16.0,
    "75354":           16.0,
    "76016":            4.0,
    "csg_wind_5":      35.0,
    "gefc12_wind_7":    1.0,
    "gefc14_wind_10":   1.0,
    "penmanshiel_15": 2080.0,
}


def get_resolution(dataset_name: str) -> int:
    """
    Return the temporal resolution (minutes) for a given dataset.

    Performs an exact match first, then a substring match to support
    dataset names with suffixes (e.g. '30651_fold0').

    Raises:
        KeyError: If no matching entry is found.
    """
    if dataset_name in DATASET_RESOLUTION:
        return DATASET_RESOLUTION[dataset_name]
    for key, res in DATASET_RESOLUTION.items():
        if key in dataset_name:
            return res
    raise KeyError(
        f"No resolution entry found for dataset '{dataset_name}'. "
        f"Available keys: {list(DATASET_RESOLUTION.keys())}"
    )


def get_capacity(dataset_name: str) -> float:
    """
    Return the installed capacity (MW) for a given dataset.

    Performs an exact match first, then a substring match.

    Raises:
        KeyError: If no matching entry is found.
    """
    if dataset_name in DATASET_CAPACITY:
        return DATASET_CAPACITY[dataset_name]
    for key, cap in DATASET_CAPACITY.items():
        if key in dataset_name:
            return cap
    raise KeyError(
        f"No capacity entry found for dataset '{dataset_name}'. "
        f"Available keys: {list(DATASET_CAPACITY.keys())}"
    )


def get_pred_len(dataset_name: str, horizon_h: int) -> int:
    """
    Convert a horizon in hours to a number of time steps.

    Args:
        dataset_name: Dataset identifier used to look up resolution.
        horizon_h:    Forecast horizon in hours.

    Returns:
        Number of prediction steps: ceil(horizon_h * 60 / resolution).

    Raises:
        KeyError: If dataset_name is not registered.
    """
    import math
    resolution = get_resolution(dataset_name)
    return math.ceil(horizon_h * 60 / resolution)