# mpl_config.py
"""Matplotlib configuration for the example notebooks."""

__all__ = [
    "COLORS",
]

import cycler
import matplotlib.pyplot as plt


COLORS = [
    "#990007",  # Red
    "#5180d7",  # Blue
    "#69aa62",  # Green
    "#984ea3",  # Purple
    "#eeb586",  # Orange
    "#ffda00",
    "#a65628",
]

plt.rc("animation", embed_limit=50)
plt.rc("axes", prop_cycle=cycler.cycler(color=COLORS))
plt.rc("axes.spines", top=False, right=False)
plt.rc("figure", dpi=200, figsize=(12, 4))
plt.rc("font", family="serif", size=16)
plt.rc("legend", frameon=False)
plt.rc("text", usetex=True)
