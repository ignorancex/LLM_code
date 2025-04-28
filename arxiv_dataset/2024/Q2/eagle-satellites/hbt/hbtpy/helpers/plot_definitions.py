"""
Global plotting definitions (bad idea, I know!)
"""
import numpy as np

ccolor = "0.2"
scolor = "k"

max_events = {
    f"max_{m}": "max,sub" if m == "Mbound" else f"max,{m[1:]}"
    for m in ("Mbound", "Mstar", "Mgas", "Mdm")
}
event_names = {
    "last_infall": "acc",
    "first_infall": "infall",
    "birth": "birth",
    "cent": "cent",
    "sat": "sat",
    **max_events,
}

massnames = ["Mbound", "Mstar", "Mdm", "Mgas", "Mass", "M200", "MVir"]
units = {
    "mass": r"\mathrm{M}_\odot",
    "time": "\mathrm{Gya}",
    "distance": "h^{-1}\mathrm{Mpc}",
}
xbins = {
    "ComovingMostBoundDistance": np.logspace(-1.7, 0.5, 9),
    #'logComovingMostBoundDistance': np.logspace(-2, 0.5, 9),
    #'PhysicalMostBoundPeculiarVelocity': np.
    "M200Mean": np.logspace(13, 14.85, 6),
    "mu": np.logspace(-5, 0, 9),
    "Mstar": np.logspace(8.9, 11.3, 9),
    "Mgas": np.logspace(8, 11, 10),
    "Mbound": np.logspace(9, 13, 10),
    #'time': np.append(np.arange(2), np.arange(2, 13.5, 2)),
    "time": np.arange(0, 14.5, 2),
    #'time': np.arange(0, 14.5, 1),
    "z": np.array([0, 0.5, 1, 1.5, 2, 3, 5]),
    # some common ratios
    "ComovingMostBoundDistance/R200Mean": np.logspace(-1.7, 0.7, 10),
    # it seems there's nothing beyond 4R/R200m
    "ComovingMostBoundDistance/R200MeanComoving": np.append(
        [0.02], np.logspace(-1.5, 0.6, 9)
    ),
    # np.logspace(-1.7, 0.5, 9)
    #'PhysicalMostBoundPeculiarVelocity/PhysicalMostBoundHostVelocityDispersion'
    "history:sat:time-history:first_infall:time": np.linspace(0, 10, 5),
}
for event in event_names.keys():
    xbins[f"history:{event}:time"] = np.linspace(0, 11, 6)
ylims = {
    "Mbound/history:first_infall:Mbound": (5e-3, 2),
    "Mdm/history:first_infall:Mdm": (5e-4, 2),
    "Mstar/history:first_infall:Mstar": (0.1, 20),
}
binlabel = {
    #'ComovingMostBoundDistance': '$R_\mathrm{com}$ ($h^{-1}$Mpc)',
    "ComovingMostBoundDistance": "R",
    "R200Mean": r"R_\mathrm{200m}",
    "R200MeanComoving": r"R_\mathrm{200m}",
    "M200Mean": r"M_\mathrm{200m}",
    "mu": r"\mu",
    "Mbound": r"m_\mathrm{sub}",
    "Mstar": "m_{\u2605}",
    "Mdm": r"m_\mathrm{DM}",
    "Mgas": r"m_\mathrm{gas}",
    "LastMaxMass": "m_\mathrm{sub,max}",
    "time": "t_\mathrm{lookback}",
    "z": "z",
}
for m in ("Mbound", "Mstar", "Mdm", "Mgas"):
    name = "sub" if m == "Mbound" else m[1:]
    binlabel[f"max_{m}"] = rf"{binlabel[m]}^\mathrm{{max,{name}}}"
_xx = "ComovingMostBoundDistance"
for i in "012":
    p = "p" + i.replace("0", "")
    xbins[f"{_xx}{i}"] = xbins[_xx]
    binlabel[f"{_xx}{i}"] = f"{binlabel[_xx]}_\mathrm{{{p}}}"
    xbins[f"{_xx}{i}/R200Mean"] = xbins[f"{_xx}/R200Mean"]
for p, n in zip(("01", "02", "12"), ("xy", "xz", "yz")):
    xbins[f"{_xx}{p}"] = xbins[_xx]
    binlabel[f"{_xx}{p}"] = f"{binlabel[_xx]}_{{{n}}}"
for event, event_label in event_names.items():
    h = f"history:{event}"
    elabel = f"\mathrm{{{event_label}}}"
    binlabel[f"{h}:Mbound"] = rf"m_\mathrm{{sub}}^{elabel}"
    binlabel[f"{h}:Mstar"] = f"m_{{\u2605}}^{elabel}"
    binlabel[f"{h}:Mdm"] = rf"m_\mathrm{{DM}}^{elabel}"
    binlabel[f"{h}:Mgas"] = rf"m_\mathrm{{gas}}^{elabel}"
    binlabel[f"{h}:time"] = rf"t_\mathrm{{lookback}}^{elabel}"
    binlabel[f"{h}:z"] = rf"z^{elabel}"
    xbins[f"{h}:Mbound/{h}:Mstar"] = np.logspace(1, 2.7, 6)
    xbins[f"Mstar/{h}:Mbound"] = np.logspace(-4, -1, 6)
    xbins[f"{h}:z"] = xbins["z"]
axlabel = {}
for key, label in binlabel.items():
    ismass = np.any([mn in key for mn in massnames])
    if ismass or "time" in key:
        label = label.replace("$", "")
        unit_key = "mass" if ismass else ("time" if "time" in key else "distance")
        un = units[unit_key].replace("$", "")
        axlabel[key] = rf"${label}\,({un})$"
    elif "distance" in key.lower():
        un = units["distance"].replace("$", "")
        axlabel[key] = rf"${label}\,({un})$"
    else:
        axlabel[key] = label
