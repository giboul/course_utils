#!/usr/bin/env python
from pathlib import Path
from cycler import cycler
import numpy as np
from matplotlib import pyplot as plt


params = dict(
    rocks = dict(),
    click_count = 0,
    freq = 10e6,
    # freq = float(input("Sample frequency [Hz]: ")),
)

def pick(event):
    if event.dblclick is True and event.inaxes is not None:
        params["rocks"][file.stem][params["click_count"]] = f"{event.xdata:.3e}"
        print(f"{file.stem}, {params['rocks'][file.stem]}", flush=True)
        params["click_count"] = (params["click_count"]+1)%3
def hover(event):
    x = event.xdata
    ax = event.inaxes
    if ax is not None:
        vbar.set_xy1((x, 0))
        ax.draw_artist(vbar)
        ax.figure.canvas.draw()

print("file_name,t_i,t_p,t_s", flush=True)
files = list(sorted(Path().glob(f"Ex_VpVs_*.txt")))
for file in files:

    params["click_count"] = 0
    params["rocks"][file.stem] = [0., 0., 0.]

    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    ax.set_xlabel("time [s]")
    ax.set_ylabel("Voltage [V]")
    ax2.set_ylabel("Cumulated absolute Vs")

    Vi, Vp, Vs = np.loadtxt(file).T
    t = np.arange(Vp.size) / params["freq"]

    ax.plot(t, Vi)
    ax.plot(t, Vp)
    ax.plot(t, Vs)
    ax2.plot((t[:-1]+t[1:])/2, np.cumsum(np.abs(Vs))[:-1]*np.diff(t), "-.", lw=0.5)
    vbar = ax.axline((0, 0), slope=float("inf"), ls="-.", lw=0.5, c="grey", alpha=0.6)
    fig.canvas.mpl_connect("button_press_event", pick)
    fig.canvas.mpl_connect("motion_notify_event", hover)
    plt.show()

with open("picks.csv", "w") as file:
    print("file_name,t_i,t_p,t_s", file=file)
    for filename, picks in params["rocks"].items():
        ti, tp, ts = picks
        print(f"{filename},{ti},{tp},{ts}", file=file)

