"""Identify sampling activity of exploration robots from position telemetry.

Method
------
* The logged velocity is noisy, but positions are integrated exactly
  (x[k+1] = x[k] + v[k+1]*dt), so velocity and acceleration are rebuilt
  from finite differences of position.
* Near each waypoint the robots follow a PD law  a = (Kp*(T - x) - Kd*v) / m.
  Fitting a = k*(T - x) - c*v on short windows gives k = Kp/m, so the
  relative mass of the robot during each leg is m ~ 1/k (c/k stays constant,
  which confirms only the mass changes).
* A robot "arrives" at a waypoint (|e| small, speed ~0) and immediately heads
  for the next one; at that instant its mass may change.  A mass increase is
  a sample picked at that position, a decrease is a sample dropped.
* Picked-sample positions of all robots are grouped by single linkage with a
  1 m threshold; groups need more than two points.  The largest group is the
  promising region and its centroid is reported.
"""
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

DT = 0.05
WIN = 8
FIT_TOL = 1e-5      # max residual for a window to be considered "clean PD"
MASS_TOL = 0.05     # relative mass change regarded as a real event
LINK_DIST = 1.0


def window_fit(X, V, A, ks):
    M, r = [], []
    for k in ks:
        M.append([1, 0, -X[k + 1, 0], -V[k, 0]]); r.append(A[k, 0])
        M.append([0, 1, -X[k + 1, 1], -V[k, 1]]); r.append(A[k, 1])
    M, r = np.array(M), np.array(r)
    sol, *_ = np.linalg.lstsq(M, r, rcond=None)
    return sol[2], sol[3], sol[:2] / sol[2], np.abs(M @ sol - r).max()


def agent_events(path):
    d = pd.read_csv(path)
    X = d[["x", "y"]].values
    dt = float(np.median(np.diff(d.time_s))) if len(d) > 1 else DT
    V = np.diff(X, axis=0) / dt
    A = np.diff(V, axis=0) / dt
    n = len(A)

    # target switches show up as discontinuities in the jerk
    J = np.diff(A, axis=0)
    jj = np.hypot(*np.diff(J, axis=0).T)
    cand = [i for i in np.where(jj > 0.02)[0] if i > 10]
    bnd = [0]
    for i in cand:
        if i - bnd[-1] > 20:
            bnd.append(i)
    bnd.append(n)

    legs = []
    for s, e in zip(bnd[:-1], bnd[1:]):
        best = None
        for st in range(max(s + 3, e - 100), e - WIN - 3):
            res = window_fit(X, V, A, range(st, st + WIN))
            if res[3] < FIT_TOL and (best is None or res[3] < best[3]):
                best = res
        if best is None:
            continue
        # a pure mass change scales Kp and Kd alike, so c/k must stay constant;
        # legs disturbed by contacts or far-field behaviour break that ratio
        ratio = best[1] / best[0]
        if legs and abs(ratio / legs[0]["ratio"] - 1) > 0.1:
            continue
        legs.append(dict(end=e, mass=1.0 / best[0], ratio=ratio, pos=X[min(e + 1, len(X) - 1)]))

    events = []
    for a, b in zip(legs[:-1], legs[1:]):
        dm = (b["mass"] - a["mass"]) / legs[0]["mass"]
        if abs(dm) > MASS_TOL:
            events.append(dict(pos=a["pos"], dm=dm, step=a["end"] + 1))
    return events


def clusters(points):
    n = len(points)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if np.hypot(*(points[i] - points[j])) <= LINK_DIST:
                parent[find(i)] = find(j)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [g for g in groups.values() if len(g) > 2]


def analyze_zone(zdir):
    zone = os.path.basename(os.path.normpath(zdir))
    files = glob.glob(os.path.join(zdir, f"{zone}_agent_*.csv"))
    picks = []  # (agent, position)
    counts = {}
    for f in sorted(files):
        aid = int(re.search(r"_agent_(\d+)\.csv$", f).group(1))
        ev = agent_events(f)
        counts[aid] = 0
        for e in ev:
            if e["dm"] > 0:
                counts[aid] += 1
                picks.append((aid, e["pos"]))
            print(f"{zone} agent {aid} step {e['step']:5d} "
                  f"{'pick' if e['dm'] > 0 else 'drop'} dm={e['dm']:+.2f} "
                  f"at ({e['pos'][0]:.3f},{e['pos'][1]:.3f})")

    best = max(counts.values())
    more = sorted(a for a, c in counts.items() if c == best)

    pts = np.array([p for _, p in picks])
    groups = clusters(pts) if len(pts) else []
    result = {"more_samples": more, "promising_region": [], "took_promising_samples": []}
    if groups:
        g = max(groups, key=len)
        center = pts[g].mean(axis=0)
        per = {}
        for i in g:
            per[picks[i][0]] = per.get(picks[i][0], 0) + 1
        top = max(per.values())
        result["promising_region"] = [round(float(center[0]), 2), round(float(center[1]), 2)]
        result["took_promising_samples"] = sorted(a for a, c in per.items() if c == top)
        print(f"{zone} picks per agent {counts}; promising cluster {len(g)} pts, per agent {per}")

    out = os.path.join(zdir, f"{zone}_explored.json")
    with open(out, "w") as fh:
        json.dump(result, fh, indent=1)
    print(out, result)


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "zones")
    for z in sorted(glob.glob(os.path.join(root, "zone*"))):
        if os.path.isdir(z):
            analyze_zone(z)
