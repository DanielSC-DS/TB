"""Identify sampling activity of exploration robots from velocity telemetry.

Method
------
* Positions are rebuilt by integrating the reported velocity from the initial
  position in zoneX_env.json (x[k] = x[k-1] + v[k]*dt).
* The reported velocity is corrupted by sporadic noise, but many samples are
  clean.  A robot moves between waypoints with a PD law  m*a = Kp*e - Kd*v,
  which with semi-implicit Euler gives the exact recursion
      v[n+1] = (2 - k*dt^2 - c*dt) v[n] - (1 - c*dt) v[n-1],
  k = Kp/m, c = Kd/m.  For every leg, (k, c) is found by RANSAC over clean
  sample triples, accepting only fits with c/k ~ Kd/Kp (constant ~1.27), so
  m ~ 1/k for each leg.
* A leg starts when the robot is (almost) at rest and its acceleration
  changes abruptly (new waypoint).  Rebounds from robot contacts are rejected
  because the robot was not at rest before them.
* At each waypoint the relative mass may change: an increase is a sample
  picked at that position, a decrease a sample dropped.
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
REST_SPEED = 0.012   # speed below which the robot is considered at rest
ACC_JUMP = 0.1       # acceleration change marking a new waypoint
FIT_TOL = 2e-5       # max residual of a clean sample triple
RATIO = (1.2, 1.36)  # accepted range of c/k = Kd/Kp
MASS_TOL = 0.10      # relative mass change regarded as a real event
LINK_DIST = 1.0


def load(zdir, zone, aid, x0):
    d = pd.read_csv(os.path.join(zdir, f"{zone}_agent_{aid}.csv"))
    V = d[["vx", "vy"]].values
    dt = float(np.median(np.diff(d.time_s))) if len(d) > 1 else DT
    X = np.asarray(x0, float) + np.vstack([[0, 0], np.cumsum(V[1:], 0) * dt])
    return X, V, dt


def leg_starts(V, dt):
    s = np.hypot(*V.T)
    A = np.diff(V, axis=0) / dt
    out = []
    for k in range(5, len(A) - 5):
        if s[k] > 0.06:
            continue
        jump = np.median(A[k:k + 4], axis=0) - np.median(A[k - 4:k], axis=0)
        if np.hypot(*jump) <= ACC_JUMP:
            continue
        j = k - 3 + int(np.argmin(s[k - 3:k + 4]))
        if s[j] < REST_SPEED and s[j - 3] < REST_SPEED and (not out or j - out[-1] >= 10):
            out.append(j)
    return out


def leg_mass(V, s, e, dt):
    """Relative mass 1/k of the leg from clean PD triples (None if not enough data)."""
    idx = np.arange(s + 2, e - 1)
    if len(idx) < 10:
        return None
    A = np.stack([V[idx], V[idx - 1]], -1)
    Y = V[idx + 1]
    best = None
    for i in range(len(idx) - 1):
        M = np.r_[A[i], A[i + 1]]
        y = np.r_[Y[i], Y[i + 1]]
        if np.linalg.cond(M) > 1e9:
            continue
        sol, *_ = np.linalg.lstsq(M, y, rcond=None)
        if np.abs(M @ sol - y).max() > FIT_TOL:
            continue
        c = (1 + sol[1]) / dt
        k = (2 - sol[0] - c * dt) / dt ** 2
        if k <= 0 or not RATIO[0] < c / k < RATIO[1]:
            continue
        inl = np.abs(np.einsum("nij,j->ni", A, sol) - Y).max(1) < FIT_TOL
        if best is None or inl.sum() > best.sum():
            best = inl
    if best is None:
        return None
    sol, *_ = np.linalg.lstsq(A[best].reshape(-1, 2), Y[best].reshape(-1), rcond=None)
    c = (1 + sol[1]) / dt
    k = (2 - sol[0] - c * dt) / dt ** 2
    return 1.0 / k


def agent_events(X, V, dt):
    bnd = [0] + leg_starts(V, dt) + [len(V) - 1]
    legs = []
    for s, e in zip(bnd[:-1], bnd[1:]):
        m = leg_mass(V, s, e, dt)
        if m is not None:
            legs.append(dict(start=s, mass=m, pos=X[s]))
    events = []
    for a, b in zip(legs[:-1], legs[1:]):
        dm = (b["mass"] - a["mass"]) / a["mass"]
        if abs(dm) > MASS_TOL:
            events.append(dict(pos=b["pos"], dm=dm, step=b["start"]))
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
    with open(os.path.join(zdir, f"{zone}_env.json")) as fh:
        env = json.load(fh)
    files = glob.glob(os.path.join(zdir, f"{zone}_agent_*.csv"))
    picks = []  # (agent, position)
    counts = {}
    for f in sorted(files, key=lambda p: int(re.search(r"_agent_(\d+)\.csv$", p).group(1))):
        aid = int(re.search(r"_agent_(\d+)\.csv$", f).group(1))
        X, V, dt = load(zdir, zone, aid, env["initial_positions"][aid])
        counts[aid] = 0
        for e in agent_events(X, V, dt):
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
    result = {"more_samples": more, "promising_region": [], "picked_promising_samples": []}
    if groups:
        g = max(groups, key=len)
        center = pts[g].mean(axis=0)
        per = {}
        for i in g:
            per[picks[i][0]] = per.get(picks[i][0], 0) + 1
        top = max(per.values())
        result["promising_region"] = [round(float(center[0]), 2), round(float(center[1]), 2)]
        result["picked_promising_samples"] = sorted(a for a, c in per.items() if c == top)
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
