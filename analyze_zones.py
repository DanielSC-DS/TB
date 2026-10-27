"""Identify sampling activity of exploration robots from velocity telemetry.

Method
------
* Only (noisy) velocity is logged.  The noise is multiplicative (~3 % of the
  speed), so positions are rebuilt by integrating it,
  x[k+1] = x[k] + v[k+1]*dt, with every robot starting at the origin.
* Each leg towards a waypoint T follows a saturated velocity-tracking law
      v_cmd = vmax*tanh(g*|e|/vmax) * e/|e|,   e = T - x
      a     = c * (v_cmd - v)
  g and vmax are robot constants; only c = Kd/m changes from leg to leg, so
  the relative mass carried during a leg is m ~ 1/c.
* Legs are split where the acceleration jumps (a new target is set).  Each
  leg is fitted by simulating the law above.  A step in mass between two
  legs happens at the waypoint just reached: an increase is a sample picked
  there, a decrease a sample dropped.  Samples have different sizes, so each
  mass step counts as one sample whatever its size.
* Picked-sample positions of all robots are grouped by single linkage with a
  1 m threshold; groups need more than two points.  The group with the most
  samples is the promising region and its centroid is reported.
"""
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

DT = 0.05
NOISE = 0.03        # relative velocity noise
MASS_TOL = 0.3      # mass step, relative to the empty robot, regarded as an event
LINK_DIST = 1.0


def nelder_mead(f, x0, step=0.2, iters=4000, tol=1e-12):
    n = len(x0)
    pts = [np.array(x0, float)]
    for i in range(n):
        p = np.array(x0, float)
        p[i] = p[i] * (1 + step) if p[i] != 0 else step
        pts.append(p)
    vals = [f(p) for p in pts]
    for _ in range(iters):
        o = np.argsort(vals)
        pts = [pts[i] for i in o]
        vals = [vals[i] for i in o]
        if abs(vals[-1] - vals[0]) < tol * (abs(vals[0]) + 1e-20):
            break
        c = np.mean(pts[:-1], axis=0)
        xr = c + (c - pts[-1]); fr = f(xr)
        if fr < vals[0]:
            xe = c + 2 * (c - pts[-1]); fe = f(xe)
            pts[-1], vals[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < vals[-2]:
            pts[-1], vals[-1] = xr, fr
        else:
            xc = c + 0.5 * (pts[-1] - c); fc = f(xc)
            if fc < vals[-1]:
                pts[-1], vals[-1] = xc, fc
            else:
                for i in range(1, n + 1):
                    pts[i] = pts[0] + 0.5 * (pts[i] - pts[0]); vals[i] = f(pts[i])
    i = int(np.argmin(vals))
    return pts[i], vals[i]


def segments(V):
    dv = np.diff(V, axis=0)
    jump = np.hypot(*np.diff(dv, axis=0).T)
    speed = np.hypot(*V.T)
    score = jump / (0.1 * speed[1:-1] + 0.002)
    bnd = [0]
    for i in np.where(score > 3)[0] + 1:
        if i - bnd[-1] > 15:
            bnd.append(int(i))
    return bnd + [len(V) - 1]


def simulate(c, g, vmax, T, x0, v0, n):
    x, v, out = x0.copy(), v0.copy(), [v0.copy()]
    for _ in range(n):
        e = T - x
        ne = np.hypot(*e)
        vc = vmax * np.tanh(g * ne / vmax) * e / ne if ne > 1e-12 else 0 * e
        v = v + c * (vc - v) * DT
        x = x + v * DT
        out.append(v.copy())
    return np.array(out)


def fit_leg(X, V, s, e, p0, fixed=None):
    Vo = V[s:e + 1]
    w = 1 / (NOISE * np.hypot(*Vo.T)[:, None] + 3e-4)

    def cost(p):
        c, g, vmax, tx, ty = p if fixed is None else (p[0], *fixed, p[1], p[2])
        if c <= 0 or g <= 0 or vmax <= 0:
            return 1e30
        return np.sum(((simulate(c, g, vmax, np.array([tx, ty]), X[s], V[s], e - s) - Vo) * w) ** 2)

    best = None
    for _ in range(3):
        p, f = nelder_mead(cost, p0 if best is None else best[0])
        if best is None or f < best[1]:
            best = (p, f)
    p = best[0]
    return p if fixed is None else np.array([p[0], *fixed, p[1], p[2]])


def agent_events(path):
    d = pd.read_csv(path)
    V = d[["vx", "vy"]].values
    X = np.vstack([[0.0, 0.0], np.cumsum(V[1:] * DT, axis=0)])
    bnd = segments(V)
    legs = [(s, e) for s, e in zip(bnd[:-1], bnd[1:]) if e - s > 15]

    free = [fit_leg(X, V, s, e, [3, 0.8, 0.8, *X[e]]) for s, e in legs]
    fixed = (np.median([p[1] for p in free]), np.median([p[2] for p in free]))
    fits = [fit_leg(X, V, s, e, [p[0], p[3], p[4]], fixed) for (s, e), p in zip(legs, free)]
    mass = [1.0 / p[0] for p in fits]

    events = []
    for i in range(len(fits) - 1):
        dm = (mass[i + 1] - mass[i]) / mass[0]
        if abs(dm) > MASS_TOL:
            events.append(dict(pos=fits[i][3:5], dm=dm, step=legs[i][1]))
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
        counts[aid] = 0
        for e in agent_events(f):
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
    result = {"more_samples": more, "promising_region": [],
              "took_promising_samples": [], "picked_promising_samples": []}
    if groups:
        g = max(groups, key=len)
        center = pts[g].mean(axis=0)
        per = {}
        for i in g:
            per[picks[i][0]] = per.get(picks[i][0], 0) + 1
        top = max(per.values())
        result["promising_region"] = [round(float(center[0]), 2), round(float(center[1]), 2)]
        result["took_promising_samples"] = sorted(a for a, c in per.items() if c == top)
        # the task's example names this field differently; write both
        result["picked_promising_samples"] = result["took_promising_samples"]
        print(f"{zone} picks per agent {counts}; groups {[len(x) for x in groups]}; "
              f"promising group {len(g)} pts, per agent {per}")

    out = os.path.join(zdir, f"{zone}_explored.json")
    with open(out, "w") as fh:
        json.dump(result, fh, indent=1)
    print(out, result)


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "zones")
    for z in sorted(glob.glob(os.path.join(root, "zone*"))):
        if os.path.isdir(z):
            analyze_zone(z)
