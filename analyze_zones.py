"""Identify sampling activity of exploration robots from velocity telemetry.

Method
------
* Positions are rebuilt by integrating the logged velocity from the initial
  position given in zoneX_env.json (x[k] = x0 + sum(v[1..k]) * dt).
* The robots follow a cascaded controller: a commanded speed that depends only
  on the distance to the current waypoint, v_cmd(|e|), tracked by a first-order
  loop  a = (b/m) * (v_cmd(|e|) * e_hat - v).  The gain C = b/m therefore
  measures the mass carried by the robot during each leg.
* Waypoint switches show up as an abrupt jump of the acceleration while the
  robot is almost stopped; jumps that happen while another robot is close are
  contacts between robots and are ignored.
* v_cmd(|e|) is learned from the first leg of every robot (empty robot, very
  fast inner loop).  For every leg the mass is then fitted by simulating the
  leg from its initial state and matching the logged velocities.
* A mass increase at a waypoint is a sample picked there, a decrease is a
  drop.  Samples come in different sizes; picks much heavier than the typical
  one are very dense minerals.
* Picked-sample positions are grouped by single linkage (1 m); groups need
  more than two points.  Groups of dense picks are dense-mineral regions, and
  a candidate region closer than 1.5 m (closest pair of samples) to one of
  them is discarded.  The remaining group with most picks is the promising
  region; its centroid is reported.
"""
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd

LINK_DIST = 1.0         # max distance between chained samples of a region
MIN_GROUP = 3           # a region has more than two samples
DENSE_CLEARANCE = 1.5   # promising region must be at least this far from dense regions
NEAR_ROBOT = 0.5        # acceleration jumps with another robot this close are contacts
DENSE_FACTOR = 1.8      # picks this many times heavier than the median pick are dense
MIN_DM = 0.5            # mass change (in median-pick units) regarded as a real event


def load_zone(zdir):
    zone = os.path.basename(os.path.normpath(zdir))
    with open(os.path.join(zdir, f"{zone}_env.json")) as fh:
        env = json.load(fh)
    X, V, dt = {}, {}, None
    for f in glob.glob(os.path.join(zdir, f"{zone}_agent_*.csv")):
        aid = int(re.search(r"_agent_(\d+)\.csv$", f).group(1))
        d = pd.read_csv(f).sort_values("step")
        v = d[["vx", "vy"]].to_numpy(float)
        dt = float(np.median(np.diff(d.time_s))) if len(d) > 1 else 0.05
        X[aid] = np.asarray(env["initial_positions"][aid], float) + np.cumsum(v * dt, axis=0)
        V[aid] = v
    return zone, env, X, V, dt


def switches(X, V, aid):
    """Steps where the robot starts heading to a new waypoint."""
    x, v = X[aid], V[aid]
    sp = np.hypot(*v.T)
    jump = np.hypot(*(v[2:] - 2 * v[1:-1] + v[:-2]).T)
    rel = jump / (sp[1:-1] + 1e-4)
    out = []
    for k in np.where((rel > 1.5) & (jump > 0.004))[0] + 1:
        others = [np.hypot(*(X[j][k] - x[k])) for j in X if j != aid]
        if others and min(others) < NEAR_ROBOT:
            continue
        if out and k - out[-1] < 5:
            continue
        out.append(int(k))
    return out


def first_leg_gain(v, dt, n=14):
    """Inner-loop gain of the empty robot from its start-up transient."""
    sp = np.hypot(*v[:n].T)
    best = None
    for r in np.arange(0.5, 0.99, 0.0025):
        b = 1 - r ** np.arange(n)
        s = (b @ sp) / (b @ b)
        res = np.sum((sp - s * b) ** 2)
        if best is None or res < best[0]:
            best = (res, r)
    return (1 - best[1]) / dt


def smooth(y, w=7):
    p = w // 2
    return np.convolve(np.r_[np.full(p, y[0]), y, np.full(p, y[-1])], np.ones(w) / w, mode="valid")


def speed_command(X, V, dt, sw, C0):
    """Commanded speed as a function of the distance to the waypoint."""
    pts = []
    for aid in X:
        if not sw[aid]:
            continue
        x, v, e = X[aid], V[aid], sw[aid][0]
        E = x[e] - x[:e]
        en = np.hypot(*E.T)
        ue = E / np.maximum(en, 1e-9)[:, None]
        vp = smooth(np.sum(v[:e] * ue, axis=1))
        a = np.gradient(vp, dt)
        pts += [(en[k], vp[k] + a[k] / C0) for k in range(12, e - 4)]
    pts = np.array(pts)
    edges = np.r_[0, np.geomspace(0.01, max(3.2, pts[:, 0].max()), 40)]
    ee, vv = [0.0], [0.0]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (pts[:, 0] >= lo) & (pts[:, 0] < hi)
        if m.sum() >= 3:
            ee.append(np.median(pts[m, 0]))
            vv.append(np.median(pts[m, 1]))
    ee, vv = np.array(ee), np.maximum.accumulate(vv)
    return lambda d: np.interp(d, ee, vv, right=vv[-1])


def leg_mass(x, v, s, e, cmd, C0, dt):
    """Mass (empty robot = 1) that best reproduces the leg s..e."""
    T = x[e]
    ref = v[s + 1:e + 1]

    def err(logm):
        C = C0 / np.exp(logm)
        p, w, out = x[s].copy(), v[s].copy(), np.empty_like(ref)
        for k in range(len(ref)):
            E = T - p
            d = np.hypot(*E)
            w = w + dt * C * (cmd(d) * E / max(d, 1e-9) - w)
            p = p + dt * w
            out[k] = w
        return np.mean((out - ref) ** 2)

    lo, hi = np.log(0.3), np.log(60.0)
    grid = np.linspace(lo, hi, 60)
    i = int(np.argmin([err(g) for g in grid]))
    lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    g = (np.sqrt(5) - 1) / 2
    for _ in range(30):
        a, b = hi - g * (hi - lo), lo + g * (hi - lo)
        if err(a) < err(b):
            hi = b
        else:
            lo = a
    return float(np.exp((lo + hi) / 2))


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
    return [g for g in groups.values() if len(g) >= MIN_GROUP]


def analyze_zone(zdir):
    zone, env, X, V, dt = load_zone(zdir)
    sw = {aid: switches(X, V, aid) for aid in X}
    C0 = float(np.mean([first_leg_gain(V[aid], dt) for aid in X]))
    cmd = speed_command(X, V, dt, sw, C0)

    events = []  # (agent, step, position, dm)
    for aid in sorted(X):
        x, v = X[aid], V[aid]
        bnd = [0] + sw[aid] + [len(v) - 1]
        masses = [leg_mass(x, v, s, e, cmd, C0, dt) for s, e in zip(bnd[:-1], bnd[1:])]
        print(f"{zone} agent {aid} leg masses {np.round(masses, 2).tolist()}")
        for n, k in enumerate(sw[aid]):
            events.append((aid, k, x[k], masses[n + 1] - masses[n]))

    unit = np.median([dm for *_, dm in events if dm > 0]) if any(e[3] > 0 for e in events) else 1.0
    picks = [ev for ev in events if ev[3] > MIN_DM * unit]
    for aid, k, p, dm in events:
        kind = "pick" if dm > MIN_DM * unit else "drop" if dm < -MIN_DM * unit else "none"
        dense = " dense" if kind == "pick" and dm > DENSE_FACTOR * unit else ""
        print(f"{zone} agent {aid} step {k:5d} {kind}{dense} dm={dm:+.2f} at ({p[0]:.3f},{p[1]:.3f})")

    counts = {aid: 0 for aid in X}
    for aid, *_ in picks:
        counts[aid] += 1
    best = max(counts.values())
    result = {"more_samples": sorted(a for a, c in counts.items() if c == best),
              "promising_region": [], "picked_promising_samples": []}

    pts = np.array([p for _, _, p, _ in picks]).reshape(-1, 2)
    dense_pts = np.array([p for _, _, p, dm in picks if dm > DENSE_FACTOR * unit]).reshape(-1, 2)
    dense_regions = [dense_pts[g] for g in clusters(dense_pts)]
    candidates = []
    for g in clusters(pts):
        gap = min((np.min(np.hypot(*(pts[g][:, None, :] - d[None, :, :]).T)) for d in dense_regions),
                  default=np.inf)
        print(f"{zone} region of {len(g)} picks centred at {pts[g].mean(axis=0).round(3)}, "
              f"{gap:.2f} m from dense minerals")
        if gap >= DENSE_CLEARANCE:
            candidates.append(g)
    if candidates:
        g = max(candidates, key=len)
        center = pts[g].mean(axis=0)
        per = {}
        for i in g:
            per[picks[i][0]] = per.get(picks[i][0], 0) + 1
        top = max(per.values())
        result["promising_region"] = [round(float(center[0]), 2), round(float(center[1]), 2)]
        result["picked_promising_samples"] = sorted(a for a, c in per.items() if c == top)
    print(f"{zone} picks per agent {counts}")

    out = os.path.join(zdir, f"{zone}_explored.json")
    with open(out, "w") as fh:
        json.dump(result, fh, indent=1)
    print(out, result)


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "zones")
    for z in sorted(glob.glob(os.path.join(root, "zone*"))):
        if os.path.isdir(z):
            analyze_zone(z)
