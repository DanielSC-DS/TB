#!/usr/bin/env python3
"""Assess robot logs in each caveX folder and write caveX_outcome.json.

Usage: python3 assess_caves.py [caves_dir]   (default: ./caves next to this script)

Method summary
--------------
* Carrying: a pick-up is valid when the gripper closes on something with the arm
  fully extended (phi ~ pi), stops at a partial opening (<0.95, i.e. it did not
  close on empty space), holds a lifting current (~3 A), the arm fully retracts
  (phi < 0.1) and only then the gripper reopens (object resting on the robot).
  A put-down is the reverse (close at phi ~ 0, extend to pi, reopen).  Drops
  mid-retraction, partial arm extension, weak current or a fully closed gripper
  are not valid pick-ups.  Two robots completing pick-ups at the same steps
  while close to each other are a cooperating pair.
* Rescue path: shortest path from the carrier's (lowest id's) first logged
  position to its (pair midpoint) last position keeping 0.35 m from walls
  (grid Dijkstra + line-of-sight smoothing).
* Obstruction: a robot (other than the carriers) whose judged position is
  within the 0.35 m clearance of the path.  Robots slowing down at the end of
  their log are judged at their expected stopping point (go-to-goal exponential
  braking, v = k * remaining distance, k estimated per robot from its speed
  tail).  STATIC if speed ~0 at the end of the log, else MOVING.
* Conflicts: within the 0.5 m sensor range, both robots moving, and linear
  extrapolation predicts a collision (closest approach < 2 radii) within 1 s.
  frontal if headings differ by > 135 deg, otherwise squeeze.  The reported step
  is the step of minimum separation during that encounter.
"""
import heapq
import json
import math
import os
import sys

import numpy as np
import pandas as pd

DT = 0.05
CLEARANCE = 0.35
SENSOR = 0.5
TTC_MAX = 1.0
V_MOVE = 0.02
K_GOAL_DEFAULT = 0.5


# ---------------------------------------------------------------- geometry
def segdist(P, W):
    P = np.asarray(P, float)
    sh = P.shape[:-1]
    P = P.reshape(-1, 2)
    best = np.full(len(P), np.inf)
    for x1, y1, x2, y2 in W:
        dx, dy = x2 - x1, y2 - y1
        L = dx * dx + dy * dy
        if L == 0:
            t = np.zeros(len(P))
        else:
            t = np.clip(((P[:, 0] - x1) * dx + (P[:, 1] - y1) * dy) / L, 0, 1)
        best = np.minimum(best, np.hypot(P[:, 0] - x1 - t * dx, P[:, 1] - y1 - t * dy))
    return best.reshape(sh)


def poly_dist(p, Q):
    if len(Q) == 1:
        return float(np.hypot(*(np.asarray(p) - Q[0])))
    return float(segdist(p, np.c_[Q[:-1], Q[1:]]))


def plan_path(W, s, e, res=0.01, cl=CLEARANCE):
    xs = np.r_[W[:, 0], W[:, 2]]
    ys = np.r_[W[:, 1], W[:, 3]]
    x0, y0 = xs.min(), ys.min()
    nx = int((xs.max() - x0) / res) + 2
    ny = int((ys.max() - y0) / res) + 2
    gx = x0 + np.arange(nx) * res
    gy = y0 + np.arange(ny) * res
    G = np.stack(np.meshgrid(gx, gy, indexing="ij"), -1)
    free = segdist(G, W) >= cl
    idx = lambda p: (int(round((p[0] - x0) / res)), int(round((p[1] - y0) / res)))
    si, ei = idx(s), idx(e)
    free[si] = free[ei] = True
    dist = np.full((nx, ny), np.inf)
    prev = {}
    dist[si] = 0.0
    h = [(0.0, si)]
    nb = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1),
          (1, 2), (2, 1), (-1, 2), (-2, 1), (1, -2), (2, -1), (-1, -2), (-2, -1)]
    nb = [(a, b, math.hypot(a, b)) for a, b in nb]
    while h:
        dd, u = heapq.heappop(h)
        if u == ei:
            break
        if dd > dist[u]:
            continue
        for a, b, w in nb:
            v = (u[0] + a, u[1] + b)
            if not (0 <= v[0] < nx and 0 <= v[1] < ny and free[v]):
                continue
            if abs(a) + abs(b) == 3 and not free[u[0] + (a > 0) - (a < 0), u[1] + (b > 0) - (b < 0)]:
                continue
            nd = dd + w
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(h, (nd, v))
    if not np.isfinite(dist[ei]):
        return None
    path = [ei]
    while path[-1] != si:
        path.append(prev[path[-1]])
    P = np.array([[gx[i], gy[j]] for i, j in path[::-1]])
    P[0], P[-1] = s, e

    def visible(a, b):
        n = int(np.hypot(*(b - a)) / (res / 2)) + 2
        pts = a + (b - a) * np.linspace(0, 1, n)[:, None]
        return segdist(pts, W).min() >= cl - 1e-9

    out = [P[0]]
    i = 0
    while i < len(P) - 1:
        j = len(P) - 1
        while j > i + 1 and not visible(P[i], P[j]):
            j -= 1
        out.append(P[j])
        i = j
    return np.array(out)


# ---------------------------------------------------------------- gripper
def grip_segments(df):
    o = (df.open.values > 0.005)
    segs, i = [], 0
    while i < len(o):
        if o[i]:
            j = i
            while j < len(o) and o[j]:
                j += 1
            segs.append((i, j - 1))
            i = j
        else:
            i += 1
    return segs


def carry_events(df):
    """Return list of (kind, start_step, end_step) with kind 'pick' or 'drop'."""
    ev = []
    phi, op, cu = df.phi.values, df.open.values, df.curr.values
    for a, b in grip_segments(df):
        held = (cu[a:b + 1] > 2.5).any() and op[a:b + 1].max() < 0.95
        if not held:
            continue
        p0, p1 = phi[a], phi[b]
        if p0 > math.pi - 0.05 and p1 < 0.1:
            ev.append(("pick", a, b))
        elif p0 < 0.1 and p1 > math.pi - 0.05:
            ev.append(("drop", a, b))
    return ev


def is_carrying(df):
    state = False
    pick = None
    for kind, a, b in carry_events(df):
        if kind == "pick":
            state, pick = True, (a, b)
        else:
            state, pick = False, None
    return state, pick


# ---------------------------------------------------------------- motion
def judged_position(df):
    """Position used to judge obstruction, and whether the robot is moving."""
    x, y = df.x.values, df.y.values
    sp = np.hypot(np.diff(x), np.diff(y)) / DT
    last = np.array([x[-1], y[-1]])
    v = sp[-1]
    if v <= V_MOVE:
        return last, False
    decel = sp[-1] < sp[-2] < sp[-3]
    if not decel:
        return last, True
    tail = sp[-8:]
    r = math.exp(np.polyfit(np.arange(len(tail)), np.log(np.maximum(tail, 1e-9)), 1)[0])
    k = (1 - r) / DT
    if not (0.4 <= k <= 0.7):
        k = K_GOAL_DEFAULT
    u = np.array([x[-1] - x[-3], y[-1] - y[-3]])
    u /= np.hypot(*u)
    return last + u * (v / k), True


def conflicts(agents, radius):
    PV = []
    for df in agents:
        x, y = df.x.values, df.y.values
        PV.append((np.c_[x, y], np.c_[np.gradient(x, DT), np.gradient(y, DT)]))
    out = []
    n = len(agents)
    for i in range(n):
        for j in range(i + 1, n):
            L = min(len(PV[i][0]), len(PV[j][0]))
            P = PV[j][0][:L] - PV[i][0][:L]
            V = PV[j][1][:L] - PV[i][1][:L]
            dist = np.hypot(*P.T)
            vv = (V ** 2).sum(1)
            ts = -(P * V).sum(1) / np.maximum(vv, 1e-12)
            dmin = np.hypot(*(P + V * np.clip(ts, 0, None)[:, None]).T)
            si = np.hypot(*PV[i][1][:L].T)
            sj = np.hypot(*PV[j][1][:L].T)
            flag = (dist < SENSOR) & (ts > 0) & (ts < TTC_MAX) & (dmin < 2 * radius) & (si > 0.1) & (sj > 0.1)
            k = 0
            while k < L:
                if dist[k] < SENSOR:
                    m = k
                    while m < L and dist[m] < SENSOR:
                        m += 1
                    if flag[k:m].any():
                        f = k + int(np.argmax(flag[k:m]))
                        kk = k + int(np.argmin(dist[k:m]))
                        c = (PV[i][1][f] @ PV[j][1][f]) / (si[f] * sj[f])
                        ang = math.degrees(math.acos(max(-1.0, min(1.0, c))))
                        typ = "frontal" if ang > 135 else "squeeze"
                        out.append([typ, int(agents[i].step.values[kk]), i, j])
                    k = m
                else:
                    k += 1
    return sorted(out, key=lambda e: (e[1], e[2], e[3]))


# ---------------------------------------------------------------- main
def assess(cave_dir):
    name = os.path.basename(os.path.normpath(cave_dir))
    env = json.load(open(os.path.join(cave_dir, f"{name}_env.json")))
    W = np.array(env["scenario"], float).reshape(-1, 4)
    radius = env.get("agent_radius", 0.075)
    agents = [pd.read_csv(os.path.join(cave_dir, f"{name}_agent_{a}.csv")) for a in range(env["num_agents"])]

    carriers = {}
    for a, df in enumerate(agents):
        st, pick = is_carrying(df)
        if st:
            carriers[a] = pick
    carrier_ids = sorted(carriers)
    coop = False
    if len(carrier_ids) >= 2:
        # cooperating pair: picked up together, close to each other at the end
        best = None
        for i in carrier_ids:
            for j in carrier_ids:
                if i < j and abs(carriers[i][0] - carriers[j][0]) <= 3:
                    dd = np.hypot(agents[i].x.values[-1] - agents[j].x.values[-1],
                                  agents[i].y.values[-1] - agents[j].y.values[-1])
                    if best is None or dd < best[0]:
                        best = (dd, [i, j])
        if best:
            carrier_ids, coop = best[1], True
        else:  # keep the most recent pick-up
            carrier_ids = [max(carrier_ids, key=lambda a: carriers[a][1])]

    res = {"name": name}
    if not carrier_ids:
        res.update(outcome="NO_RESCUE", collected=False, collected_by=None, position=None,
                   obstructed_by=None, obstruction_case=None, cooperation_flag=False)
    else:
        start = agents[carrier_ids[0]][["x", "y"]].values[0].astype(float)
        pos = np.mean([agents[a][["x", "y"]].values[-1] for a in carrier_ids], axis=0).astype(float)
        path = plan_path(W, start, pos)
        obs, case = [], []
        for a, df in enumerate(agents):
            if a in carrier_ids:
                continue
            p, moving = judged_position(df)
            if path is not None and poly_dist(p, path) < CLEARANCE:
                obs.append(a)
                case.append("MOVING" if moving else "STATIC")
        res.update(outcome="RESCUE_HARD" if obs else "RESCUE", collected=True,
                   collected_by=carrier_ids, position=[round(float(pos[0]), 4), round(float(pos[1]), 4)],
                   obstructed_by=obs or None, obstruction_case=case or None, cooperation_flag=coop)
    res["conflicts"] = conflicts(agents, radius)
    return res


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "caves")
    for d in sorted(os.listdir(base), key=lambda s: (len(s), s)):
        full = os.path.join(base, d)
        if os.path.isdir(full) and os.path.exists(os.path.join(full, f"{d}_env.json")):
            r = assess(full)
            with open(os.path.join(base, f"{d}_outcome.json"), "w") as f:
                json.dump(r, f, indent=4)
            print(json.dumps(r))


if __name__ == "__main__":
    main()
