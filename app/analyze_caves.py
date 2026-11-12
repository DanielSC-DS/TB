#!/usr/bin/env python3
"""Assess robot logs in each cave and write caveX_outcome.json files.

Usage: python3 analyze_caves.py [caves_dir]   (default: directory 'caves' next to this file)
"""
import heapq
import itertools
import json
import math
import os
import sys

import numpy as np
import pandas as pd

CLEARANCE = 0.35       # rescue clearance measured from the path centre
SENSOR_RANGE = 0.5     # detection range of the robots
CONFLICT_HORIZON = 3.0 # s, "imminent" collision look-ahead
GRID = 0.02            # path-planning grid resolution
PI = math.pi


# ----------------------------------------------------------------------------- geometry
def seg_point_dist(p, a, b):
    """Distance from points p (N,2) to segment a-b."""
    ab = b - a
    t = np.clip(((p - a) @ ab) / max(ab @ ab, 1e-12), 0, 1)
    return np.hypot(*(p - (a + t[:, None] * ab)).T)


def seg_seg_dist(p1, p2, q1, q2):
    def inter(a, b, c, d):
        def cr(o, x, y):
            return (x[0] - o[0]) * (y[1] - o[1]) - (x[1] - o[1]) * (y[0] - o[0])
        d1, d2, d3, d4 = cr(c, d, a), cr(c, d, b), cr(a, b, c), cr(a, b, d)
        return d1 * d2 < 0 and d3 * d4 < 0
    if inter(p1, p2, q1, q2):
        return 0.0
    pts = np.array([p1, p2])
    qs = np.array([q1, q2])
    return float(min(seg_point_dist(pts, q1, q2).min(), seg_point_dist(qs, p1, p2).min()))


class Walls:
    def __init__(self, lines):
        self.segs = [(np.array(a, float), np.array(b, float)) for a, b in lines]

    def dist(self, pts):
        pts = np.atleast_2d(pts)
        return np.min([seg_point_dist(pts, a, b) for a, b in self.segs], axis=0)

    def seg_clear(self, p, q, c):
        return all(seg_seg_dist(p, q, a, b) >= c for a, b in self.segs)


def polyline_point_dist(path, p):
    path = np.asarray(path)
    p = np.asarray(p, float)[None, :]
    return float(min(seg_point_dist(p, path[i], path[i + 1])[0] for i in range(len(path) - 1)))


def shortest_path(walls, start, goal, c=CLEARANCE):
    """Any-angle (Theta*) shortest path keeping clearance c from all walls."""
    pts = np.vstack([np.vstack(s) for s in walls.segs])
    lo, hi = pts.min(0), pts.max(0)
    nx, ny = int(round((hi[0] - lo[0]) / GRID)) + 1, int(round((hi[1] - lo[1]) / GRID)) + 1
    xs, ys = lo[0] + np.arange(nx) * GRID, lo[1] + np.arange(ny) * GRID
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    free = (walls.dist(np.c_[X.ravel(), Y.ravel()]) >= c).reshape(nx, ny)

    def cell(p):
        return int(round((p[0] - lo[0]) / GRID)), int(round((p[1] - lo[1]) / GRID))

    def pos(n):
        return np.array([xs[n[0]], ys[n[1]]])

    def nearest_free(p):
        i, j = cell(p)
        if 0 <= i < nx and 0 <= j < ny and free[i, j]:
            return (i, j)
        best, bd = None, 1e9
        for di in range(-25, 26):
            for dj in range(-25, 26):
                a, b = i + di, j + dj
                if 0 <= a < nx and 0 <= b < ny and free[a, b]:
                    d = np.hypot(xs[a] - p[0], ys[b] - p[1])
                    if d < bd:
                        best, bd = (a, b), d
        return best

    s, g = nearest_free(start), nearest_free(goal)
    if s is None or g is None:
        return None, free
    los_cache = {}

    def los(a, b):
        k = (a, b) if a < b else (b, a)
        if k not in los_cache:
            los_cache[k] = walls.seg_clear(pos(a), pos(b), c - 1e-9)
        return los_cache[k]

    gpos = pos(g)
    gs = {s: 0.0}
    par = {s: s}
    openq = [(np.linalg.norm(pos(s) - gpos), s)]
    closed = set()
    nbrs = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    while openq:
        _, u = heapq.heappop(openq)
        if u in closed:
            continue
        if u == g:
            break
        closed.add(u)
        for di, dj in nbrs:
            v = (u[0] + di, u[1] + dj)
            if not (0 <= v[0] < nx and 0 <= v[1] < ny) or not free[v] or v in closed:
                continue
            pu = par[u]
            if los(pu, v):
                cand, cp = gs[pu] + np.linalg.norm(pos(pu) - pos(v)), pu
            else:
                cand, cp = gs[u] + np.linalg.norm(pos(u) - pos(v)), u
            if cand < gs.get(v, 1e18):
                gs[v] = cand
                par[v] = cp
                heapq.heappush(openq, (cand + np.linalg.norm(pos(v) - gpos), v))
    if g not in par:
        return None, free
    path = [g]
    while path[-1] != s:
        path.append(par[path[-1]])
    path = [pos(n) for n in reversed(path)]
    path = [np.asarray(start, float)] + path + [np.asarray(goal, float)]
    return np.array(path), free


# ----------------------------------------------------------------------------- gripper
def segments(mask):
    out, s = [], None
    for i, v in enumerate(mask):
        if v and s is None:
            s = i
        if not v and s is not None:
            out.append((s, i - 1))
            s = None
    if s is not None:
        out.append((s, len(mask) - 1))
    return out


def gripper_events(d):
    """Return (events, holding_at_end). events: list of (step, kind)."""
    phi, opn, cur = d.phi.values, d.open.values, d.curr.values
    holding = False
    events = []
    for a, b in segments(cur > 2.0):  # gripper firmly closed on something
        o = a
        while o > 0 and cur[o - 1] > 0.05:   # onset of the closing manoeuvre
            o -= 1
        closed_open = opn[a:b + 1].max()
        if closed_open > 0.95:          # fully closed -> grasped nothing
            events.append((a, 'miss'))
            continue
        if phi[o] > PI - 0.1:           # grasp with the arm extended
            if holding:
                continue
            # success only if the grip is held until the arm is retracted
            if phi[b] < 0.1:
                holding = True
                events.append((a, 'pickup'))
            else:
                events.append((a, 'drop'))
        elif phi[o] < 0.1 and holding:  # grasp the object resting on the robot
            if phi[b] > PI - 0.1:
                holding = False
                events.append((a, 'putdown'))
    return events, holding


# ----------------------------------------------------------------------------- motion
def ray_to_wall(walls, p, u, max_d):
    """Free travel distance from p along unit vector u before hitting a wall."""
    best = max_d
    for a, b in walls.segs:
        e = b - a
        den = u[0] * (-e[1]) + u[1] * e[0]
        if abs(den) < 1e-12:
            continue
        w = a - p
        t = (w[0] * (-e[1]) + w[1] * e[0]) / den
        s = (u[0] * w[1] - u[1] * w[0]) / den
        if t > 0 and 0 <= s <= 1:
            best = min(best, t)
    return best


def stop_prediction(d, walls=None, radius=0.0):
    """Predict where a robot stops. Returns (pred_pos, moving, braking).

    Speeds are derived from the logged positions (the velocity readings are noisy).
    The robots brake by exponential speed decay (v[n+1] = r * v[n]); the decay
    ratio r differs between robots and may still be settling after a manoeuvre.
    The ratio sequence is extrapolated (geometric convergence) and the speed is
    integrated forward until the robot stops.
    """
    x, y, t = d.x.values, d.y.values, d.time_s.values
    last = np.array([x[-1], y[-1]])
    dt = float(np.median(np.diff(t)))
    sp = np.hypot(np.diff(x), np.diff(y)) / np.diff(t)
    v_end = float(sp[-1])
    if float(np.mean(sp[-3:])) < 0.02:
        return last, False, False
    rat = sp[1:] / np.maximum(sp[:-1], 1e-9)
    recent = sp[-20:]
    braking = (v_end < 0.9 * recent.max() or bool(np.all(rat[-3:] < 0.99))) and rat[-1] < 1.0
    if not braking:
        return last, True, False
    k = max(len(x) - 4, 0)
    u = np.array([x[-1] - x[k], y[-1] - y[k]])
    u = u / (np.linalg.norm(u) + 1e-12)
    # extrapolate the decay ratio
    r = min(float(rat[-1]), 0.99)
    d1, d2 = rat[-1] - rat[-2], rat[-2] - rat[-3]
    if abs(d1) > 1e-5 and abs(d2) > 1e-5 and d1 * d2 > 0 and abs(d1) < abs(d2):
        q = d1 / d2
    else:
        q, d1 = 0.0, 0.0
    v, dist, delta = v_end, 0.0, d1
    for _ in range(100000):
        delta *= q
        r = min(r + delta, 0.99)
        v *= r
        dist += v * dt
        if v < 1e-4:
            break
    if walls is not None:
        dist = min(dist, max(ray_to_wall(walls, last, u, dist + 1) - radius, 0.0))
    return last + u * dist, True, True


# ----------------------------------------------------------------------------- conflicts
def conflicts(ds, r):
    out = []
    for i, j in itertools.combinations(range(len(ds)), 2):
        L = min(len(ds[i]), len(ds[j]))
        a, b = ds[i].iloc[:L], ds[j].iloc[:L]
        P = np.c_[b.x - a.x, b.y - a.y]
        V = np.c_[b.vx - a.vx, b.vy - a.vy]
        dist = np.hypot(*P.T)
        va, vb = np.c_[a.vx, a.vy], np.c_[b.vx, b.vy]
        sa, sb = np.hypot(*va.T), np.hypot(*vb.T)
        vv = (V * V).sum(1)
        tc = np.where(vv > 1e-8, -(P * V).sum(1) / np.maximum(vv, 1e-12), 0)
        dmin = np.hypot(*(P + V * np.clip(tc, 0, None)[:, None]).T)
        risk = (dist < SENSOR_RANGE) & (tc > 0) & (tc < CONFLICT_HORIZON) & (dmin < 2 * r) \
            & (sa > 0.05) & (sb > 0.05)
        for s, e in segments(dist < SENSOR_RANGE):
            idx = [k for k in range(s, e + 1) if risk[k]]
            if not idx:
                continue
            k = idx[0]
            cosang = (va[k] @ vb[k]) / (sa[k] * sb[k])
            ang = math.degrees(math.acos(max(-1, min(1, cosang))))
            if ang < 30:      # same direction (following / side by side) - not of interest
                continue
            kind = 'frontal' if ang >= 150 else 'squeeze'
            m = s + int(np.argmin(dist[s:e + 1]))
            out.append([kind, int(m), i, j])
    out.sort(key=lambda c: (c[1], c[2], c[3]))
    return out


# ----------------------------------------------------------------------------- main
def analyze(cave_dir, name):
    env = json.load(open(os.path.join(cave_dir, f'{name}_env.json')))
    n, r = env['num_agents'], env['agent_radius']
    walls = Walls(env['scenario'])
    ds = [pd.read_csv(os.path.join(cave_dir, f'{name}_agent_{i}.csv')) for i in range(n)]

    holders, pick_step = [], {}
    for i, d in enumerate(ds):
        ev, hold = gripper_events(d)
        if hold:
            holders.append(i)
            pick_step[i] = [s for s, k in ev if k == 'pickup'][-1]

    res = dict(name=name, outcome='NO_RESCUE', collected=False, collected_by=None, position=None,
               obstructed_by=None, obstruction_case=None, cooperation_flag=False,
               conflicts=conflicts(ds, r), debug={})
    if not holders:
        return res
    # cooperation: two holders that picked up at the same time
    if len(holders) >= 2:
        best = min(itertools.combinations(holders, 2),
                   key=lambda p: abs(pick_step[p[0]] - pick_step[p[1]]))
        carriers = sorted(best)
    else:
        carriers = holders
    coop = len(carriers) == 2
    ends = [np.array([ds[i].x.iloc[-1], ds[i].y.iloc[-1]]) for i in carriers]
    goal = np.mean(ends, axis=0)
    c0 = min(carriers)
    start = np.array([ds[c0].x.iloc[0], ds[c0].y.iloc[0]])
    path, _ = shortest_path(walls, start, goal)

    obs, cases, dbg = [], [], {}
    for i in range(n):
        if i in carriers:
            continue
        pred, moving, decel = stop_prediction(ds[i], walls, r)
        last = np.array([ds[i].x.iloc[-1], ds[i].y.iloc[-1]])
        dl = polyline_point_dist(path, last)
        dp = polyline_point_dist(path, pred)
        dbg[i] = dict(last=last.round(3).tolist(), pred=pred.round(3).tolist(), moving=moving,
                      decel=decel, d_last=round(dl, 3), d_pred=round(dp, 3))
        judged = dp if decel else dl
        if judged < CLEARANCE + r:
            obs.append(i)
            cases.append('MOVING' if moving else 'STATIC')
    res.update(collected=True, collected_by=carriers,
               position=[round(float(goal[0]), 4), round(float(goal[1]), 4)],
               cooperation_flag=coop)
    if obs:
        res.update(outcome='RESCUE_HARD', obstructed_by=obs, obstruction_case=cases)
    else:
        res['outcome'] = 'RESCUE'
    res['debug'] = dict(path=np.round(path, 3).tolist(), robots=dbg,
                        path_len=round(float(np.sum(np.hypot(*np.diff(path, axis=0).T))), 3),
                        start_wall=round(float(walls.dist(start)[0]), 3),
                        goal_wall=round(float(walls.dist(goal)[0]), 3))
    return res


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('-')]
    base = args[0] if args else os.path.join(os.path.dirname(os.path.abspath(__file__)), 'caves')
    verbose = '-v' in sys.argv
    caves = sorted((c for c in os.listdir(base) if c.startswith('cave') and os.path.isdir(os.path.join(base, c))),
                   key=lambda s: int(s[4:]))
    for name in caves:
        res = analyze(os.path.join(base, name), name)
        dbg = res.pop('debug')
        if verbose:
            print(json.dumps(dbg))
        with open(os.path.join(base, f'{name}_outcome.json'), 'w') as f:
            json.dump(res, f, indent=4)
        print(json.dumps(res))


if __name__ == '__main__':
    main()
