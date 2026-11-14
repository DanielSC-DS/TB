#!/usr/bin/env python3
"""Assess robot cave logs and write caveX_outcome.json files.

Usage: python3 assess_caves.py [caves_dir]
"""
import glob, heapq, json, math, os, sys
import numpy as np
import pandas as pd

DT = 0.05
CLEARANCE = 0.35
SENSOR = 0.5
RADIUS = 0.075
# predicted miss distance (constant-velocity extrapolation) below which two robots are on a collision course
COLLISION_PRED = 0.1
# second-order approach dynamics identified from the logs:  a = -C1*v + C2*(T - p)
C1, C2 = 4.444, 2.056


# ---------------------------------------------------------------- loading
def load(cdir):
    name = os.path.basename(os.path.normpath(cdir))
    env = json.load(open(os.path.join(cdir, f'{name}_env.json')))
    ags = [pd.read_csv(os.path.join(cdir, f'{name}_agent_{i}.csv')) for i in range(env['num_agents'])]
    return name, env, ags


# ---------------------------------------------------------------- gripper
def grip_events(d):
    """Classify every gripper closing cycle of one robot.
    pick : closed on an object with the arm extended and released only once retracted
    put  : closed with arm retracted (object on robot) and released with arm extended
    fail / abort : closed on nothing, or object dropped / released mid-way."""
    o, ph, cu = d.open.values, d.phi.values, d.curr.values
    n = len(o); ev = []; i = 0
    while i < n:
        if o[i] <= 5e-4:
            i += 1; continue
        a = i
        while i < n and o[i] > 5e-4: i += 1
        b = i - 1
        hi = np.where(cu[a:b + 1] > 2.5)[0]
        maxo = o[a:b + 1].max()
        info = dict(start=a, end=b, kind='fail')
        if len(hi) and maxo < 0.6:
            closed = a + int(np.argmax(o[a:b + 1] >= maxo - 1e-3))   # gripper reaches the object
            rel = next((k for k in range(closed, b + 1) if o[k] < maxo - 5e-3), b)
            info.update(close=closed, rel=rel)
            if ph[a] > 3.1 and ph[rel] < 0.1:
                info['kind'] = 'pick'
            elif ph[a] < 0.1 and ph[rel] > 3.1:
                info['kind'] = 'put'
            else:
                info['kind'] = 'abort'
        ev.append(info)
    return ev


def carrying(ags):
    evs = []
    for i, d in enumerate(ags):
        for v in grip_events(d):
            if v['kind'] in ('pick', 'put'):
                evs.append((v['close'], i, v['kind']))
    evs.sort()
    groups = []
    for e in evs:
        if groups and groups[-1][0][2] == e[2] and abs(groups[-1][0][0] - e[0]) <= 3:
            groups[-1].append(e)
        else:
            groups.append([e])
    holding, handled, found = set(), [], None
    for g in groups:
        ids = sorted(x[1] for x in g)
        if g[0][2] == 'pick':
            if found is None:
                pts = np.array([[ags[i].x[k], ags[i].y[k]] for k, i, _ in g])
                m = pts.mean(0)
                found = [ids[0], round(float(m[0]), 4), round(float(m[1]), 4)]
            handled += ids
            holding |= set(ids)
        else:
            holding -= set(ids)
    return sorted(holding), handled, found


# ---------------------------------------------------------------- geometry / path
def seg_dist(P, A, B):
    A = np.asarray(A, float); B = np.asarray(B, float); ab = B - A; L = ab @ ab
    t = np.clip(((P - A) @ ab) / L, 0, 1) if L > 0 else np.zeros(len(P))
    return np.hypot(*(P - (A + t[:, None] * ab)).T)


def wall_dist(env, P):
    P = np.atleast_2d(np.asarray(P, float))
    return np.min([seg_dist(P, a, b) for a, b in env['scenario']], axis=0)


def shortest_path(env, start, goal, clr=CLEARANCE, h=0.01):
    W = np.array(env['scenario']).reshape(-1, 2)
    x0, y0 = W.min(0); x1, y1 = W.max(0)
    xs = np.arange(x0, x1 + h / 2, h); ys = np.arange(y0, y1 + h / 2, h)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    free = (wall_dist(env, np.c_[X.ravel(), Y.ravel()]).reshape(X.shape) >= clr)
    nx, ny = X.shape
    idx = lambda p: (int(round((p[0] - x0) / h)), int(round((p[1] - y0) / h)))
    s, g = idx(start), idx(goal)
    for p in (s, g):                     # endpoints closer than clr to a wall: connect to nearest free cell
        if not free[p]:
            fi = np.argwhere(free); q = fi[np.argmin(((fi - np.array(p)) ** 2).sum(1))]
            m = int(max(abs(q[0] - p[0]), abs(q[1] - p[1]))) + 1
            for t in np.linspace(0, 1, m + 1):
                free[int(round(p[0] + t * (q[0] - p[0]))), int(round(p[1] + t * (q[1] - p[1])))] = True
    nb = [(a, b, math.hypot(a, b)) for a, b in
          [(1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1),
           (2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2), (1, -2), (2, -1)]]
    dist = np.full(X.shape, np.inf); prev = {}; dist[s] = 0; pq = [(0.0, s)]
    while pq:
        dd, u = heapq.heappop(pq)
        if u == g: break
        if dd > dist[u]: continue
        for a, b, w in nb:
            v = (u[0] + a, u[1] + b)
            if 0 <= v[0] < nx and 0 <= v[1] < ny and free[v] and dd + w < dist[v]:
                dist[v] = dd + w; prev[v] = u; heapq.heappush(pq, (dd + w, v))
    if not np.isfinite(dist[g]):
        return None
    cells = [g]
    while cells[-1] != s: cells.append(prev[cells[-1]])
    pts = np.array([[xs[i], ys[j]] for i, j in cells[::-1]])
    pts[0], pts[-1] = start, goal
    ds, dg = wall_dist(env, [start])[0], wall_dist(env, [goal])[0]

    def ok(a, b):   # straight segment keeps the clearance (relaxed only near endpoints lacking it)
        n = max(2, int(np.hypot(*(b - a)) / 0.005) + 1)
        P = a + np.linspace(0, 1, n)[:, None] * (b - a)
        lim = np.minimum(clr - 1e-3, np.minimum(ds + np.hypot(*(P - start).T), dg + np.hypot(*(P - goal).T)))
        return np.all(wall_dist(env, P) >= lim)
    out = [pts[0]]; i = 0
    while i < len(pts) - 1:
        j = len(pts) - 1
        while j > i + 1 and not ok(pts[i], pts[j]): j -= 1
        out.append(pts[j]); i = j
    return np.array(out)


def densify(path, step=0.005):
    return np.vstack([a + np.linspace(0, 1, max(2, int(np.hypot(*(b - a)) / step) + 1))[:, None] * (b - a)
                      for a, b in zip(path[:-1], path[1:])])


# ---------------------------------------------------------------- stopping prediction
def implied_targets(P):
    V = np.diff(P, axis=0) / DT; A = np.diff(V, axis=0) / DT
    return P[1:-1] + (A + C1 * (V[:-1] + V[1:]) / 2) / C2


def expected_stop(d, still=0.02):
    """Return (state, position) where the robot is expected to come to rest."""
    P = np.c_[d.x.values, d.y.values]
    V = np.diff(P, axis=0) / DT
    if np.hypot(*V[-1]) < still:
        return 'STATIC', P[-1]
    T = implied_targets(P)
    # stable target estimate at the end of the log
    if np.ptp(T[-6:], axis=0).max() < 0.05:
        return 'MOVING', T[-1]
    # target was stable until a short disturbance in the very last steps
    for k in range(len(T) - 6, len(T) - 12, -1):
        if np.ptp(T[k - 6:k], axis=0).max() < 0.05:
            return 'MOVING', T[k - 1]
    # noisy motion: remaining distance from the averaged tangential deceleration
    v = V[-1]; sp = np.hypot(*v)
    u = (P[-1] - P[-6]) / np.hypot(*(P[-1] - P[-6]))
    A = np.diff(V, axis=0) / DT
    at = np.mean(A[-6:] @ u)
    rem = max((at + C1 * sp) / C2, sp * sp / (2 * max(-at, 1e-6)) if at < 0 else 0)
    rem = (at + C1 * sp) / C2 if (at + C1 * sp) > 0 else rem
    return 'MOVING', P[-1] + rem * u


# ---------------------------------------------------------------- conflicts
def conflicts(ags, carriers_pairs=()):
    out = []
    n = len(ags)
    for i in range(n):
        for j in range(i + 1, n):
            L = min(len(ags[i]), len(ags[j]))
            Pi = np.c_[ags[i].x, ags[i].y][:L]; Pj = np.c_[ags[j].x, ags[j].y][:L]
            Vi = np.r_[[[0, 0]], np.diff(Pi, axis=0) / DT]; Vj = np.r_[[[0, 0]], np.diff(Pj, axis=0) / DT]
            dp = Pj - Pi; dv = Vj - Vi; D = np.hypot(*dp.T)
            t = -np.sum(dp * dv, 1) / (np.sum(dv * dv, 1) + 1e-12)
            pm = np.hypot(*(dp + np.clip(t, 0, None)[:, None] * dv).T)
            si, sj = np.hypot(*Vi.T), np.hypot(*Vj.T)
            flag = (D < SENSOR) & (t > 0) & (t < 2.5) & (pm < COLLISION_PRED) & (si > 0.1) & (sj > 0.1)
            k = 0
            while k < L:
                if not flag[k]:
                    k += 1; continue
                a = k
                while a > 0 and D[a - 1] < SENSOR: a -= 1
                b = k
                while b + 1 < L and D[b + 1] < SENSOR: b += 1
                seg = np.arange(a, b + 1); kmin = int(seg[np.argmin(D[seg])])
                cosang = Vi[k] @ Vj[k] / (si[k] * sj[k])
                ang = math.degrees(math.acos(max(-1, min(1, cosang))))
                kind = 'frontal' if ang >= 150 else 'squeeze'
                out.append([kind, int(ags[i].step[kmin]), i, j])
                k = b + 1
    out.sort(key=lambda c: c[1])
    return out


# ---------------------------------------------------------------- main
def assess(cdir):
    name, env, ags = load(cdir)
    hold, handled, found = carrying(ags)
    res = dict(name=name, outcome='NO_RESCUE', collected=False, collected_by=None, position=None,
               obstructed_by=None, obstruction_case=None, cooperation_flag=False,
               conflicts=conflicts(ags), found_in=found, handled_object=handled)
    if hold:
        lo = hold[0]
        start = np.array([ags[lo].x.iloc[0], ags[lo].y.iloc[0]])
        goal = np.mean([[ags[i].x.iloc[-1], ags[i].y.iloc[-1]] for i in hold], axis=0)
        path = densify(shortest_path(env, start, goal))
        obs, case = [], []
        for i, d in enumerate(ags):
            if i in hold: continue
            state, p = expected_stop(d)
            if np.min(np.hypot(*(path - p).T)) < CLEARANCE + env['agent_radius']:
                obs.append(i); case.append(state)
        res.update(collected=True, collected_by=hold,
                   position=[round(float(goal[0]), 4), round(float(goal[1]), 4)],
                   cooperation_flag=len(hold) == 2,
                   outcome='RESCUE_HARD' if obs else 'RESCUE',
                   obstructed_by=obs or None, obstruction_case=case or None)
    order = ['name', 'outcome', 'collected', 'collected_by', 'position', 'obstructed_by', 'obstruction_case',
             'cooperation_flag', 'conflicts', 'found_in', 'handled_object']
    return {k: res[k] for k in order}


if __name__ == '__main__':
    base = sys.argv[1] if len(sys.argv) > 1 else '/app/caves'
    for cdir in sorted(glob.glob(os.path.join(base, 'cave*/')), key=lambda s: int(s.rstrip('/').split('cave')[-1])):
        r = assess(cdir)
        with open(os.path.join(base, f"{r['name']}_outcome.json"), 'w') as f:
            json.dump(r, f, indent=4)
        print(json.dumps(r))
