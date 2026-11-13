#!/usr/bin/env python3
"""Assess robot cave logs and write caveX_outcome.json files.

Usage: python3 assess.py [caves_dir]   (default: directory of this script)
"""
import glob
import heapq
import json
import math
import os
import sys

import numpy as np

CLEAR = 0.35          # rescue clearance (radius around path centre)
SENSOR = 0.5          # robot detection range
TCPA_MAX = 3.0        # "imminent" horizon [s]
GRID = 0.01           # planning grid resolution [m]
PI = math.pi


# ----------------------------------------------------------------- loading
def load_cave(folder):
    name = os.path.basename(folder.rstrip('/'))
    env = json.load(open(os.path.join(folder, f'{name}_env.json')))
    files = glob.glob(os.path.join(folder, f'{name}_agent_*.csv'))
    agents = {}
    for f in files:
        idx = int(f.rsplit('_', 1)[1].split('.')[0])
        agents[idx] = np.genfromtxt(f, delimiter=',', names=True)
    walls = np.array(env['scenario'], dtype=float)  # (N,2,2)
    return name, env, agents, walls


# --------------------------------------------------------- gripper analysis
def grip_episodes(a):
    """Contiguous intervals where the gripper is not fully open."""
    act = a['open'] > 0.005
    eps, i, n = [], 0, len(a)
    while i < n:
        if act[i]:
            j = i
            while j < n and act[j]:
                j += 1
            eps.append((i, j - 1))
            i = j
        else:
            i += 1
    return eps


def classify_episode(a, s, e):
    """Return ('pick'|'place'|'none', info).

    pick : gripper closes on an object with the arm extended (phi~pi), the
           arm retracts while holding and the gripper opens with the arm
           retracted (object rests on the robot).
    place: gripper closes on the object with the arm retracted, the arm
           extends while holding and the gripper opens at phi~pi.
    Anything else (missed grasp, gripper closing completely, low current,
    release half-way, arm not fully extended...) does not change the
    carrying state, except a release with the arm retracted while the robot
    was already carrying (handled by the caller as no change).
    """
    seg = a[s:e + 1]
    op, cur, phi = seg['open'], seg['curr'], seg['phi']
    peak = op.max()
    k_peak = int(np.argmax(op))
    # holding: gripper stopped before fully closed and the current is high
    hold_mask = (op > 0.05) & (op < 0.95) & (cur > 2.5)
    holding = hold_mask.any() and peak < 0.95
    phi_close = phi[0]
    # where did the gripper start to open again (release)?
    after = np.where(op[k_peak:] < peak - 0.02)[0]
    k_rel = k_peak + after[0] if len(after) else None
    still_closed_at_end = (e == len(a) - 1) and k_rel is None
    phi_rel = phi[k_rel] if k_rel is not None else phi[-1]
    info = dict(start=int(a['step'][s]), end=int(a['step'][e]),
                phi_close=float(phi_close), phi_release=float(phi_rel),
                peak=float(peak), holding=bool(holding),
                pos=(float(a['x'][s]), float(a['y'][s])))
    if not holding:
        return 'none', info
    if phi_close > PI - 0.05:            # arm fully extended when closing
        if still_closed_at_end:
            return 'pick', info          # log ended while holding
        if phi_rel < 0.1:
            return 'pick', info
        return 'none', info              # object released before retraction
    if phi_close < 0.1:                  # arm retracted when closing
        if still_closed_at_end:
            return 'carry_hold', info
        if phi_rel > PI - 0.05:
            return 'place', info
        return 'drop', info              # released mid-way: object fell off
    return 'none', info                  # arm not extended: invalid grasp


def carrying_state(a):
    carrying, last_pick, events = False, None, []
    for s, e in grip_episodes(a):
        kind, info = classify_episode(a, s, e)
        events.append((kind, info))
        if kind == 'pick':
            carrying, last_pick = True, info
        elif kind in ('place', 'drop'):
            carrying = False
    return carrying, last_pick, events


# -------------------------------------------------------------- geometry
def seg_dist(px, py, walls):
    """Distance from points (arrays) to the nearest wall segment."""
    d = np.full(px.shape, np.inf)
    for (x1, y1), (x2, y2) in walls:
        dx, dy = x2 - x1, y2 - y1
        L2 = dx * dx + dy * dy
        t = np.clip(((px - x1) * dx + (py - y1) * dy) / (L2 if L2 > 0 else 1), 0, 1)
        d = np.minimum(d, np.hypot(px - (x1 + t * dx), py - (y1 + t * dy)))
    return d


def point_polyline_dist(p, path):
    a, b = path[:-1], path[1:]
    ab = b - a
    L2 = (ab ** 2).sum(1)
    L2[L2 == 0] = 1e-12
    t = np.clip(((p - a) * ab).sum(1) / L2, 0, 1)
    proj = a + ab * t[:, None]
    return float(np.min(np.hypot(*(p - proj).T)))


def plan_path(walls, start, goal, clear=CLEAR):
    xs = walls[:, :, 0]
    ys = walls[:, :, 1]
    x0, x1 = min(xs.min(), start[0], goal[0]) - 0.1, max(xs.max(), start[0], goal[0]) + 0.1
    y0, y1 = min(ys.min(), start[1], goal[1]) - 0.1, max(ys.max(), start[1], goal[1]) + 0.1
    nx, ny = int((x1 - x0) / GRID) + 1, int((y1 - y0) / GRID) + 1
    gx = x0 + np.arange(nx) * GRID
    gy = y0 + np.arange(ny) * GRID
    X, Y = np.meshgrid(gx, gy, indexing='ij')
    dist = seg_dist(X, Y, walls)
    free = dist >= clear

    def cell(p):
        return int(round((p[0] - x0) / GRID)), int(round((p[1] - y0) / GRID))

    s, g = cell(start), cell(goal)
    # endpoints may sit closer than the clearance to a wall: allow them
    free_s = free.copy()
    for c in (s, g):
        free_s[c] = True
    # Dijkstra (8-connected) with A* heuristic
    INF = float('inf')
    best = np.full((nx, ny), INF)
    prev = -np.ones((nx, ny, 2), dtype=np.int32)
    best[s] = 0
    pq = [(0.0, 0.0, s)]
    nbrs = [(1, 0, 1), (-1, 0, 1), (0, 1, 1), (0, -1, 1),
            (1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)), (-1, -1, math.sqrt(2))]
    found = False
    while pq:
        f, c, u = heapq.heappop(pq)
        if c > best[u]:
            continue
        if u == g:
            found = True
            break
        for dx, dy, w in nbrs:
            v = (u[0] + dx, u[1] + dy)
            if not (0 <= v[0] < nx and 0 <= v[1] < ny) or not free_s[v]:
                # let the endpoints connect through a small blocked area
                if not (0 <= v[0] < nx and 0 <= v[1] < ny):
                    continue
                if not (_near(v, s) or _near(v, g)):
                    continue
            nc = c + w
            if nc < best[v]:
                best[v] = nc
                prev[v] = u
                h = math.hypot(v[0] - g[0], v[1] - g[1])
                heapq.heappush(pq, (nc + h, nc, v))
    if not found:
        return None, None
    cells = [g]
    while cells[-1] != s:
        cells.append(tuple(prev[cells[-1]]))
    cells.reverse()
    pts = np.array([[x0 + i * GRID, y0 + j * GRID] for i, j in cells])
    pts[0], pts[-1] = start, goal

    # string pulling: shortcut while the straight segment keeps the clearance
    def seg_ok(p, q):
        n = max(2, int(np.hypot(*(q - p)) / (GRID / 2)) + 1)
        t = np.linspace(0, 1, n)
        P = p[None] + (q - p)[None] * t[:, None]
        d = seg_dist(P[:, 0], P[:, 1], walls)
        # near the endpoints the clearance may be locally infeasible
        dS = np.hypot(*(P - start).T)
        dG = np.hypot(*(P - goal).T)
        ok = (d >= clear - 1e-3) | (dS < 0.05) | (dG < 0.05)
        return ok.all()

    out = [pts[0]]
    i = 0
    while i < len(pts) - 1:
        j = len(pts) - 1
        while j > i + 1 and not seg_ok(pts[i], pts[j]):
            j = (i + j) // 2 if j - i > 64 else j - 1
        out.append(pts[j])
        i = j
    path = np.array(out)
    return path, dist


def _near(v, c, r=6):
    return abs(v[0] - c[0]) <= r and abs(v[1] - c[1]) <= r


# --------------------------------------------------------- stop prediction
def braking_ratio(sp):
    """Characteristic per-step speed ratio of a robot while braking.

    The robots slow down exponentially (speed ratio ~constant per step);
    estimate it from all the steady braking samples of the log."""
    r = sp[1:] / np.maximum(sp[:-1], 1e-9)
    m = (sp[1:] > 0.03) & (sp[:-1] > sp[1:]) & (r > 0.9) & (r < 0.995)
    # keep only samples where the ratio is locally steady
    steady = np.zeros_like(m)
    steady[1:-1] = np.abs(r[2:] - r[:-2]) < 0.002
    m &= steady
    if m.sum() >= 5:
        return float(np.median(r[m]))
    return 0.978


def predict_stop(a, walls, radius):
    """Predict where a robot will stop given its speed trend at log end.

    Speeds are derived from logged positions (velocity readings are noisy).
    A robot whose speed keeps decreasing over the final steps is braking:
    its remaining travel is estimated with its exponential braking ratio q,
    d = v*dt*q/(1-q), along its current heading (and limited by walls).
    Returns (stop_position, moving_flag, info)."""
    x, y = a['x'], a['y']
    dt = float(np.median(np.diff(a['time_s'])))
    last = np.array([x[-1], y[-1]])
    sp = np.hypot(np.diff(x), np.diff(y)) / dt
    v_end = float(sp[-1])
    if v_end < 0.02:
        return last, False, dict(v_end=round(v_end, 4), braking=False, dist=0.0)
    h = np.array([x[-1] - x[-2], y[-1] - y[-2]])
    h = h / (np.linalg.norm(h) + 1e-12)
    tail = sp[-4:]
    braking = bool(np.all(np.diff(tail) < 0) and (tail[0] - tail[-1]) > 0.01 * tail[0])
    if not braking:
        return last, True, dict(v_end=round(v_end, 4), braking=False, dist=0.0)
    q = braking_ratio(sp)
    dist = v_end * dt * q / (1 - q)
    # a robot cannot drive through walls
    n = max(2, int(dist / 0.005) + 1)
    s = np.linspace(0, dist, n)
    P = last[None] + h[None] * s[:, None]
    wd = seg_dist(P[:, 0], P[:, 1], walls)
    hit = np.where(wd < radius)[0]
    if len(hit) and hit[0] > 0:
        dist = float(s[hit[0] - 1])
    return last + h * dist, True, dict(v_end=round(v_end, 4), braking=True, q=round(q, 4),
                                       dist=round(float(dist), 3))


# ------------------------------------------------------------- conflicts
def smooth(x, w=5):
    k = np.ones(w) / w
    return np.convolve(np.pad(x, (w // 2, w // 2), mode='edge'), k, 'valid')


def find_conflicts(agents, radius, coop_pairs):
    out = []
    ids = sorted(agents)
    for ii, i in enumerate(ids):
        for j in ids[ii + 1:]:
            a, b = agents[i], agents[j]
            L = min(len(a), len(b))
            a, b = a[:L], b[:L]
            P = np.c_[a['x'], a['y']]
            Q = np.c_[b['x'], b['y']]
            Va = np.c_[smooth(a['vx']), smooth(a['vy'])]
            Vb = np.c_[smooth(b['vx']), smooth(b['vy'])]
            r = Q - P
            v = Vb - Va
            d = np.hypot(*r.T)
            vv = (v * v).sum(1) + 1e-12
            tc = -(r * v).sum(1) / vv
            dc = np.hypot(*(r + v * np.clip(tc, 0, None)[:, None]).T)
            sa, sb = np.hypot(*Va.T), np.hypot(*Vb.T)
            cosang = (Va * Vb).sum(1) / (sa * sb + 1e-12)
            ang = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
            flag = ((d < SENSOR) & (tc > 0) & (tc < TCPA_MAX) & (dc < 2 * radius)
                    & (sa > 0.05) & (sb > 0.05) & (ang > 45))
            if (i, j) in coop_pairs:
                continue
            # encounters = contiguous windows inside sensor range
            inr = np.where(d < SENSOR)[0]
            if len(inr) == 0:
                continue
            groups, cur = [], [inr[0]]
            for k in inr[1:]:
                if k - cur[-1] <= 3:
                    cur.append(k)
                else:
                    groups.append(cur)
                    cur = [k]
            groups.append(cur)
            for g in groups:
                g = np.array(g)
                fk = g[flag[g]]
                if len(fk) == 0:
                    continue
                k0 = fk[0]
                kmin = int(g[np.argmin(d[g])])
                kind = 'frontal' if ang[k0] >= 150 else 'squeeze'
                step = int(a['step'][kmin])
                out.append([kind, step, i, j])
    out.sort(key=lambda c: (c[1], c[2], c[3]))
    return out


# ------------------------------------------------------------------ main
def assess(folder, verbose=True):
    name, env, agents, walls = load_cave(folder)
    radius = float(env.get('agent_radius', 0.075))
    states = {}
    for i, a in agents.items():
        states[i] = carrying_state(a)
    carriers = sorted(i for i, (c, _, _) in states.items() if c)
    coop = False
    if len(carriers) >= 2:
        # find a cooperating pair: picked at the same time, close together
        pair = None
        for ii, i in enumerate(carriers):
            for j in carriers[ii + 1:]:
                pi_, pj = states[i][1], states[j][1]
                if abs(pi_['start'] - pj['start']) <= 10:
                    ai, aj = agents[i], agents[j]
                    dend = math.hypot(ai['x'][-1] - aj['x'][-1], ai['y'][-1] - aj['y'][-1])
                    if dend < 0.6:
                        pair = (i, j)
        if pair:
            carriers, coop = list(pair), True
        else:
            # keep the most recent pick
            carriers = [max(carriers, key=lambda k: states[k][1]['start'])]
    # cooperating pairs over the whole log (for conflict exclusion)
    coop_pairs = set()
    for i in agents:
        for kind, info in states[i][2]:
            if kind not in ('pick', 'place'):
                continue
            for j in agents:
                if j <= i:
                    continue
                for kind2, info2 in states[j][2]:
                    if kind2 == kind and abs(info2['start'] - info['start']) <= 10:
                        coop_pairs.add((i, j))

    res = dict(name=name, outcome='NO_RESCUE', collected=False, collected_by=None,
               position=None, obstructed_by=None, obstruction_case=None,
               cooperation_flag=False, conflicts=[])
    debug = {}
    if carriers:
        res['collected'] = True
        res['collected_by'] = carriers
        res['cooperation_flag'] = coop
        lead = min(carriers)
        start = np.array([agents[lead]['x'][0], agents[lead]['y'][0]])
        goal = np.mean([[agents[k]['x'][-1], agents[k]['y'][-1]] for k in carriers], axis=0)
        res['position'] = [round(float(goal[0]), 4), round(float(goal[1]), 4)]
        path, _ = plan_path(walls, start, goal)
        debug['path'] = path
        obst, cases = [], []
        if path is not None:
            for i in sorted(agents):
                if i in carriers:
                    continue
                a = agents[i]
                last = np.array([a['x'][-1], a['y'][-1]])
                stop, moving, info = predict_stop(a, walls, radius)
                d_stop = point_polyline_dist(stop, path)
                d_last = point_polyline_dist(last, path)
                debug[i] = dict(last=last.round(3).tolist(), stop=stop.round(3).tolist(),
                                d_last=round(d_last, 3), d_stop=round(d_stop, 3), moving=moving, **info)
                if d_stop < CLEAR + radius:
                    obst.append(i)
                    cases.append('MOVING' if moving else 'STATIC')
        if obst:
            res['outcome'] = 'RESCUE_HARD'
            res['obstructed_by'] = obst
            res['obstruction_case'] = cases
        else:
            res['outcome'] = 'RESCUE'
    res['conflicts'] = find_conflicts(agents, radius, coop_pairs)
    if verbose:
        print(name, 'carriers', carriers, 'coop', coop)
        for i in sorted(agents):
            print('   agent', i, [(k, inf['start'], round(inf['phi_close'], 2), round(inf['phi_release'], 2))
                                  for k, inf in states[i][2]])
        if 'path' in debug and debug['path'] is not None:
            print('   path', np.round(debug['path'], 3).tolist())
        for i in sorted(k for k in debug if isinstance(k, int)):
            print('   robot', i, debug[i])
        print('   ->', json.dumps(res))
    return res


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
    for folder in sorted(glob.glob(os.path.join(base, 'cave*')), key=lambda p: int(''.join(c for c in os.path.basename(p) if c.isdigit()) or 0)):
        if not os.path.isdir(folder):
            continue
        res = assess(folder)
        with open(os.path.join(base, f"{res['name']}_outcome.json"), 'w') as f:
            json.dump(res, f, indent=4)


if __name__ == '__main__':
    main()
