#!/usr/bin/env python3
"""Assess robot cave logs and write caveX_outcome.json files.

Usage: python3 assess_caves.py [caves_dir]      (default: app/caves next to this tools dir)

Only numpy and pandas are required.

Method summary
--------------
* Gripper: a successful pick-up is arm fully extended (phi ~ pi) with the gripper
  open, gripper closing to a partial opening (object between the fingers, not fully
  closed = miss) with high holding current, and the arm retracting to rest while the
  object is still held. A put-down is the reverse cycle (close at rest, extend, open).
  Cycles where the object is released before the arm is retracted are drops.
* Rescue path: shortest path on a 1 cm grid (16-connected Dijkstra) through the
  free space whose distance to every wall segment is >= 0.35, followed by
  string-pulling with exact clearance checks.
* Obstruction: a robot interferes with the clearance band when its body
  (radius from the env file) intersects it, i.e. centre distance < 0.35 + r.
  Robots that are decelerating at the end of their log are judged at the expected
  stop point. Robots follow a proportional approach law v = k * distance-to-goal
  (k ~ 0.524 1/s identified from the logs), so the remaining distance is v / k
  (using the robot's own fitted decay rate when it is already in the clean
  exponential regime). The stop point is clipped at walls.
* Conflicts: while two robots are within sensor range (0.5 m), the constant-velocity
  closest point of approach is checked; a predicted miss distance below two radii
  within 3 s is a collision course. Opposite headings (>= 150 deg) = frontal,
  other crossing directions (>= 45 deg) = squeeze. The reported step is the step of
  minimum separation in that encounter.
"""
import heapq, json, os, sys
import numpy as np
import pandas as pd

DT = 0.05
CLEARANCE = 0.35
SENSOR = 0.5
K_APPROACH = 0.524
HORIZON = 3.0


def load(cdir):
    name = os.path.basename(os.path.normpath(cdir))
    env = json.load(open(os.path.join(cdir, f'{name}_env.json')))
    ags = [pd.read_csv(os.path.join(cdir, f'{name}_agent_{i}.csv')) for i in range(env['num_agents'])]
    return name, env, ags, np.array(env['scenario'], float)


# ------------------------------------------------------------------ gripper
def grip_events(d):
    ph, op, cu = d.phi.values, d.open.values, d.curr.values
    n = len(d)
    ev, carrying, i = [], False, 0
    while i < n:
        if not carrying and ph[i] > 3.1 and op[i] < 0.005 and i + 1 < n and op[i + 1] > 0.005:
            j = i + 1
            while j + 1 < n and op[j + 1] > op[j] + 1e-4:
                j += 1
            plateau = op[j]
            k, ok = j, True
            while k < n and ph[k] > 0.03:
                if op[k] < plateau - 0.02:
                    ok = False
                    break
                k += 1
            if k >= n:
                ok = False
            ok = ok and 0.05 < plateau < 0.95 and np.median(cu[j:max(k, j + 1)]) > 2.5
            ev.append(dict(kind='pick', ok=bool(ok), step=int(d.step[j]), idx=j))
            carrying = carrying or ok
            i = max(k, i) + 1
            continue
        if carrying and ph[i] < 0.03 and op[i] < 0.005 and i + 1 < n and op[i + 1] > 0.005:
            j = i + 1
            while j + 1 < n and op[j + 1] > op[j] + 1e-4:
                j += 1
            k = j
            while k < n and op[k] > 0.005:
                k += 1
            ok = k < n and ph[j:k].max() > 3.1
            ev.append(dict(kind='put', ok=bool(ok), step=int(d.step[min(k, n - 1)]), idx=k))
            if ok:
                carrying = False
            i = k + 1
            continue
        i += 1
    return ev, carrying


# ------------------------------------------------------------------ geometry
def wall_dist(P, walls):
    P = np.asarray(P, float)
    sh = P.shape[:-1]
    P = P.reshape(-1, 2)
    best = np.full(len(P), np.inf)
    for a, b in walls:
        ab = b - a
        t = np.clip(((P - a) @ ab) / (ab @ ab), 0, 1)
        best = np.minimum(best, np.hypot(*(P - (a + t[:, None] * ab)).T))
    return best.reshape(sh)


def poly_dist(p, poly):
    best = np.inf
    for a, b in zip(poly[:-1], poly[1:]):
        ab = b - a
        t = np.clip(((p - a) @ ab) / max(ab @ ab, 1e-12), 0, 1)
        best = min(best, float(np.hypot(*(p - (a + t * ab)))))
    return best


def shortest_path(walls, a, b, res=0.01, cl=CLEARANCE):
    x0, y0 = walls[:, :, 0].min(), walls[:, :, 1].min()
    nx = int(round((walls[:, :, 0].max() - x0) / res)) + 1
    ny = int(round((walls[:, :, 1].max() - y0) / res)) + 1
    X, Y = np.meshgrid(x0 + np.arange(nx) * res, y0 + np.arange(ny) * res, indexing='ij')
    free = wall_dist(np.stack([X, Y], -1), walls) >= cl
    idx = lambda p: (int(round((p[0] - x0) / res)), int(round((p[1] - y0) / res)))
    ia, ib = idx(a), idx(b)
    free[ia] = free[ib] = True
    nb = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1),
          (2, 1), (1, 2), (-2, 1), (-1, 2), (2, -1), (1, -2), (-2, -1), (-1, -2)]
    dist = np.full((nx, ny), np.inf)
    prev = {}
    dist[ia] = 0
    h = [(0.0, ia)]
    while h:
        dc, u = heapq.heappop(h)
        if u == ib:
            break
        if dc > dist[u]:
            continue
        for di, dj in nb:
            v = (u[0] + di, u[1] + dj)
            if not (0 <= v[0] < nx and 0 <= v[1] < ny) or not free[v]:
                continue
            if abs(di) == 2 or abs(dj) == 2:
                m = (u[0] + (di // 2 if abs(di) == 2 else 0), u[1] + (dj // 2 if abs(dj) == 2 else 0))
                if not free[m]:
                    continue
            nd = dc + np.hypot(di, dj)
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(h, (nd, v))
    if not np.isfinite(dist[ib]):
        return None
    cells = [ib]
    while cells[-1] != ia:
        cells.append(prev[cells[-1]])
    pts = [np.array([x0 + i * res, y0 + j * res]) for i, j in cells[::-1]]
    pts[0], pts[-1] = np.asarray(a, float), np.asarray(b, float)

    def visible(p, q):
        n = max(2, int(np.hypot(*(q - p)) / 0.004) + 1)
        s = p + np.linspace(0, 1, n)[:, None] * (q - p)
        return bool((wall_dist(s, walls) >= cl - 1e-3).all())

    out, i = [pts[0]], 0
    while i < len(pts) - 1:
        j = len(pts) - 1
        while j > i + 1 and not visible(pts[i], pts[j]):
            j -= 1
        out.append(pts[j])
        i = j
    k = 1
    while k < len(out) - 1:
        if visible(out[k - 1], out[k + 1]):
            out.pop(k)
        else:
            k += 1
    return np.array(out)


# ------------------------------------------------------------------ motion at end of log
def end_state(d, walls, radius):
    x, y = d.x.values, d.y.values
    s = np.hypot(np.diff(x), np.diff(y)) / DT
    last = np.array([x[-1], y[-1]])
    v = float(s[-1])
    moving = v > 0.02
    decel = moving and s[-1] < s[-4] - 0.003 and s[-1] <= s[-2] + 1e-4
    if not decel:
        return dict(moving=moving, decel=False, pos=last, rem=0.0)
    seg = s[-8:]
    t = np.arange(len(seg)) * DT
    kfit = -np.polyfit(t, np.log(np.maximum(seg, 1e-9)), 1)[0]
    k = kfit if kfit >= 0.45 else K_APPROACH
    rem = v / k
    h = np.array([x[-1] - x[-4], y[-1] - y[-4]])
    h /= np.linalg.norm(h)
    q = last
    for st in np.arange(0.005, rem + 1e-9, 0.005):
        c = last + h * st
        if wall_dist(c[None], walls)[0] < radius:
            break
        q = c
    return dict(moving=True, decel=True, pos=q, rem=rem)


# ------------------------------------------------------------------ conflicts
def conflicts(ags, radius):
    out = []
    n = len(ags)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = ags[i], ags[j]
            L = min(len(a), len(b))
            P = np.c_[a.x[:L], a.y[:L]]
            Q = np.c_[b.x[:L], b.y[:L]]
            VA = np.c_[a.vx[:L], a.vy[:L]]
            VB = np.c_[b.vx[:L], b.vy[:L]]
            D = np.hypot(*(P - Q).T)
            idx = np.where(D < SENSOR)[0]
            if not len(idx):
                continue
            for g in np.split(idx, np.where(np.diff(idx) > 1)[0] + 1):
                kind = None
                for k in g:
                    r, rv = Q[k] - P[k], VB[k] - VA[k]
                    sa, sb = np.hypot(*VA[k]), np.hypot(*VB[k])
                    if sa < 0.05 or sb < 0.05 or rv @ rv < 1e-12:
                        continue
                    tc = -(r @ rv) / (rv @ rv)
                    if not (0 < tc <= HORIZON):
                        continue
                    if np.hypot(*(r + tc * rv)) >= 2 * radius:
                        continue
                    ang = np.degrees(np.arccos(np.clip(VA[k] @ VB[k] / (sa * sb), -1, 1)))
                    if ang >= 150:
                        kind = 'frontal'
                    elif ang >= 45:
                        kind = 'squeeze'
                    if kind:
                        break
                if kind:
                    km = int(g[np.argmin(D[g])])
                    out.append([kind, int(a.step[km]), i, j])
    return sorted(out, key=lambda c: (c[1], c[2], c[3]))


# ------------------------------------------------------------------ main
def assess(cdir):
    name, env, ags, walls = load(cdir)
    radius = float(env['agent_radius'])
    r4 = lambda v: round(float(v), 4)
    evs = {r: grip_events(d) for r, d in enumerate(ags)}
    carriers = sorted(r for r, (_, car) in evs.items() if car)

    # chronological successful pick-ups, grouping simultaneous ones (cooperation)
    picks = sorted((e['step'], r, e['idx']) for r, (ev, _) in evs.items() for e in ev if e['kind'] == 'pick' and e['ok'])
    groups = []
    for st, r, ix in picks:
        if groups and abs(groups[-1][0] - st) <= 3:
            groups[-1][1].append((r, ix))
        else:
            groups.append([st, [(r, ix)]])
    handled = []
    for _, g in groups:
        handled += sorted(r for r, _ in g)
    found_in = None
    if groups:
        g = sorted(groups[0][1])
        pos = np.mean([[ags[r].x[ix], ags[r].y[ix]] for r, ix in g], axis=0)
        found_in = [g[0][0], r4(pos[0]), r4(pos[1])]

    res = dict(name=name, outcome='NO_RESCUE', collected=bool(carriers),
               collected_by=carriers or None, position=None, obstructed_by=None,
               obstruction_case=None, cooperation_flag=len(carriers) == 2,
               conflicts=conflicts(ags, radius), found_in=found_in, handled_object=handled)
    if carriers:
        lo = carriers[0]
        start = np.array([ags[lo].x.iloc[0], ags[lo].y.iloc[0]])
        end = np.mean([[ags[r].x.iloc[-1], ags[r].y.iloc[-1]] for r in carriers], axis=0)
        res['position'] = [r4(end[0]), r4(end[1])]
        path = shortest_path(walls, start, end)
        obs, case = [], []
        for r, d in enumerate(ags):
            if r in carriers:
                continue
            st = end_state(d, walls, radius)
            if poly_dist(st['pos'], path) < CLEARANCE + radius:
                obs.append(r)
                case.append('MOVING' if st['moving'] else 'STATIC')
        if obs:
            res.update(outcome='RESCUE_HARD', obstructed_by=obs, obstruction_case=case)
        else:
            res['outcome'] = 'RESCUE'
    return res


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'caves')
    base = os.path.abspath(base)
    for cdir in sorted(os.listdir(base), key=lambda s: (len(s), s)):
        full = os.path.join(base, cdir)
        if not (os.path.isdir(full) and cdir.startswith('cave')):
            continue
        res = assess(full)
        with open(os.path.join(base, f'{cdir}_outcome.json'), 'w') as f:
            json.dump(res, f, indent=4)
        print(json.dumps(res))


if __name__ == '__main__':
    main()
