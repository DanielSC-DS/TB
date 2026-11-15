"""Assess each cave: object carrier(s), rescue path with 0.35 wall clearance,
obstructing robots (using predicted stop positions for braking robots) and conflicts.
Usage: python3 assess_caves.py [caves_dir]   (default: ../caves relative to this file)"""
import sys, os, json, glob, heapq
import numpy as np

DT = 0.05
CLEAR = 0.35
CAVES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'caves')

# ---------------------------------------------------------------- loading
def load(c):
    d = os.path.join(CAVES, f'cave{c}')
    env = json.load(open(os.path.join(d, f'cave{c}_env.json')))
    A = [np.loadtxt(os.path.join(d, f'cave{c}_agent_{i}.csv'), delimiter=',', skiprows=1)
         for i in range(env['num_agents'])]
    return env, A

# ---------------------------------------------------------------- gripper events
def runs(mask):
    r, i = [], 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j < len(mask) and mask[j]: j += 1
            r.append((i, j)); i = j
        else: i += 1
    return r

def primitives(a):
    ev = []
    dphi, dop = np.diff(a[:, 6]), np.diff(a[:, 7])
    for s, e in runs(dphi > 1e-3): ev.append([s, e, 'EXT', a[s, 6], a[e, 6]])
    for s, e in runs(dphi < -1e-3): ev.append([s, e, 'RET', a[s, 6], a[e, 6]])
    for s, e in runs(dop > 1e-3): ev.append([s, e, 'CLOSE', a[s, 7], a[e, 7]])
    for s, e in runs(dop < -1e-3): ev.append([s, e, 'OPEN', a[s, 7], a[e, 7]])
    ev.sort(key=lambda x: x[0])
    m = []
    for x in ev:
        if m and m[-1][2] == x[2] and x[0] - m[-1][1] < 5:
            m[-1][1] = x[1]; m[-1][4] = x[4]
        else: m.append(x)
    # at the same start step a CLOSE with retracted arm comes before EXT (put-down sequence)
    m.sort(key=lambda x: (x[0], 0 if (x[2] == 'CLOSE' and a[x[0], 6] < 0.1) else 1))
    return m

def grip_events(a):
    """PICK: extend(full) -> close on object (partial opening, high current) -> retract holding -> open.
       DROP: close (holding, arm retracted) -> extend(full) -> open -> retract."""
    m, out, i = primitives(a), [], 0
    while i < len(m):
        x = m[i]
        if x[2] == 'EXT' and x[3] < 0.1 and x[4] > 3.0:
            seq = m[i + 1:i + 4]
            if [s[2] for s in seq] == ['CLOSE', 'RET', 'OPEN']:
                cl, rt, op = seq
                curr = np.median(a[cl[1] + 3:cl[1] + 15, 8])
                cmin = a[rt[0]:rt[1] + 1, 8].min()
                if 0.05 < cl[4] < 0.95 and curr > 2 and cmin > 2 and rt[4] < 0.1:
                    out.append(('PICK', int(a[cl[0], 0]))); i += 4; continue
            out.append(('FAILPICK', int(a[x[0], 0]))); i += 1; continue
        if x[2] == 'CLOSE' and a[x[0], 6] < 0.1:
            seq = m[i + 1:i + 4]
            curr = np.median(a[x[1] + 3:x[1] + 10, 8])
            if [s[2] for s in seq] == ['EXT', 'OPEN', 'RET'] and seq[0][4] > 3.0 and curr > 2:
                out.append(('DROP', int(a[seq[1][0], 0]))); i += 4; continue
            out.append(('FAILDROP', int(a[x[0], 0]))); i += 1; continue
        i += 1
    return out

def carriers(A):
    res = []
    for k, a in enumerate(A):
        holding = False
        for kind, _ in grip_events(a):
            if kind == 'PICK': holding = True
            elif kind == 'DROP': holding = False
        if holding: res.append(k)
    return res

# ---------------------------------------------------------------- geometry / path
def seg_dist(P, segs):
    P = np.atleast_2d(P)
    A_ = segs[:, 0][None]; B_ = segs[:, 1][None]; p = P[:, None]
    AB = B_ - A_
    t = np.clip(((p - A_) * AB).sum(-1) / np.maximum((AB * AB).sum(-1), 1e-12), 0, 1)
    q = A_ + t[..., None] * AB
    return np.sqrt(((p - q) ** 2).sum(-1)).min(1)

def plan(segs, start, goal, res=0.01, cl=CLEAR):
    lo = segs.reshape(-1, 2).min(0); hi = segs.reshape(-1, 2).max(0)
    nx = int(round((hi[0] - lo[0]) / res)) + 1; ny = int(round((hi[1] - lo[1]) / res)) + 1
    xs = lo[0] + np.arange(nx) * res; ys = lo[1] + np.arange(ny) * res
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    free = (seg_dist(np.c_[X.ravel(), Y.ravel()], segs) >= cl).reshape(nx, ny)
    idx = lambda p: (int(round((p[0] - lo[0]) / res)), int(round((p[1] - lo[1]) / res)))
    s, g = idx(start), idx(goal)
    free[s] = free[g] = True
    nb = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1),
          (2, 1), (1, 2), (-2, 1), (-1, 2), (2, -1), (1, -2), (-2, -1), (-1, -2)]
    dist = np.full((nx, ny), np.inf); prev = {}; dist[s] = 0; pq = [(0.0, s)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]: continue
        if u == g: break
        for dx, dy in nb:
            v = (u[0] + dx, u[1] + dy)
            if 0 <= v[0] < nx and 0 <= v[1] < ny and free[v]:
                if (abs(dx) == 2 or abs(dy) == 2) and not (free[u[0] + np.sign(dx), u[1]] or free[u[0], u[1] + np.sign(dy)]):
                    continue
                nd = d + np.hypot(dx, dy)
                if nd < dist[v]: dist[v] = nd; prev[v] = u; heapq.heappush(pq, (nd, v))
    if not np.isfinite(dist[g]): return None
    cells = [g]
    while cells[-1] != s: cells.append(prev[cells[-1]])
    pts = np.array([[xs[i], ys[j]] for i, j in cells[::-1]]); pts[0] = start; pts[-1] = goal
    def visible(a, b):
        n = max(2, int(np.hypot(*(b - a)) / 0.005) + 1)
        P = a + np.linspace(0, 1, n)[:, None] * (b - a)
        return np.all(seg_dist(P, segs) >= cl - 1e-3)
    out, i = [pts[0]], 0
    while i < len(pts) - 1:          # string pulling -> (near) shortest path
        j = len(pts) - 1
        while j > i + 1 and not visible(pts[i], pts[j]): j -= 1
        out.append(pts[j]); i = j
    return np.array(out)

def dist_to_path(p, path):
    return float(seg_dist(np.asarray(p)[None], np.stack([path[:-1], path[1:]], 1))[0])

# ---------------------------------------------------------------- braking / stop prediction
def ps_of(a):  # speed from positions (velocity column is noisy)
    return np.hypot(*np.diff(a[:, 2:4], axis=0).T) / DT

K = 6
def brake_library(all_A):
    """Windows of observed speed from every complete braking manoeuvre in the logs,
    labelled with the distance still travelled until the robot stopped."""
    F, R = [], []
    for A in all_A:
        for a in A:
            ps = ps_of(a); n = len(ps); i = 0
            while i < n - 1:
                if ps[i] > 0.1 and ps[i + 1] < ps[i]:
                    j = i
                    while j < n - 1 and ps[j + 1] <= ps[j] + 1e-4: j += 1
                    if j < n - 1 and ps[j] < 0.03 and j - i > 20:
                        seg = ps[i:j + 1]; tail = ps[j] * 1.9
                        cum = np.cumsum(seg[::-1])[::-1] * DT
                        for t in range(K, len(seg)):
                            F.append(seg[t - K + 1:t + 1]); R.append(cum[t] - seg[t] * DT + tail)
                    i = j + 1
                else: i += 1
    return np.array(F), np.array(R)

def end_state(a, segs, lib, r=0.075):
    """STILL / MOVING (cruising, accelerating, turning) / BRAKING with expected stop point."""
    ps = ps_of(a); v = ps[-1]
    if v < 0.015:
        return 'STILL', a[-1, 2:4]
    d = np.diff(ps[-4:])
    # cruising robots lose ~0.002 m/s per step; braking is a clearly faster, sustained decrease
    if not (d[-1] < 0 and -d.mean() > 0.003):
        return 'MOVING', a[-1, 2:4]
    F, R = lib
    err = np.sqrt(((F - ps[-K:]) ** 2).mean(1)); idx = np.argsort(err)[:5]
    rem = float(np.median(R[idx])) if err[idx].max() < 0.01 else 1.95 * v   # ~v/k, k=1/1.9s^-1
    u = a[-1, 2:4] - a[-2, 2:4]; u /= np.linalg.norm(u)
    pos = a[-1, 2:4].copy()
    for s in np.arange(0, rem, 0.005):             # straight-line stop, halted by walls
        p = a[-1, 2:4] + u * s
        if seg_dist(p, segs)[0] < r: break
        pos = p
    return 'BRAKING', pos

# ---------------------------------------------------------------- conflicts
def conflicts(A, r):
    out = []
    n = len(A)
    for i in range(n):
        for j in range(i + 1, n):
            L = min(len(A[i]), len(A[j]))
            P1, P2 = A[i][:L, 2:4], A[j][:L, 2:4]
            V1 = np.vstack([[0, 0], np.diff(P1, axis=0) / DT]); V2 = np.vstack([[0, 0], np.diff(P2, axis=0) / DT])
            d = np.hypot(*(P1 - P2).T); inside = d < 0.5     # sensor range
            k = 0
            while k < L:
                if not inside[k]: k += 1; continue
                e = k
                while e < L and inside[e]: e += 1
                s = max(k, 1)                                  # moment of detection
                s1, s2 = np.hypot(*V1[s]), np.hypot(*V2[s])
                dp, dv = P2[s] - P1[s], V2[s] - V1[s]
                tc = -np.dot(dp, dv) / max(np.dot(dv, dv), 1e-12)
                miss = np.hypot(*(dp + dv * max(tc, 0)))
                if s1 > 0.05 and s2 > 0.05 and 0 < tc < 3.0 and miss < 2 * r:
                    ang = np.degrees(np.arccos(np.clip(np.dot(V1[s], V2[s]) / (s1 * s2), -1, 1)))
                    t = k + int(np.argmin(d[k:e]))
                    kind = 'frontal' if ang >= 150 else ('squeeze' if ang > 30 else None)
                    if kind: out.append([kind, int(A[i][t, 0]), i, j])
                k = e
    return sorted(out, key=lambda x: x[1])

# ---------------------------------------------------------------- main
def main():
    ids = sorted(int(os.path.basename(p)[4:]) for p in glob.glob(os.path.join(CAVES, 'cave*')) if os.path.isdir(p))
    data = {c: load(c) for c in ids}
    lib = brake_library([data[c][1] for c in ids])
    for c in ids:
        env, A = data[c]; segs = np.array(env['scenario'], float); r = env['agent_radius']
        cr = carriers(A)
        out = dict(name=f'cave{c}', outcome='NO_RESCUE', collected=False, collected_by=None, position=None,
                   obstructed_by=None, obstruction_case=None, cooperation_flag=False, conflicts=conflicts(A, r))
        if cr:
            start = A[min(cr)][0, 2:4]
            goal = np.mean([A[k][-1, 2:4] for k in cr], 0)
            path = plan(segs, start, goal)
            obs, case = [], []
            for k, a in enumerate(A):
                if k in cr: continue
                st, pos = end_state(a, segs, lib)
                if dist_to_path(pos, path) < CLEAR + r:
                    obs.append(k); case.append('STATIC' if st == 'STILL' else 'MOVING')
            out.update(collected=True, collected_by=cr, position=[round(float(x), 4) for x in goal],
                       cooperation_flag=len(cr) == 2, outcome='RESCUE_HARD' if obs else 'RESCUE',
                       obstructed_by=obs or None, obstruction_case=case or None)
        with open(os.path.join(CAVES, f'cave{c}_outcome.json'), 'w') as f:
            json.dump(out, f, indent=4)
        print(json.dumps(out))

if __name__ == '__main__':
    main()
