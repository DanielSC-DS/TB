#!/usr/bin/env python3
"""Assess each cave in app/caves/caveX and write caveX_outcome.json.

Pipeline per cave:
  1. Gripper/arm state machine per robot -> who carries an object at log end
     (valid pickup: extend open, close on object at phi=pi with high current and
     partial closure, retract closed, open at phi~0 to rest object on robot;
     put-down: close at phi~0, extend, open at phi=pi, retract).
  2. Shortest path with 0.35 wall clearance (grid Dijkstra + exact string pulling)
     from the first logged position of the (lowest-ID) carrier to the carrier
     position (midpoint for a cooperating pair).
  3. Every other robot is judged at its last position, or, if braking at log end,
     at its predicted stop position (exponential braking fitted on the log).
     It obstructs if its body intrudes in the 0.35 clearance around the path.
  4. Conflicts: within the 0.5 m sensor range, both robots moving and on a
     collision course (closest point of approach < 2 radii, in the future).
     frontal: headings > 150 deg apart; squeeze: 45..150 deg apart.
"""
import glob, heapq, json, os, sys
import numpy as np
import pandas as pd

DT = 0.05
CLEARANCE = 0.35
SENSOR = 0.5
HERE = os.path.dirname(os.path.abspath(__file__))
CAVES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "caves")


# ---------------------------------------------------------------- geometry
def pt_seg_dist(P, segs):
    P = np.asarray(P, float)
    out = np.full(P.shape[:-1], np.inf)
    for a, b in segs:
        ab = b - a
        t = np.clip(((P - a) @ ab) / (ab @ ab), 0, 1)
        out = np.minimum(out, np.linalg.norm(P - (a + t[..., None] * ab), axis=-1))
    return out


def _psd(p, a, b):
    ab = b - a
    L = ab @ ab
    t = 0.0 if L == 0 else np.clip((p - a) @ ab / L, 0, 1)
    return np.linalg.norm(p - (a + t * ab))


def seg_seg_dist(p1, p2, q1, q2):
    def cr(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) - (b[1] - a[1]) * (c[0] - a[0])
    if cr(p1, p2, q1) * cr(p1, p2, q2) < 0 and cr(q1, q2, p1) * cr(q1, q2, p2) < 0:
        return 0.0
    return min(_psd(p1, q1, q2), _psd(p2, q1, q2), _psd(q1, p1, p2), _psd(q2, p1, p2))


def seg_clear(p, q, segs):
    return min(seg_seg_dist(p, q, a, b) for a, b in segs)


def plan(segs, start, goal, cl=CLEARANCE, res=0.01):
    """Shortest path keeping `cl` from walls (endpoints may be closer)."""
    start, goal = np.asarray(start, float), np.asarray(goal, float)
    mn = segs.reshape(-1, 2).min(0) - 0.05
    mx = segs.reshape(-1, 2).max(0) + 0.05
    xs, ys = np.arange(mn[0], mx[0], res), np.arange(mn[1], mx[1], res)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    D = pt_seg_dist(np.stack([X, Y], -1), segs)
    sc = float(pt_seg_dist(start[None], segs)[0])
    gc = float(pt_seg_dist(goal[None], segs)[0])
    idx = lambda p: (int(round((p[0] - mn[0]) / res)), int(round((p[1] - mn[1]) / res)))
    s, g = idx(start), idx(goal)

    def ok(c):
        if D[c] >= cl:
            return True
        if np.hypot(c[0] - s[0], c[1] - s[1]) * res < 0.4 and D[c] >= sc - 1e-9:
            return True
        return np.hypot(c[0] - g[0], c[1] - g[1]) * res < 0.4 and D[c] >= gc - 1e-9

    nb = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1),
          (1, 2), (2, 1), (-1, 2), (-2, 1), (1, -2), (2, -1), (-1, -2), (-2, -1)]
    dist = np.full(D.shape, np.inf)
    prev = {}
    dist[s] = 0
    pq = [(0.0, s)]
    while pq:
        dd, c = heapq.heappop(pq)
        if c == g:
            break
        if dd > dist[c]:
            continue
        for dx, dy in nb:
            n = (c[0] + dx, c[1] + dy)
            if not (0 <= n[0] < D.shape[0] and 0 <= n[1] < D.shape[1]) or not ok(n):
                continue
            if abs(dx) + abs(dy) == 3:
                m = (c[0] + int(np.sign(dx)) * (abs(dx) == 2), c[1] + int(np.sign(dy)) * (abs(dy) == 2))
                if not ok(m):
                    continue
            nd = dd + np.hypot(dx, dy) * res
            if nd < dist[n]:
                dist[n] = nd
                prev[n] = c
                heapq.heappush(pq, (nd, n))
    if not np.isfinite(dist[g]):
        return None
    cells = [g]
    while cells[-1] != s:
        cells.append(prev[cells[-1]])
    cells = cells[::-1]
    pts = [start] + [np.array([xs[i], ys[j]]) for i, j in cells[1:-1]] + [goal]
    out, i = [pts[0]], 0
    while i < len(pts) - 1:  # string pulling with exact clearance test
        j = len(pts) - 1
        while j > i + 1:
            thr = cl
            if i == 0:
                thr = min(thr, sc)
            if j == len(pts) - 1:
                thr = min(thr, gc)
            if seg_clear(pts[i], pts[j], segs) >= thr - 1e-3:
                break
            j -= 1
        out.append(pts[j])
        i = j
    return np.array(out)


def pt_path_dist(p, path):
    return float(pt_seg_dist(np.asarray(p, float)[None], np.stack([path[:-1], path[1:]], 1))[0])


# ---------------------------------------------------------------- gripper
def carry_history(d):
    """Return (carrying_at_end, pickup_step_of_current_object, events)."""
    phi, op, cu, st = d.phi.values, d.open.values, d.curr.values, d.step.values
    n = len(d)
    carrying, pick_step, events = False, None, []
    i = 1
    while i < n:
        if op[i] > 1e-3 and op[i - 1] <= 1e-3:  # gripper starts closing
            j = i
            while j + 1 < n and op[j + 1] > 1e-3:
                j += 1  # j = last closed sample
            k0 = i
            closed = op[i:j + 1]
            held = cu[i:j + 1]
            closure = closed.max()
            holding = held.max() > 2.5 and closure < 0.95
            released_at = phi[j + 1] if j + 1 < n else None
            if not carrying and phi[i] > 3.1:
                ok = (holding and released_at is not None and released_at < 0.05
                      and phi[i:j + 1].min() < 0.05)
                events.append(("pickup" if ok else "failed_pickup", int(st[k0])))
                if ok:
                    carrying, pick_step = True, int(st[k0])
            elif carrying and phi[i] < 0.05:
                ok = released_at is not None and released_at > 3.1 and phi[i:j + 1].max() > 3.1
                events.append(("putdown" if ok else "failed_putdown", int(st[k0])))
                if ok:
                    carrying, pick_step = False, None
            else:
                events.append(("anomalous_grasp", int(st[k0])))
            i = j + 1
        i += 1
    return carrying, pick_step, events


# ---------------------------------------------------------------- motion at log end
def braking_constant(fd, W=10, default=0.52):
    """Per-robot exponential braking rate k [1/s] (v' = -k v), calibrated on the
    robot's own earlier clean decelerations (speeds from positions, so velocity
    reading noise does not matter)."""
    ks = []
    t = np.arange(W) * DT
    for i in range(0, len(fd) - W - 20):
        w = fd[i:i + W]
        if w.min() < 0.05 or w.max() > 0.6 or np.any(np.diff(w) > 0):
            continue
        p = np.polyfit(t, np.log(w), 1)
        if np.std(np.log(w) - np.polyval(p, t)) < 0.01 and -p[0] > 0.1:
            ks.append(-p[0])
    return float(np.median(ks)) if ks else default


def end_state(d, segs):
    """STATIC / BRAKING / MOVING and the position the robot is judged by."""
    x, y = d.x.values, d.y.values
    fd = np.hypot(np.diff(x), np.diff(y)) / DT
    last = np.array([x[-1], y[-1]])
    v = float(np.mean(fd[-3:]))
    if v < 0.03:
        return "STATIC", last, last, v
    slope = np.polyfit(np.arange(8) * DT, fd[-8:], 1)[0]
    if slope < -0.03:  # reducing speed at log end -> judged where it will stop
        k = braking_constant(fd)
        h = np.array([x[-1] - x[-6], y[-1] - y[-6]])
        h /= np.linalg.norm(h) + 1e-12
        D = v / k
        # it cannot drive through a wall: stop one radius before the first wall hit
        for s in np.arange(0, D + 1e-9, 0.01):
            if pt_seg_dist((last + h * s)[None], segs)[0] < 0.075:
                D = max(s - 0.01, 0.0)
                break
        return "BRAKING", last, last + h * D, v
    return "MOVING", last, last, v


# ---------------------------------------------------------------- conflicts
def conflicts(ag, radius, coop_pairs):
    out = []
    n = len(ag)
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) in coop_pairs:
                continue
            L = min(len(ag[i]), len(ag[j]))
            A, B = ag[i].iloc[:L], ag[j].iloc[:L]
            P, Q = A[["x", "y"]].values, B[["x", "y"]].values
            VA, VB = A[["vx", "vy"]].values, B[["vx", "vy"]].values
            dist = np.linalg.norm(P - Q, axis=1)
            rel, rv = Q - P, VB - VA
            tc = -(rel * rv).sum(1) / ((rv * rv).sum(1) + 1e-12)
            dc = np.linalg.norm(rel + rv * np.clip(tc, 0, None)[:, None], axis=1)
            sa, sb = np.linalg.norm(VA, axis=1), np.linalg.norm(VB, axis=1)
            cosang = (VA * VB).sum(1) / (sa * sb + 1e-12)
            ang = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
            flag = ((dist < SENSOR) & (tc > 0) & (tc < 3.0) & (dc < 2 * radius)
                    & (sa > 0.05) & (sb > 0.05) & (ang > 45))
            idx = np.where(dist < SENSOR)[0]
            if not len(idx):
                continue
            for b in np.split(idx, np.where(np.diff(idx) > 1)[0] + 1):
                f = b[flag[b]]
                if not len(f):
                    continue
                k = b[np.argmin(dist[b])]
                kind = "frontal" if ang[f[0]] > 150 else "squeeze"
                out.append([kind, int(A.step.values[k]), i, j])
    return sorted(out, key=lambda e: (e[1], e[2], e[3]))


# ---------------------------------------------------------------- main
def assess(cdir):
    name = os.path.basename(cdir.rstrip("/"))
    env = json.load(open(os.path.join(cdir, f"{name}_env.json")))
    segs = np.array(env["scenario"], float)
    radius = float(env["agent_radius"])
    ag = [pd.read_csv(os.path.join(cdir, f"{name}_agent_{a}.csv")) for a in range(env["num_agents"])]

    hist = [carry_history(d) for d in ag]
    carriers = [a for a, h in enumerate(hist) if h[0]]
    coop = False
    if len(carriers) == 2 and abs(hist[carriers[0]][1] - hist[carriers[1]][1]) <= 5:
        coop = True
    elif len(carriers) > 1:
        raise RuntimeError(f"{name}: ambiguous carriers {carriers}")

    # pairs that picked up together (cooperative transport) are not conflicts
    coop_pairs = set()
    for a in range(len(ag)):
        for b in range(a + 1, len(ag)):
            pa = [s for e, s in hist[a][2] if e == "pickup"]
            pb = [s for e, s in hist[b][2] if e == "pickup"]
            if any(abs(x - y) <= 5 for x in pa for y in pb):
                coop_pairs.add((a, b))

    res = dict(name=name, outcome="NO_RESCUE", collected=False, collected_by=None,
               position=None, obstructed_by=None, obstruction_case=None,
               cooperation_flag=False, conflicts=conflicts(ag, radius, coop_pairs))
    if carriers:
        start = ag[carriers[0]][["x", "y"]].values[0]
        goal = np.mean([ag[a][["x", "y"]].values[-1] for a in carriers], 0)
        path = plan(segs, start, goal)
        if path is None:
            raise RuntimeError(f"{name}: no path with clearance")
        obs, case = [], []
        for a in range(len(ag)):
            if a in carriers:
                continue
            state, last, judged, v = end_state(ag[a], segs)
            dd = pt_path_dist(judged, path)
            if dd < CLEARANCE + radius:
                obs.append(a)
                case.append("STATIC" if state == "STATIC" else "MOVING")
        res.update(collected=True, collected_by=carriers,
                   position=[round(float(goal[0]), 4), round(float(goal[1]), 4)],
                   cooperation_flag=coop,
                   outcome="RESCUE_HARD" if obs else "RESCUE",
                   obstructed_by=obs or None, obstruction_case=case or None)
    return res


if __name__ == "__main__":
    for cdir in sorted(glob.glob(os.path.join(CAVES, "cave*")), key=lambda p: int(p.rsplit("cave", 1)[1])):
        if not os.path.isdir(cdir):
            continue
        r = assess(cdir)
        with open(os.path.join(cdir, f"{r['name']}_outcome.json"), "w") as f:
            json.dump(r, f, indent=4)
        print(json.dumps(r))
