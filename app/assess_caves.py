#!/usr/bin/env python3
"""Assess robot cave logs for a rescue mission.

For every /app/caves/caveX folder (or the folder given as argv[1]) this writes
caveX_outcome.json with the carrier, rescue-path obstruction and conflict info.
"""
import glob
import heapq
import json
import math
import os
import sys

import numpy as np
import pandas as pd

CLEARANCE = 0.35       # required rescue-path clearance (from path centre)
SENSOR = 0.5           # detection range of the robots
HOLD_CURR = 1.8        # current above which the gripper is clamped on something
EMPTY_OPEN = 0.95      # gripper fully closed -> nothing grasped
ARM_EXT = 3.0          # phi considered "extended"
ARM_RET = 0.1          # phi considered "retracted"
GRID = 0.01            # planning grid resolution
STILL_SPEED = 0.02     # speed below which a robot is considered still


# --------------------------------------------------------------------------- io
def load_cave(folder):
    name = os.path.basename(folder.rstrip("/"))
    env = json.load(open(os.path.join(folder, f"{name}_env.json")))
    agents = {}
    for f in glob.glob(os.path.join(folder, f"{name}_agent_*.csv")):
        aid = int(f.rsplit("_", 1)[1].split(".")[0])
        agents[aid] = pd.read_csv(f)
    return name, env, dict(sorted(agents.items()))


# -------------------------------------------------------------- gripper events
def hold_segments(d):
    """Contiguous runs where the gripper current shows a clamp."""
    h = (d.curr.values > HOLD_CURR)
    n = len(d)
    segs, i = [], 0
    while i < n:
        if h[i]:
            j = i
            # tolerate single-sample dips in the current reading
            while j < n and (h[j] or (j + 1 < n and h[j + 1])):
                j += 1
            segs.append((i, j - 1))
            i = j
        else:
            i += 1
    return segs


def gripper_events(d):
    """Return list of (kind, step_index) with kind in {'pick', 'drop'}.

    pick: gripper clamps (not fully closed) while the arm is extended, the arm
          retracts completely while clamped and the gripper then opens with
          the arm retracted -> object resting on the robot.
    drop: gripper clamps with the arm retracted (object resting on robot),
          the arm extends while clamped and the gripper opens extended.
    """
    phi, opn = d.phi.values, d.open.values
    ev = []
    for s0, s1 in hold_segments(d):
        grip = np.median(opn[s0:s1 + 1])
        if grip >= EMPTY_OPEN:            # closed on nothing
            continue
        rel = min(s1 + 1, len(d) - 1)     # step where the clamp is released
        if phi[s0] > ARM_EXT and phi[s0:s1 + 1].min() < ARM_RET and phi[rel] < ARM_RET:
            ev.append(("pick", s0))
        elif phi[s0] < 1.0 and phi[max(0, s0 - 15)] < ARM_RET and phi[s0:s1 + 1].max() > ARM_EXT \
                and phi[rel] > ARM_EXT:
            ev.append(("drop", s0))
    return ev


def object_history(agents):
    """Chronological pick/drop events, grouping simultaneous pairs."""
    raw = []
    for aid, d in agents.items():
        for kind, i in gripper_events(d):
            raw.append((int(d.step.iloc[i]), kind, aid, i))
    raw.sort()
    groups = []
    for step, kind, aid, i in raw:
        if groups and groups[-1]["kind"] == kind and abs(groups[-1]["step"] - step) <= 3 \
                and aid not in groups[-1]["agents"]:
            groups[-1]["agents"].append(aid)
            groups[-1]["idx"].append(i)
        else:
            groups.append(dict(step=step, kind=kind, agents=[aid], idx=[i]))
    for g in groups:
        order = np.argsort(g["agents"])
        g["agents"] = [g["agents"][k] for k in order]
        g["idx"] = [g["idx"][k] for k in order]
    return groups


# ------------------------------------------------------------------- geometry
def seg_dist(px, py, segs):
    """Distance from points (arrays) to the nearest of the segments."""
    best = np.full(np.shape(px), np.inf)
    for (ax, ay), (bx, by) in segs:
        dx, dy = bx - ax, by - ay
        L2 = dx * dx + dy * dy
        t = np.clip(((px - ax) * dx + (py - ay) * dy) / L2, 0, 1) if L2 > 0 else 0
        best = np.minimum(best, np.hypot(px - (ax + t * dx), py - (ay + t * dy)))
    return best


def point_poly_dist(p, poly):
    segs = [(poly[k], poly[k + 1]) for k in range(len(poly) - 1)]
    if not segs:
        return math.dist(p, poly[0])
    return float(seg_dist(np.array(p[0]), np.array(p[1]), segs))


class Planner:
    def __init__(self, walls):
        self.walls = walls
        pts = np.array(walls).reshape(-1, 2)
        self.x0, self.y0 = pts.min(0) - 0.05
        x1, y1 = pts.max(0) + 0.05
        self.nx = int(math.ceil((x1 - self.x0) / GRID)) + 1
        self.ny = int(math.ceil((y1 - self.y0) / GRID)) + 1
        xs = self.x0 + np.arange(self.nx) * GRID
        ys = self.y0 + np.arange(self.ny) * GRID
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        self.clear = seg_dist(X, Y, walls)

    def cell(self, p):
        return (int(round((p[0] - self.x0) / GRID)), int(round((p[1] - self.y0) / GRID)))

    def xy(self, c):
        return (self.x0 + c[0] * GRID, self.y0 + c[1] * GRID)

    def free(self, c):
        return 0 <= c[0] < self.nx and 0 <= c[1] < self.ny and self.clear[c] >= CLEARANCE

    def line_ok(self, a, b):
        n = max(2, int(math.dist(a, b) / (GRID * 0.5)) + 1)
        xs = np.linspace(a[0], b[0], n)
        ys = np.linspace(a[1], b[1], n)
        return bool(np.all(seg_dist(xs, ys, self.walls) >= CLEARANCE - 1e-9))

    def nearest_free(self, c):
        if self.free(c):
            return c
        best, bd = None, 1e9
        r = 1
        while best is None and r < 200:
            for i in range(c[0] - r, c[0] + r + 1):
                for j in range(c[1] - r, c[1] + r + 1):
                    if self.free((i, j)):
                        dd = math.hypot(i - c[0], j - c[1])
                        if dd < bd:
                            best, bd = (i, j), dd
            r += 1
        return best

    def shortest_path(self, a, b):
        """Shortest path keeping CLEARANCE from walls (16-connected Dijkstra +
        string pulling).  Returns list of (x, y) points."""
        sa, sb = self.nearest_free(self.cell(a)), self.nearest_free(self.cell(b))
        moves = []
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if (dx, dy) != (0, 0) and math.gcd(abs(dx), abs(dy)) == 1:
                    moves.append((dx, dy, math.hypot(dx, dy)))
        free = self.clear >= CLEARANCE
        dist = np.full(free.shape, np.inf)
        prev = {}
        dist[sa] = 0
        pq = [(0.0, sa)]
        bx, by = sb
        while pq:
            dcur, c = heapq.heappop(pq)
            if c == sb:
                break
            if dcur > dist[c]:
                continue
            for dx, dy, w in moves:
                n = (c[0] + dx, c[1] + dy)
                if not (0 <= n[0] < self.nx and 0 <= n[1] < self.ny) or not free[n]:
                    continue
                if abs(dx) == 2 or abs(dy) == 2:   # knight move: middle cells free
                    m1 = (c[0] + math.floor(dx / 2), c[1] + math.floor(dy / 2))
                    m2 = (c[0] + math.ceil(dx / 2), c[1] + math.ceil(dy / 2))
                    if not (free[m1] and free[m2]):
                        continue
                nd = dcur + w
                if nd < dist[n]:
                    dist[n] = nd
                    prev[n] = c
                    heapq.heappush(pq, (nd, n))
        if not np.isfinite(dist[sb]):
            return None
        cells = [sb]
        while cells[-1] != sa:
            cells.append(prev[cells[-1]])
        cells.reverse()
        pts = [self.xy(c) for c in cells]
        # string pulling
        out = [pts[0]]
        i = 0
        while i < len(pts) - 1:
            j = len(pts) - 1
            while j > i + 1 and not self.line_ok(pts[i], pts[j]):
                j -= 1
            out.append(pts[j])
            i = j
        path = [tuple(a)] + out + [tuple(b)]
        return path


# ----------------------------------------------------------------- kinematics
def speed(d):
    return np.hypot(d.vx.values, d.vy.values)


def _diffs(P, dt):
    v = (P[2:] - P[:-2]) / (2 * dt)
    acc = (P[2:] - 2 * P[1:-1] + P[:-2]) / dt ** 2
    return P[1:-1], v, acc


def _fit_window(P, dt):
    """Least squares fit of x'' = a (g - x) - b x' on a window (shared a, b)."""
    x, v, acc = _diffs(P, dt)
    m = len(x)
    A = np.zeros((2 * m, 4))
    r = np.concatenate([acc[:, 0], acc[:, 1]])
    A[:m, 0], A[:m, 1], A[:m, 2] = -x[:, 0], -v[:, 0], 1
    A[m:, 0], A[m:, 1], A[m:, 3] = -x[:, 1], -v[:, 1], 1
    sol = np.linalg.lstsq(A, r, rcond=None)[0]
    return sol, float(np.sqrt(np.mean((A @ sol - r) ** 2)))


def robot_dynamics(P, dt):
    """Identify the robot's own waypoint controller x'' = a (g - x) - b x'
    from the clean (unperturbed, unsaturated) parts of its log.  Robots
    brake differently, so a and b are identified per robot."""
    ps = np.hypot(*np.diff(P, axis=0).T) / dt
    ab = []
    for i in range(0, len(P) - 12, 2):
        if ps[i:i + 12].min() < 0.03:
            continue
        sol, rms = _fit_window(P[i:i + 12], dt)
        if rms < 1e-3 and sol[0] > 0.8 and sol[1] > 0:
            ab.append(sol[:2])
    if len(ab) < 5:
        return 2.056, 4.444
    ab = np.array(ab)
    return float(np.median(ab[:, 0])), float(np.median(ab[:, 1]))


def predicted_stop(d, walls, radius):
    """Expected rest position of a robot and its STATIC/MOVING label.

    The robots drive to waypoints with a second-order controller
    x'' = a (g - x) - b x' (overdamped, so they come to rest at g without
    overshoot).  For a robot that is slowing down at the end of its log the
    waypoint g it is braking towards is recovered from its final samples
    using its own identified a, b; that is where it is expected to stop.
    Positions are used instead of the noisy velocity readings.  Robots that
    are still are STATIC; robots that are not slowing down (accelerating,
    cruising or being deflected) are judged at their last logged position."""
    P = d[["x", "y"]].values
    dt = float(np.median(np.diff(d.time_s.values)))
    ps = np.hypot(*np.diff(P, axis=0).T) / dt
    last = P[-1].copy()
    if ps[-1] < STILL_SPEED:
        return last, "STATIC", 0.0
    slowing = bool(np.all(np.diff(ps[-4:]) < 0))
    if not slowing:
        return last, "MOVING", 0.0
    a, b = robot_dynamics(P[:-20], dt)
    x, v, acc = _diffs(P[-10:], dt)
    goals = x + (acc + b * v) / a
    g = np.median(goals[-6:], axis=0)
    spread = float(np.max(np.linalg.norm(goals[-6:] - g, axis=1)))
    if spread > 0.05:
        # not a clean approach: fall back to exponential braking v/k
        k = (b - math.sqrt(max(b * b - 4 * a, 0))) / 2
        u = (P[-1] - P[-2]) / np.linalg.norm(P[-1] - P[-2])
        g = last + u * ps[-1] / k
    rem = float(np.linalg.norm(g - last))
    if rem > 1e-9:
        u = (g - last) / rem
        n = max(2, int(rem / 0.005) + 1)
        ss = np.linspace(0, rem, n)
        pts = last[None, :] + ss[:, None] * u[None, :]
        hit = np.nonzero(seg_dist(pts[:, 0], pts[:, 1], walls) < radius)[0]
        if len(hit) and hit[0] > 0:      # cannot drive through a wall
            rem = float(ss[hit[0] - 1])
        g = last + u * rem
    return g, "MOVING", rem


# ------------------------------------------------------------------ conflicts
def detect_conflicts(agents, radius, horizon=3.0, min_speed=0.05):
    """Conflicts between robot pairs.

    An encounter is a contiguous period in which two robots are within
    sensor range (0.5 m).  It is a conflict when, at some step of the
    encounter, both robots are moving and keeping their current straight
    trajectories would bring their centres closer than two radii (bodies
    touching) within a short horizon, i.e. an imminent collision course.
    Robots travelling in the same direction (convoys, cooperating pairs) are
    not conflicts.  Opposite headings (> 150 deg apart) are "frontal",
    any other crossing direction is "squeeze".  The reported step is the
    step of minimum separation in the encounter."""
    ids = sorted(agents)
    out = []
    for ia in range(len(ids)):
        for ib in range(ia + 1, len(ids)):
            a, b = ids[ia], ids[ib]
            da, db = agents[a], agents[b]
            n = min(len(da), len(db))
            pa = da[["x", "y"]].values[:n]
            pb = db[["x", "y"]].values[:n]
            dt = float(np.median(np.diff(da.time_s.values)))
            va = np.gradient(pa, axis=0) / dt
            vb = np.gradient(pb, axis=0) / dt
            dist = np.linalg.norm(pa - pb, axis=1)
            inr = dist <= SENSOR
            k = 0
            while k < n:
                if not inr[k]:
                    k += 1
                    continue
                e = k
                while e + 1 < n and inr[e + 1]:
                    e += 1
                best = None
                for s in range(k, e + 1):
                    sa, sb = np.linalg.norm(va[s]), np.linalg.norm(vb[s])
                    if sa < min_speed or sb < min_speed:
                        continue
                    ang = math.degrees(math.acos(np.clip(va[s] @ vb[s] / (sa * sb), -1, 1)))
                    if ang < 45:
                        continue
                    r = pb[s] - pa[s]
                    vr = vb[s] - va[s]
                    tc = -(r @ vr) / (vr @ vr)
                    if tc <= 0 or tc > horizon:
                        continue
                    miss = np.linalg.norm(r + vr * tc)
                    if miss < 2 * radius and (best is None or miss < best[0]):
                        best = (miss, ang, s)
                if best is not None:
                    kmin = k + int(np.argmin(dist[k:e + 1]))
                    out.append(dict(kind="frontal" if best[1] > 150 else "squeeze",
                                    step=int(da.step.iloc[kmin]), a=a, b=b, angle=round(best[1], 1),
                                    miss=round(float(best[0]), 3), dmin=round(float(dist[kmin]), 3),
                                    window=(int(k), int(e))))
                k = e + 1
    out.sort(key=lambda c: (c["step"], c["a"], c["b"]))
    return out


# ------------------------------------------------------------------- assess
def assess(folder, verbose=False):
    name, env, agents = load_cave(folder)
    radius = env["agent_radius"]
    walls = env["scenario"]
    hist = object_history(agents)

    holder, picks = None, []
    for g in hist:
        if g["kind"] == "pick":
            if holder is None:
                holder = g["agents"]
                picks.append(g)
        elif holder is not None and g["agents"] == holder:
            holder = None

    res = dict(name=name)
    found_in = None
    handled = []
    if picks:
        g = picks[0]
        pos = np.mean([agents[a][["x", "y"]].values[i] for a, i in zip(g["agents"], g["idx"])], axis=0)
        found_in = [g["agents"][0], round(float(pos[0]), 4), round(float(pos[1]), 4)]
        for g in picks:
            handled.extend(g["agents"])

    obstructed, cases, info = None, None, {}
    if holder is None:
        res.update(outcome="NO_RESCUE", collected=False, collected_by=None, position=None)
        coop = False
    else:
        coop = len(holder) == 2
        end = np.mean([agents[a][["x", "y"]].values[-1] for a in holder], axis=0)
        start = agents[holder[0]][["x", "y"]].values[0]
        planner = Planner(walls)
        path = planner.shortest_path(tuple(start), tuple(end))
        obstructed, cases = [], []
        for aid, d in agents.items():
            if aid in holder:
                continue
            p, case, rem = predicted_stop(d, walls, radius)
            dd = point_poly_dist(tuple(p), path)
            info[aid] = dict(judged=np.round(p, 3).tolist(), case=case, rem=rem, dist=round(dd, 3),
                             last=np.round(d[["x", "y"]].values[-1], 3).tolist(),
                             dist_last=round(point_poly_dist(tuple(d[["x", "y"]].values[-1]), path), 3))
            if dd < CLEARANCE + radius:
                obstructed.append(aid)
                cases.append(case)
        info["path"] = [tuple(round(v, 3) for v in q) for q in path]
        if obstructed:
            res["outcome"] = "RESCUE_HARD"
        else:
            res["outcome"] = "RESCUE"
            obstructed, cases = None, None
        res.update(collected=True, collected_by=holder,
                   position=[round(float(end[0]), 4), round(float(end[1]), 4)])
    res.update(obstructed_by=obstructed, obstruction_case=cases, cooperation_flag=coop)
    conf = detect_conflicts(agents, radius)
    res["conflicts"] = [[c["kind"], c["step"], c["a"], c["b"]] for c in conf]
    res["found_in"] = found_in
    res["handled_object"] = handled
    if verbose:
        print(name, json.dumps(res))
        print("   history:", [(g["step"], g["kind"], g["agents"]) for g in hist])
        for k, v in info.items():
            print("   ", k, v)
        for c in conf:
            print("   conflict", c)
    return res


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "/app/caves"
    verbose = "-v" in sys.argv
    folders = sorted(glob.glob(os.path.join(root, "cave*")), key=lambda p: int(''.join(ch for ch in os.path.basename(p) if ch.isdigit()) or 0))
    for f in folders:
        if not os.path.isdir(f):
            continue
        res = assess(f, verbose)
        with open(os.path.join(root, f"{res['name']}_outcome.json"), "w") as fh:
            json.dump(res, fh, indent=4)


if __name__ == "__main__":
    main()
