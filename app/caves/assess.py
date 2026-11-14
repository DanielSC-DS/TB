import json, glob, os, heapq, zlib, struct
import numpy as np, pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
RES = 0.01
CLR = 0.35


def load(c):
    env = json.load(open(f'{BASE}/cave{c}/cave{c}_env.json'))
    ags = {}
    for f in glob.glob(f'{BASE}/cave{c}/cave{c}_agent_*.csv'):
        i = int(f.rsplit('_', 1)[1][:-4])
        ags[i] = pd.read_csv(f)
    return env, dict(sorted(ags.items()))


def seg_dist(px, py, segs):
    """min distance from points (arrays) to list of segments"""
    d = np.full(px.shape, np.inf)
    for (a, b) in segs:
        ax, ay = a; bx, by = b
        dx, dy = bx - ax, by - ay
        L2 = dx * dx + dy * dy
        t = np.clip(((px - ax) * dx + (py - ay) * dy) / L2, 0, 1) if L2 > 0 else 0
        qx, qy = ax + t * dx, ay + t * dy
        d = np.minimum(d, np.hypot(px - qx, py - qy))
    return d


class Grid:
    def __init__(self, segs, res=RES):
        pts = np.array([p for s in segs for p in s])
        self.x0, self.y0 = pts.min(0) - 0.05
        x1, y1 = pts.max(0) + 0.05
        self.res = res
        self.nx = int(np.ceil((x1 - self.x0) / res)) + 1
        self.ny = int(np.ceil((y1 - self.y0) / res)) + 1
        xs = self.x0 + np.arange(self.nx) * res
        ys = self.y0 + np.arange(self.ny) * res
        self.X, self.Y = np.meshgrid(xs, ys, indexing='ij')
        self.D = seg_dist(self.X, self.Y, segs)
        self.segs = segs

    def idx(self, p):
        return (int(round((p[0] - self.x0) / self.res)), int(round((p[1] - self.y0) / self.res)))

    def pt(self, ij):
        return (self.x0 + ij[0] * self.res, self.y0 + ij[1] * self.res)


def inside_component(grid, free, start):
    """flood fill from start over free cells -> bool mask"""
    from collections import deque
    m = np.zeros_like(free)
    s = grid.idx(start)
    if not free[s]:
        return m
    q = deque([s]); m[s] = True
    while q:
        i, j = q.popleft()
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a, b = i + di, j + dj
            if 0 <= a < grid.nx and 0 <= b < grid.ny and free[a, b] and not m[a, b]:
                m[a, b] = True; q.append((a, b))
    return m


NB = []
for di in range(-2, 3):
    for dj in range(-2, 3):
        if (di, dj) != (0, 0) and np.gcd(abs(di), abs(dj)) == 1:
            NB.append((di, dj, float(np.hypot(di, dj))))


def dijkstra(grid, free, s, g):
    nx, ny = grid.nx, grid.ny
    dist = np.full((nx, ny), np.inf); prev = {}
    dist[s] = 0
    h = [(0.0, s)]
    while h:
        d, u = heapq.heappop(h)
        if u == g: break
        if d > dist[u]: continue
        i, j = u
        for di, dj, w in NB:
            a, b = i + di, j + dj
            if 0 <= a < nx and 0 <= b < ny and free[a, b]:
                if abs(di) == 2 or abs(dj) == 2:
                    if not free[i + (di > 0) - (di < 0), j + (dj > 0) - (dj < 0)]:
                        continue
                nd = d + w
                if nd < dist[a, b]:
                    dist[a, b] = nd; prev[(a, b)] = u
                    heapq.heappush(h, (nd, (a, b)))
    if not np.isfinite(dist[g]):
        return None
    path = [g]
    while path[-1] != s:
        path.append(prev[path[-1]])
    return path[::-1]


def los(p, q, segs, clr):
    """segment p-q keeps distance >= clr from all walls"""
    n = max(2, int(np.hypot(q[0] - p[0], q[1] - p[1]) / 0.005) + 1)
    t = np.linspace(0, 1, n)
    xs = p[0] + t * (q[0] - p[0]); ys = p[1] + t * (q[1] - p[1])
    return seg_dist(xs, ys, segs).min() >= clr - 1e-9


def smooth(pts, segs, clr):
    out = [pts[0]]; i = 0
    while i < len(pts) - 1:
        j = len(pts) - 1
        while j > i + 1 and not los(pts[i], pts[j], segs, clr):
            j -= 1
        out.append(pts[j]); i = j
    return out


def plan(grid, start, goal, clr=CLR):
    free = grid.D >= clr
    s, g = grid.idx(start), grid.idx(goal)
    info = dict(start_clr=float(seg_dist(np.array([start[0]]), np.array([start[1]]), grid.segs)[0]),
                goal_clr=float(seg_dist(np.array([goal[0]]), np.array([goal[1]]), grid.segs)[0]))
    f2 = free.copy(); f2[s] = True; f2[g] = True
    path = dijkstra(grid, f2, s, g)
    if path is None:
        return None, info
    pts = [tuple(start)] + [grid.pt(p) for p in path[1:-1]] + [tuple(goal)]
    sp = smooth(pts, grid.segs, clr - 0.006)
    return sp, info


def point_poly_dist(p, poly):
    best = np.inf
    for a, b in zip(poly[:-1], poly[1:]):
        best = min(best, float(seg_dist(np.array([p[0]]), np.array([p[1]]), [(a, b)])[0]))
    return best


def path_len(poly):
    return sum(np.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(poly[:-1], poly[1:]))


# ---------- tiny PNG writer for visual checks ----------
def write_png(fn, img):
    h, w, _ = img.shape
    raw = b''.join(b'\x00' + img[r].astype(np.uint8).tobytes() for r in range(h))
    def chunk(t, d):
        c = struct.pack('>I', len(d)) + t + d
        return c + struct.pack('>I', zlib.crc32(t + d) & 0xffffffff)
    open(fn, 'wb').write(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0)) +
                         chunk(b'IDAT', zlib.compress(raw, 6)) + chunk(b'IEND', b''))
PI = np.pi


def episodes(d):
    """gripper-closing episodes with classification"""
    g = (d.open.values > 0.02); n = len(d)
    phi = d.phi.values; op = d.open.values; cu = d.curr.values
    eps = []; i = 0
    while i < n:
        if not g[i]:
            i += 1; continue
        j = i
        while j < n and g[j]: j += 1
        seg = slice(i, j)
        mo = op[seg].max()
        plateau = np.where(op[seg] > mo - 0.01)[0] + i
        pc = float(np.median(cu[plateau]))
        phi_close = phi[i]
        phi_rel = phi[min(j, n - 1)] if j < n else phi[n - 1]
        # the phi when gripper actually releases (open starts decreasing)
        k = plateau[-1]
        phi_release = phi[min(j, n - 1)]
        held = (pc > 2.0) and (mo < 0.95)
        e = dict(start=i, end=j, open=float(mo), curr=pc, phi_close=float(phi_close),
                 phi_release=float(phi[k]), phi_after=float(phi_rel), held=held,
                 ended=(j >= n))
        # extension start (arm leaving rest) before close
        k0 = i
        while k0 > 0 and phi[k0 - 1] > 0.03 and phi[k0 - 1] <= phi[k0] + 1e-9 and abs(phi[k0-1]-0.02) > 1e-4:
            k0 -= 1
        e['ext_start'] = k0
        e['phimax'] = float(phi[i:j].max())
        if phi_close > 3.1 and held and abs(e['phi_after'] - 0.02) < 0.005 and phi[k] < 0.05:
            e['kind'] = 'pickup'
        elif phi_close < 0.05 and held and phi[k] > 3.1:
            e['kind'] = 'drop'
        elif phi_close > 3.1 and held:
            e['kind'] = 'failed_pick'   # gripped object but lost it
        else:
            e['kind'] = 'other'
        eps.append(e)
        i = j
    return eps


def carrying_state(d):
    eps = episodes(d)
    carrying = False; hist = []
    for e in eps:
        if e['kind'] == 'pickup':
            carrying = True
        elif e['kind'] == 'drop':
            carrying = False
        hist.append(e)
    # must also be resting at end
    if carrying and abs(d.phi.iloc[-1] - 0.02) > 0.005:
        carrying = False
    return carrying, eps


# ---------- braking model ----------
DT = 0.05
P = 0.82
def fitcmd(v, W=None):
    """v: position-derived speeds (trailing braking window). Reconstruct cmd, fit models, simulate."""
    cmd=(v[1:]-P*v[:-1])/(1-P)
    n=np.arange(len(cmd),dtype=float)
    qs=np.linspace(0.0,0.97,98)
    best={}
    # exp: A r^n + B q^n
    b=(np.inf,)
    for r in np.linspace(0.95,0.999,491):
        for q in qs:
            if q>=r: continue
            A=np.vstack([r**n,q**n]).T
            c,*_=np.linalg.lstsq(A,cmd,rcond=None); e=np.sum((A@c-cmd)**2)
            if e<b[0]: b=(e,r,q,c)
    best['exp']=b
    b=(np.inf,)
    for q in qs:
        A=np.vstack([np.ones_like(n),n,q**n]).T
        c,*_=np.linalg.lstsq(A,cmd,rcond=None); e=np.sum((A@c-cmd)**2)
        if e<b[0]: b=(e,q,c)
    best['lin']=b
    N=len(cmd); vv=v[-1]
    # simulate exp
    e,r,q,c=best['exp']; d=0; x=vv
    for m in range(N,N+40000):
        cm=c[0]*r**m+c[1]*q**m
        x=P*x+(1-P)*cm; 
        if x<1e-6: break
        d+=x*DT
    de=d
    e2,q2,c2=best['lin']; d=0; x=vv
    for m in range(N,N+40000):
        cm=max(c2[0]+c2[1]*m+c2[2]*q2**m,0)
        x=P*x+(1-P)*cm
        if x<1e-6: break
        d+=x*DT
    dl=d
    return dict(sse_exp=best['exp'][0], r=best['exp'][1], q=best['exp'][2], sse_lin=best['lin'][0], a=best['lin'][2][1], ql=best['lin'][1], de=de, dl=dl, cmd=cmd)


RAD_DEFAULT = 0.075
P_FILT = 0.82


def trailing_brake_start(ps):
    i = len(ps) - 1
    while i > 0 and ps[i - 1] >= ps[i] - 1e-4:
        i -= 1
    return i


def end_state(d, segs, radius):
    """STATIC / MOVING and the position used to judge obstruction."""
    X = d[['x', 'y']].values
    V = np.diff(X, axis=0) / DT
    ps = np.hypot(*V.T)
    last = X[-1]
    if ps[-1] < 0.02:
        return 'STATIC', last, 'still'
    C = (V[1:] - P * V[:-1]) / (1 - P)          # reconstructed commanded velocity
    jumps = np.hypot(*np.diff(C[-8:], axis=0).T)
    i = trailing_brake_start(ps)
    s = ps[max(i - 1, 0):]
    if jumps.max() > 0.25 or len(s) < 6:
        # robot is manoeuvring (evasive kick) - its stop point cannot be inferred
        return 'MOVING', last, 'manoeuvring'
    f = fitcmd(s)
    u = C[-1] / (np.linalg.norm(C[-1]) + 1e-12)
    dist = f['de'] if f['sse_exp'] <= f['sse_lin'] else f['dl']
    # walk forward, stopping at a wall
    stop = last.copy()
    for t in np.arange(0, dist, 0.005):
        p = last + u * t
        if seg_dist(np.array([p[0]]), np.array([p[1]]), segs)[0] < radius:
            break
        stop = p
    return 'MOVING', stop, 'braking %.3f m' % dist


def analyse(c):
    env, ags = load(c)
    segs = env['scenario']; radius = env.get('agent_radius', RAD_DEFAULT)
    info = {k: carrying_state(d) for k, d in ags.items()}
    carriers = sorted(k for k, (car, _) in info.items() if car)
    res = dict(name=f'cave{c}')
    # ---- found / handled
    finds = []
    handled = []
    for k, (_, eps) in info.items():
        for e in eps:
            if e['held'] and e['phimax'] > 3.1 and e['phi_close'] > 3.1:
                finds.append((e['ext_start'], k))
        kinds = [e['kind'] for e in eps]
        for a in range(len(kinds)):
            if kinds[a] == 'pickup' and 'drop' in kinds[a + 1:]:
                handled.append(k); break
    found_in = None
    if finds:
        t0 = min(f[0] for f in finds)
        who = sorted(k for t, k in finds if t == t0)
        pts = [ags[k][['x', 'y']].values[t0] for k in who]
        # simultaneous finders that both held the object = cooperative find
        p = np.mean(pts, 0)
        found_in = [who[0], round(float(p[0]), 4), round(float(p[1]), 4)]
    if not carriers:
        res.update(outcome='NO_RESCUE', collected=False, collected_by=None, position=None,
                   obstructed_by=None, obstruction_case=None, cooperation_flag=False)
    else:
        start = ags[carriers[0]][['x', 'y']].values[0]
        goal = np.mean([ags[k][['x', 'y']].values[-1] for k in carriers], 0)
        grid = Grid(segs)
        path, _ = plan(grid, tuple(start), tuple(goal))
        obs = []
        for k, d in ags.items():
            if k in carriers:
                continue
            st, pos, why = end_state(d, segs, radius)
            dd = point_poly_dist(pos, path)
            if dd < CLR + radius:
                obs.append((k, st))
        res.update(outcome='RESCUE_HARD' if obs else 'RESCUE', collected=True, collected_by=carriers,
                   position=[round(float(goal[0]), 4), round(float(goal[1]), 4)],
                   obstructed_by=[k for k, _ in obs] or None,
                   obstruction_case=[s for _, s in obs] or None,
                   cooperation_flag=len(carriers) == 2)
    res['conflicts'] = conflicts(ags, radius)
    res['found_in'] = found_in
    res['handled_object'] = sorted(handled)
    return res


def conflicts(ags, radius, R=0.5, vmin=0.05, tmax=3.0):
    ks = list(ags); out = []
    for a in range(len(ks)):
        for b in range(a + 1, len(ks)):
            i, j = ks[a], ks[b]; A, B = ags[i], ags[j]
            n = min(len(A), len(B))
            Pp = A[['x', 'y']].values[:n]; Q = B[['x', 'y']].values[:n]
            dist = np.hypot(*(Pp - Q).T)
            inr = dist < R; t = 0
            while t < n:
                if not inr[t]:
                    t += 1; continue
                s = t
                while t < n and inr[t]:
                    t += 1
                tm = s + int(np.argmin(dist[s:t]))
                best = None
                for k in range(max(s, 5), tm + 1):
                    u = (Pp[k] - Pp[k - 1]) / DT; w = (Q[k] - Q[k - 1]) / DT
                    if np.linalg.norm(u) < vmin or np.linalg.norm(w) < vmin:
                        continue
                    dp = Pp[k] - Q[k]; dv = u - w
                    tc = -np.dot(dp, dv) / (np.dot(dv, dv) + 1e-12)
                    if tc < 0 or tc > tmax:
                        continue
                    cpa = np.linalg.norm(dp + dv * tc)
                    ang = np.degrees(np.arccos(np.clip(np.dot(u, w) / (np.linalg.norm(u) * np.linalg.norm(w)), -1, 1)))
                    if best is None or cpa < best[0]:
                        best = (cpa, ang)
                if best and best[0] < 2 * radius and best[1] >= 45:
                    out.append(['frontal' if best[1] >= 150 else 'squeeze', int(tm), i, j])
    return sorted(out, key=lambda x: x[1])


if __name__ == '__main__':
    import sys
    outdirs = [BASE] + [p for p in ['/app/caves'] if os.path.isdir(p)]
    for f in sorted(glob.glob(f'{BASE}/cave*/cave*_env.json')):
        c = int(os.path.basename(f).split('_')[0][4:])
        r = analyse(c)
        print(json.dumps(r))
        for od in outdirs:
            json.dump(r, open(f'{od}/cave{c}_outcome.json', 'w'), indent=4)
