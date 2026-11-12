import pandas as pd, numpy as np, json, glob
B='/app/caves'
def load(c):
    env=json.load(open(f'{B}/cave{c}/cave{c}_env.json'))
    D=[pd.read_csv(f'{B}/cave{c}/cave{c}_agent_{a}.csv') for a in range(env['num_agents'])]
    return env,D
def carry_events(d):
    """returns list of (step, event, phi) ; events: grip (holding) / release"""
    hold=((d.curr>2.0)&(d.open>0.1)&(d.open<0.9)).values
    ev=[];state=False
    for i in range(len(d)):
        if hold[i] and not state: ev.append((int(d.step[i]),'grip',round(float(d.phi[i]),2))); state=True
        elif state and d.open[i]<0.05: ev.append((int(d.step[i]),'release',round(float(d.phi[i]),2))); state=False
    return ev,state
def carrying(d):
    ev,state=carry_events(d)
    if state: return True,ev   # still gripping
    # object on robot: last release happened with arm retracted
    if ev and ev[-1][1]=='release' and ev[-1][2]<1.0: return True,ev
    return False,ev
def segdist(P,a,b):
    P=np.atleast_2d(P); a=np.asarray(a,float); b=np.asarray(b,float)
    ab=b-a; t=np.clip(((P-a)@ab)/(ab@ab),0,1)
    return np.hypot(*(P-(a+t[:,None]*ab)).T)
def walldist(env,P):
    return np.min([segdist(P,s[0],s[1]) for s in env['scenario']],axis=0)
def carry_events2(d):
    hold=((d.curr>2.0)&(d.open>0.1)&(d.open<0.9)).values
    ev=[];state=False
    for i in range(len(d)):
        if hold[i] and not state: ev.append(['grip',i,float(d.phi[i]),float(d.open[i:i+20].max())]); state=True
        elif state and d.open[i]<0.05: ev.append(['release',i,float(d.phi[i])]); state=False
    return ev,state
def carrying2(d):
    ev,state=carry_events2(d)
    on=False; acts=[]
    k=0
    while k<len(ev):
        g=ev[k]; r=ev[k+1] if k+1<len(ev) else None
        if g[2]>3.0 and (r is None or r[2]<0.1): on=True; acts.append(('pickup',g[1], r[1] if r else None))
        elif g[2]<1.0 and (r is None or r[2]>3.0): on=False if r else on; acts.append(('putdown',g[1], r[1] if r else None))
        else: acts.append(('invalid',g[1],r[1] if r else None,round(g[2],2),round(r[2],2) if r else None,round(g[3],3)))
        k+=2
    return on,acts
DT=0.05
def stop_pred(d,k=8):
    ps=d[['x','y']].values
    dp=np.diff(ps,axis=0)[-k:]; fs=np.hypot(*dp.T)/DT
    v=fs[-1]
    if v<0.02: return ps[-1],'STATIC',v,None
    r=fs[1:]/np.maximum(fs[:-1],1e-9)
    lam=np.exp(np.mean(np.log(r[-5:])))
    if lam<0.995 and np.all(np.diff(fs[-4:])<0):
        dist=v*DT*lam/(1-lam)
    else: dist=0.0
    u=dp[-1]/np.hypot(*dp[-1])
    return ps[-1]+u*dist,'MOVING',v,(round(lam,4),round(dist,3))
import numpy as np, heapq
CL=0.35
def seg_seg_dist(p,q,a,b):
    def cross(o,u,v): return (u[0]-o[0])*(v[1]-o[1])-(u[1]-o[1])*(v[0]-o[0])
    d1,d2,d3,d4=cross(a,b,p),cross(a,b,q),cross(p,q,a),cross(p,q,b)
    if ((d1>0)!=(d2>0)) and ((d3>0)!=(d4>0)): return 0.0
    return min(segdist(p,a,b)[0],segdist(q,a,b)[0],segdist(a,p,q)[0],segdist(b,p,q)[0])
def seg_clear(env,p,q,cl=CL):
    return all(seg_seg_dist(p,q,np.array(s[0]),np.array(s[1]))>=cl for s in env['scenario'])
def plan(env,start,goal,res=0.02,cl=CL):
    W=np.array(env['scenario']).reshape(-1,2); lo=W.min(0); hi=W.max(0)
    nx=int((hi[0]-lo[0])/res)+1; ny=int((hi[1]-lo[1])/res)+1
    X,Y=np.meshgrid(lo[0]+np.arange(nx)*res, lo[1]+np.arange(ny)*res, indexing='ij')
    P=np.c_[X.ravel(),Y.ravel()]
    free=(walldist(env,P)>=cl).reshape(nx,ny)
    idx=lambda p:(int(round((p[0]-lo[0])/res)),int(round((p[1]-lo[1])/res)))
    s=idx(start); g=idx(goal)
    def nearest_free(c):
        best=None
        for i in range(max(0,c[0]-5),min(nx,c[0]+6)):
            for j in range(max(0,c[1]-5),min(ny,c[1]+6)):
                if free[i,j]:
                    dd=(i-c[0])**2+(j-c[1])**2
                    if best is None or dd<best[0]: best=(dd,(i,j))
        return best[1]
    s=nearest_free(s); g=nearest_free(g)
    nb=[(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1),(1,2),(2,1),(-1,2),(-2,1),(1,-2),(2,-1),(-1,-2),(-2,-1)]
    dist={s:0};prev={};pq=[(0,s)]
    while pq:
        dcur,u=heapq.heappop(pq)
        if u==g: break
        if dcur>dist[u]: continue
        for di,dj in nb:
            v=(u[0]+di,u[1]+dj)
            if 0<=v[0]<nx and 0<=v[1]<ny and free[v]:
                if abs(di)+abs(dj)==3:
                    m=(u[0]+di//2 if abs(di)==2 else u[0], u[1]+dj//2 if abs(dj)==2 else u[1])
                    if not free[m] or not free[(u[0]+(1 if di>0 else -1), u[1]+(1 if dj>0 else -1))]: continue
                nd=dcur+np.hypot(di,dj)
                if nd<dist.get(v,1e18): dist[v]=nd; prev[v]=u; heapq.heappush(pq,(nd,v))
    if g not in dist: return None
    path=[g]
    while path[-1]!=s: path.append(prev[path[-1]])
    path=[lo+np.array(c)*res for c in path[::-1]]
    path=[np.array(start,float)]+path+[np.array(goal,float)]
    # string pulling
    out=[path[0]]; i=0
    while i<len(path)-1:
        j=len(path)-1
        while j>i+1 and not seg_clear(env,path[i],path[j],cl-1e-3): j-=1
        out.append(path[j]); i=j
    return np.array(out)
def path_len(p): return np.hypot(*np.diff(p,axis=0).T).sum()
def dist_to_path(path,P):
    return np.min([segdist(P,path[k],path[k+1]) for k in range(len(path)-1)],axis=0)
import itertools, os, sys

def vel(d, k=3):
    P = d[['x', 'y']].values; V = np.zeros_like(P)
    V[k:] = (P[k:] - P[:-k]) / (k * DT); return V

def conflicts(D, R=0.5, CPA_MAX=0.15, MIN_SPEED=0.1, HORIZON=2.5):
    """Pairs entering sensor range (0.5) on a collision course (predicted closest
    approach < 2 radii within HORIZON seconds), both moving. frontal: headings
    opposite (>150 deg); squeeze: other converging directions."""
    out = []
    for i, j in itertools.combinations(range(len(D)), 2):
        n = min(len(D[i]), len(D[j]))
        P = D[i][['x', 'y']].values[:n]; Q = D[j][['x', 'y']].values[:n]
        V = vel(D[i])[:n]; U = vel(D[j])[:n]
        dd = np.hypot(*(P - Q).T); idx = np.where(dd < R)[0]
        if not len(idx): continue
        for e in np.split(idx, np.where(np.diff(idx) > 1)[0] + 1):
            k = e[np.argmin(dd[e])]; best = None
            for s in e:
                if np.linalg.norm(V[s]) < MIN_SPEED or np.linalg.norm(U[s]) < MIN_SPEED: continue
                r = Q[s] - P[s]; w = U[s] - V[s]
                if r @ w >= 0: continue
                t = -(r @ w) / (w @ w)
                if t > HORIZON: continue
                md = np.linalg.norm(r + w * t)
                ang = np.degrees(np.arccos(np.clip(V[s] @ U[s] / np.linalg.norm(V[s]) / np.linalg.norm(U[s]), -1, 1)))
                if ang < 30: continue
                if best is None or md < best[0]: best = (md, ang)
            if best and best[0] < CPA_MAX:
                out.append(['frontal' if best[1] > 150 else 'squeeze', int(D[i].step[k]), i, j])
    return sorted(out, key=lambda x: x[1])

def wall_limited(env, p0, u, dist, r):
    n = max(1, int(dist / 0.005)); last = p0
    for k in range(1, n + 1):
        q = p0 + u * dist * k / n
        if walldist(env, q[None])[0] < r: break
        last = q
    return last

def expected_stop(env, d, r):
    """Last position, or projected stop for robots braking at log end
    (exponential speed decay -> geometric-series stopping distance)."""
    ps = d[['x', 'y']].values; dp = np.diff(ps, axis=0); fs = np.hypot(*dp.T) / DT
    v = fs[-1]
    if v < 0.02: return ps[-1], 'STATIC'
    lam = np.exp(np.mean(np.log(fs[-4:] / fs[-5:-1])))
    acc = -np.polyfit(np.arange(5) * DT, fs[-5:], 1)[0]
    if acc > 0.05 and np.all(np.diff(fs[-4:]) < 0) and lam < 1:
        u = dp[-1] / np.linalg.norm(dp[-1])
        return wall_limited(env, ps[-1], u, v * DT * lam / (1 - lam), r), 'MOVING'
    return ps[-1], 'MOVING'

def analyze(cdir):
    name = os.path.basename(cdir.rstrip('/'))
    env = json.load(open(f'{cdir}/{name}_env.json')); r = env['agent_radius']
    D = [pd.read_csv(f'{cdir}/{name}_agent_{a}.csv') for a in range(env['num_agents'])]
    cr = [a for a, d in enumerate(D) if carrying2(d)[0]]
    res = dict(name=name, outcome='NO_RESCUE', collected=bool(cr), collected_by=cr or None,
               position=None, obstructed_by=None, obstruction_case=None,
               cooperation_flag=len(cr) == 2, conflicts=conflicts(D))
    if cr:
        start = D[min(cr)][['x', 'y']].values[0]
        goal = np.mean([D[a][['x', 'y']].values[-1] for a in cr], axis=0)
        res['position'] = [round(float(goal[0]), 4), round(float(goal[1]), 4)]
        path = plan(env, start, goal, res=0.01)
        obs = []
        for a, d in enumerate(D):
            if a in cr: continue
            p, case = expected_stop(env, d, r)
            if dist_to_path(path, p[None])[0] < CL + r: obs.append((a, case))
        if obs:
            res['outcome'] = 'RESCUE_HARD'
            res['obstructed_by'] = [a for a, _ in obs]; res['obstruction_case'] = [c for _, c in obs]
        else:
            res['outcome'] = 'RESCUE'
    return res

if __name__ == '__main__':
    base = sys.argv[1] if len(sys.argv) > 1 else '/app/caves'
    for cdir in sorted(glob.glob(f'{base}/cave*/')):
        res = analyze(cdir)
        json.dump(res, open(f"{base}/{res['name']}_outcome.json", 'w'), indent=4)
        print(json.dumps(res))
