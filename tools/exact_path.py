import numpy as np, json, heapq
B='/home/user/TB/app/caves'
EPS=1e-7
def pseg(p,a,b):
    ab=b-a; t=np.clip(np.dot(p-a,ab)/np.dot(ab,ab),0,1); return np.hypot(*(p-(a+t*ab)))
def segseg(p,q,a,b):
    d1=q-p; d2=b-a
    def cr(u,v): return u[0]*v[1]-u[1]*v[0]
    den=cr(d1,d2)
    if abs(den)>1e-12:
        t=cr(a-p,d2)/den; u=cr(a-p,d1)/den
        if 0<=t<=1 and 0<=u<=1: return 0.0
    return min(pseg(p,a,b),pseg(q,a,b),pseg(a,p,q),pseg(b,p,q))
def exact_path(W,S,T,r):
    W=[(np.array(a,float),np.array(b,float)) for a,b in W]
    rr=r*(1+1e-9)
    verts=[]
    for a,b in W:
        for v in (a,b):
            if not any(np.allclose(v,u) for u in verts): verts.append(v)
    def segfree(p,q):
        return all(segseg(p,q,a,b)>=r-1e-6 for a,b in W)
    def ptfree(p): return all(pseg(p,a,b)>=r-1e-6 for a,b in W)
    # nodes: list of (pos, circle idx or -1, angle)
    nodes=[(np.array(S,float),-1,0),(np.array(T,float),-1,0)]
    edges={}
    def add_node(p,ci,ang):
        nodes.append((p,ci,ang)); return len(nodes)-1
    def addedge(i,j,w):
        edges.setdefault(i,[]).append((j,w)); edges.setdefault(j,[]).append((i,w))
    if segfree(nodes[0][0],nodes[1][0]): addedge(0,1,np.hypot(*(nodes[0][0]-nodes[1][0])))
    circ_pts={i:[] for i in range(len(verts))}
    # point-circle tangents
    for pi in (0,1):
        P=nodes[pi][0]
        for ci,C in enumerate(verts):
            d=np.hypot(*(P-C))
            if d<=rr: continue
            base=np.arctan2(*(P-C)[::-1]); al=np.arccos(rr/d)
            for s in (-1,1):
                ang=base+s*al; X=C+rr*np.array([np.cos(ang),np.sin(ang)])
                if ptfree(X) and segfree(P,X):
                    k=add_node(X,ci,ang); circ_pts[ci].append(k); addedge(pi,k,np.hypot(*(P-X)))
    # circle-circle tangents (equal radii): outer and inner
    for i in range(len(verts)):
        for j in range(i+1,len(verts)):
            C1,C2=verts[i],verts[j]; D=C2-C1; d=np.hypot(*D)
            if d<1e-9: continue
            th=np.arctan2(D[1],D[0])
            cands=[]
            for s in (-1,1):  # outer: normal offset
                n=np.array([np.cos(th+s*np.pi/2),np.sin(th+s*np.pi/2)])
                cands.append((C1+rr*n,C2+rr*n))
            if d>2*rr:
                al=np.arccos(2*rr/d)
                for s in (-1,1):
                    a1=th+s*al; n1=np.array([np.cos(a1),np.sin(a1)])
                    cands.append((C1+rr*n1,C2-rr*n1))
            for X,Y in cands:
                if ptfree(X) and ptfree(Y) and segfree(X,Y):
                    k1=add_node(X,i,np.arctan2(*(X-C1)[::-1])); k2=add_node(Y,j,np.arctan2(*(Y-C2)[::-1]))
                    circ_pts[i].append(k1); circ_pts[j].append(k2); addedge(k1,k2,np.hypot(*(X-Y)))
    # arcs
    for ci,ks in circ_pts.items():
        if len(ks)<2: continue
        C=verts[ci]
        ks=sorted(ks,key=lambda k:nodes[k][2]%(2*np.pi))
        for m in range(len(ks)):
            k1=ks[m]; k2=ks[(m+1)%len(ks)]
            a1=nodes[k1][2]%(2*np.pi); a2=nodes[k2][2]%(2*np.pi)
            da=(a2-a1)%(2*np.pi)
            ok=True
            for t in np.linspace(0,da,max(3,int(da/0.01))):
                X=C+rr*np.array([np.cos(a1+t),np.sin(a1+t)])
                if not ptfree(X): ok=False;break
            if ok: addedge(k1,k2,rr*da)
    dist={0:0};prev={};h=[(0,0)]
    while h:
        d,u=heapq.heappop(h)
        if u==1: break
        if d>dist[u]: continue
        for v,w in edges.get(u,[]):
            if d+w<dist.get(v,np.inf): dist[v]=d+w; prev[v]=u; heapq.heappush(h,(d+w,v))
    if 1 not in dist: return None,None
    seq=[1]
    while seq[-1]!=0: seq.append(prev[seq[-1]])
    seq=seq[::-1]
    # dense sample
    pts=[]
    for u,v in zip(seq[:-1],seq[1:]):
        pu,cu,au=nodes[u]; pv,cv,av=nodes[v]
        if cu>=0 and cu==cv and not np.allclose(pu,pv) and abs(np.hypot(*(pu-pv)) - 0)>0 and (u,v) and _isarc(edges,u,v,pu,pv):
            C=verts[cu]; a1=au%(2*np.pi); a2=av%(2*np.pi)
            da=(a2-a1)%(2*np.pi)
            if da>np.pi: da=da-2*np.pi
            for t in np.linspace(0,1,max(2,int(abs(da)*rr/0.002))):
                pts.append(C+rr*np.array([np.cos(a1+t*da),np.sin(a1+t*da)]))
        else:
            n=max(2,int(np.hypot(*(pu-pv))/0.002))
            for t in np.linspace(0,1,n): pts.append(pu+(pv-pu)*t)
    return np.array(pts),dist[1]
def _isarc(edges,u,v,pu,pv):
    for j,w in edges[u]:
        if j==v and abs(w-np.hypot(*(pu-pv)))>1e-9: return True
    return False
