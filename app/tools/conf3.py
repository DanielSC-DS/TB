from geom import *
from conf2 import ang
from carry import gripper_events
def coop_pairs(ags):
    # pairs picking at same step (cooperative transport)
    picks={}
    for a,d in enumerate(ags):
        for e in gripper_events(d):
            if e[0] in('pick','drop'): picks.setdefault((e[0],e[1]),[]).append(a)
    return {tuple(v) for v in picks.values() if len(v)==2}
def episodes(c,R=0.5):
    env,W,ags=load(c); out=[]
    for i in range(len(ags)):
        for j in range(i+1,len(ags)):
            a=ags[i].set_index('step'); b=ags[j].set_index('step')
            st=a.index.intersection(b.index)
            pa=a.loc[st,['x','y']].values; pb=b.loc[st,['x','y']].values
            va=np.gradient(pa,axis=0)/DT; vb=np.gradient(pb,axis=0)/DT
            dist=np.linalg.norm(pa-pb,axis=1)
            k=0
            while k<len(st):
                if dist[k]<R:
                    e=k
                    while e+1<len(st) and dist[e+1]<R: e+=1
                    m=k+int(np.argmin(dist[k:e+1]))
                    best=None
                    for t in range(max(k-1,0),e+1):
                        dp=pb[t]-pa[t]; dv=vb[t]-va[t]
                        tc=-np.dot(dp,dv)/max(np.dot(dv,dv),1e-9); cpa=np.linalg.norm(dp+max(tc,0)*dv)
                        if best is None: best=(st[t],cpa,tc,np.linalg.norm(va[t]),np.linalg.norm(vb[t]),ang(va[t],vb[t]),dist[t])
                        break
                    out.append(dict(i=i,j=j,s=int(st[k]),e=int(st[e]),m=int(st[m]),dmin=dist[m],entry=best))
                    k=e+1
                else: k+=1
    return out,coop_pairs(ags)
if __name__=='__main__':
    for c in range(1,10):
        eps,cp=episodes(c)
        print('== cave',c,'coop',cp)
        for e in eps:
            s,cpa,tc,sa,sb,an,d0=e['entry']
            print(f"  {e['i']}-{e['j']} {e['s']}-{e['e']} closest@{e['m']} dmin={e['dmin']:.3f} | entry d={d0:.3f} cpa={cpa:.3f} tc={tc:.2f} va={sa:.2f} vb={sb:.2f} ang={an:.0f}")
