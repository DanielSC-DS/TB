from geom import *
def pair_episodes(c, R=0.5):
    env,W,ags=load(c); out=[]
    n=len(ags)
    for i in range(n):
        for j in range(i+1,n):
            a=ags[i].set_index('step'); b=ags[j].set_index('step')
            st=a.index.intersection(b.index)
            pa=a.loc[st,['x','y']].values; pb=b.loc[st,['x','y']].values
            va=np.gradient(pa,axis=0)/DT; vb=np.gradient(pb,axis=0)/DT
            va_r=a.loc[st,['vx','vy']].values; vb_r=b.loc[st,['vx','vy']].values
            dist=np.linalg.norm(pa-pb,axis=1)
            inr=dist<R
            k=0
            while k<len(st):
                if inr[k]:
                    e=k
                    while e+1<len(st) and inr[e+1]: e+=1
                    m=k+np.argmin(dist[k:e+1])
                    # collision-course metrics at entry and over the episode
                    info=[]
                    for t in range(k,e+1):
                        dp=pb[t]-pa[t]; dv=vb[t]-va[t]
                        tc=-np.dot(dp,dv)/max(np.dot(dv,dv),1e-9)
                        dmin=np.linalg.norm(dp+max(tc,0)*dv)
                        info.append((tc,dmin))
                    info=np.array(info)
                    def ang(u,v):
                        nu,nv=np.linalg.norm(u),np.linalg.norm(v)
                        if nu<0.02 or nv<0.02: return np.nan
                        return np.degrees(np.arccos(np.clip(np.dot(u,v)/nu/nv,-1,1)))
                    out.append(dict(i=i,j=j,start=int(st[k]),end=int(st[e]),closest=int(st[m]),dmin=dist[m],
                        ang_entry=ang(va[k],vb[k]),ang_closest=ang(va[m],vb[m]),
                        sp_entry=(np.linalg.norm(va[k]),np.linalg.norm(vb[k])),
                        min_cpa=info[:,1].min(), cpa_entry=info[0,1], tc_entry=info[0,0]))
                    k=e+1
                else: k+=1
    return out
