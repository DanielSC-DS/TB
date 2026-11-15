from geom import *
def ang(u,v):
    nu,nv=np.linalg.norm(u),np.linalg.norm(v)
    if nu<1e-6 or nv<1e-6: return np.nan
    return np.degrees(np.arccos(np.clip(np.dot(u,v)/nu/nv,-1,1)))
def series(c,i,j,s0,s1,stride=2):
    env,W,ags=load(c)
    a=ags[i].set_index('step'); b=ags[j].set_index('step')
    st=a.index.intersection(b.index)
    pa=a.loc[st,['x','y']].values; pb=b.loc[st,['x','y']].values
    va=np.gradient(pa,axis=0)/DT; vb=np.gradient(pb,axis=0)/DT
    rows=[]
    for k,s in enumerate(st):
        if s<s0 or s>s1: continue
        dp=pb[k]-pa[k]; dv=vb[k]-va[k]
        tc=-np.dot(dp,dv)/max(np.dot(dv,dv),1e-9); cpa=np.linalg.norm(dp+max(tc,0)*dv)
        rows.append((s,np.linalg.norm(dp),np.linalg.norm(va[k]),np.linalg.norm(vb[k]),ang(va[k],vb[k]),cpa,tc))
    return rows
if __name__=='__main__':
    import sys
    c,i,j,s0,s1=map(int,sys.argv[1:6])
    for r in series(c,i,j,s0,s1)[::2]:
        print('%d d=%.3f va=%.2f vb=%.2f ang=%.0f cpa=%.3f tc=%.2f'%r)
