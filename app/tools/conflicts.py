from geom import *
from conf2 import ang
R_SENSE=0.5; CPA_MAX=0.15; TC_MAX=3.0; V_MIN=0.05
def conflicts(c, verbose=False):
    env,W,ags=load(c); r=env['agent_radius']; out=[]
    for i in range(len(ags)):
        for j in range(i+1,len(ags)):
            a=ags[i].set_index('step'); b=ags[j].set_index('step')
            st=a.index.intersection(b.index)
            pa=a.loc[st,['x','y']].values; pb=b.loc[st,['x','y']].values
            va=np.gradient(pa,axis=0)/DT; vb=np.gradient(pb,axis=0)/DT
            dist=np.linalg.norm(pa-pb,axis=1)
            k=0
            while k<len(st):
                if dist[k]<R_SENSE:
                    e=k
                    while e+1<len(st) and dist[e+1]<R_SENSE: e+=1
                    m=k+int(np.argmin(dist[k:e+1]))
                    hit=None
                    for t in range(k,m+1):
                        dp=pb[t]-pa[t]; dv=vb[t]-va[t]
                        if np.linalg.norm(va[t])<V_MIN or np.linalg.norm(vb[t])<V_MIN: continue
                        tc=-np.dot(dp,dv)/max(np.dot(dv,dv),1e-9)
                        if tc<=0 or tc>TC_MAX: continue
                        cpa=np.linalg.norm(dp+tc*dv)
                        if cpa<CPA_MAX:
                            hit=(int(st[t]),cpa,tc,ang(va[t],vb[t]),np.linalg.norm(va[t]),np.linalg.norm(vb[t])); break
                    if hit:
                        kind='frontal' if hit[3]>=150 else 'squeeze'
                        out.append([kind,int(st[m]),i,j])
                        if verbose: print(f"  c{c} {i}-{j} closest@{st[m]} dmin={dist[m]:.3f} hit@{hit[0]} cpa={hit[1]:.3f} tc={hit[2]:.2f} ang={hit[3]:.0f} v={hit[4]:.2f},{hit[5]:.2f} -> {kind}")
                    k=e+1
                else: k+=1
    return out
if __name__=='__main__':
    for c in range(1,10): print(c,conflicts(c,True))
