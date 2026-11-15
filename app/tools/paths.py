from geom import *
from carry import *
import pickle
res={}
for c in range(1,10):
    env,W,ags=load(c)
    car=[a for a,d in enumerate(ags) if carrying_at_end(d)]
    if not car: print(c,'none'); continue
    s=ags[min(car)][['x','y']].values[0]
    g=np.mean([ags[a][['x','y']].values[-1] for a in car],axis=0)
    pl=Planner(W,0.35)
    path,info=pl.plan(s,g)
    res[c]=(car,s,g,path)
    print(c,car,'goal',g.round(4),info)
    for a,d in enumerate(ags):
        if a in car: continue
        p=d[['x','y']].values[-1]
        print('   a%d last dist to path %.3f'%(a,pt_poly_dist(p,path)))
pickle.dump(res,open('/tmp/claude-0/-home-user-TB/b809c2e4-fa8c-5a7f-90e6-849eeb971db4/scratchpad/paths.pkl','wb'))
