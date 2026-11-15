import numpy as np, zlib, struct, pickle, sys
from geom import *
SP='/tmp/claude-0/-home-user-TB/b809c2e4-fa8c-5a7f-90e6-849eeb971db4/scratchpad/'
def png(fn, img):
    h,w,_=img.shape
    raw=b''.join(b'\x00'+img[y].tobytes() for y in range(h))
    def ch(t,d): return struct.pack('>I',len(d))+t+d+struct.pack('>I',zlib.crc32(t+d)&0xffffffff)
    open(fn,'wb').write(b'\x89PNG\r\n\x1a\n'+ch(b'IHDR',struct.pack('>IIBBBBB',w,h,8,2,0,0,0))+ch(b'IDAT',zlib.compress(raw))+ch(b'IEND',b''))
S=160
COL=[(220,0,0),(0,150,0),(0,0,230),(200,0,200),(0,180,180)]
def to_px(p): return int(p[0]*S), int((5-p[1])*S)
def dot(img,p,col,r=2):
    x,y=to_px(p)
    img[max(0,y-r):y+r+1,max(0,x-r):x+r+1]=col
def line(img,a,b,col,r=1):
    n=int(np.linalg.norm(np.subtract(b,a))*S*2)+2
    for t in np.linspace(0,1,n): dot(img,np.add(a,t*np.subtract(b,a)),col,r)
def circle(img,c,rad,col):
    for t in np.linspace(0,2*np.pi,200): dot(img,(c[0]+rad*np.cos(t),c[1]+rad*np.sin(t)),col,0)
res=pickle.load(open(SP+'paths.pkl','rb'))
for c in range(1,10):
    env,W,ags=load(c)
    img=np.full((5*S+1,5*S+1,3),255,np.uint8)
    xs=(np.arange(5*S+1)+0.5)/S
    X,Y=np.meshgrid(xs,5-xs)
    D=seg_dist(np.c_[X.ravel(),Y.ravel()],W).reshape(X.shape)
    img[D<0.35]=(235,235,235)
    for w in W: line(img,w[0],w[1],(0,0,0),1)
    for a,d in enumerate(ags):
        p=d[['x','y']].values
        for q in p[::3]: dot(img,q,tuple(int(v*0.5+127) for v in COL[a]),0)
        dot(img,p[0],COL[a],4); dot(img,p[-1],COL[a],6); circle(img,p[-1],0.35,COL[a])
    if c in res:
        path=res[c][3]
        for i in range(len(path)-1): line(img,path[i],path[i+1],(255,140,0),1)
    png(SP+f'cave{c}.png',img)
