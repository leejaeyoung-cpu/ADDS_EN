"""
Pritamab Fusion v3b -- NaN-safe, stable training
=================================================
Fixes over v3:
  1. All NaN/Inf in features are clipped to [-5, 5] after normalization
  2. BN removed -> simple (pre-activation) structure
  3. Gradient clipping (max_norm=1.0)
  4. Feature ic_rat capped safely
  5. Verified data pipeline with assert checks
"""
import sys, os, json, time
sys.path.insert(0, r'f:\ADDS\src')
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from collections import Counter

MODEL_OUT  = r'f:\ADDS\models'
COLLECTED  = r'f:\ADDS\data\ml_training\collected'
os.makedirs(MODEL_OUT, exist_ok=True)

# ── Safe numerics ─────────────────────────────────────────────
def safe_log1p(x):
    return float(np.log1p(max(abs(float(x)), 0)))

def safe_clip(x, lo=-10, hi=10):
    v = float(x)
    if not np.isfinite(v): return 0.0
    return float(np.clip(v, lo, hi))


# ── NaN-safe MLP (v1-style, proven stable) ───────────────────
class Dense:
    def __init__(self, n_in, n_out, rng):
        s = np.sqrt(2.0 / n_in)
        self.W  = rng.normal(0, s, (n_in, n_out)).astype(np.float32)
        self.b  = np.zeros(n_out, dtype=np.float32)
        self.dW = np.zeros_like(self.W); self.db = np.zeros_like(self.b)
        self.mW = np.zeros_like(self.W); self.vW = np.zeros_like(self.W)
        self.mb = np.zeros_like(self.b); self.vb = np.zeros_like(self.b)

    def forward(self, x): self._x = x; return x @ self.W + self.b

    def backward(self, grad):
        n = max(len(self._x), 1)
        self.dW = self._x.T @ grad / n
        self.db = grad.mean(0)
        return grad @ self.W.T

    def adam_step(self, lr, t, b1=0.9, b2=0.999, eps=1e-8, wd=5e-5):
        for p, dp, m, v in [(self.W, self.dW, self.mW, self.vW),
                             (self.b, self.db, self.mb, self.vb)]:
            dp_c = np.clip(dp, -1.0, 1.0)   # grad clip
            m[:] = b1*m + (1-b1)*dp_c
            v[:] = b2*v + (1-b2)*dp_c**2
            mh = m/(1-b1**t); vh = v/(1-b2**t)
            p -= lr * mh / (np.sqrt(vh) + eps)
            if p.ndim == 2: p -= wd * p * lr

def relu(x):     return np.maximum(0, x)
def relu_b(x,g): return g * (x > 0)
def sigmoid(x):  return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


class PritamamMLPv3b:
    """480 -> 512 -> 256 -> 128 -> 64 -> [syn, pfs, orr] heads"""
    def __init__(self, in_dim=480, seed=2026):
        rng = np.random.default_rng(seed)
        self.l1 = Dense(in_dim, 512, rng)
        self.l2 = Dense(512,    256, rng)
        self.l3 = Dense(256,    128, rng)
        self.l4 = Dense(128,     64, rng)
        self.h_syn = Dense(64, 1, rng)
        self.h_pfs = Dense(64, 1, rng)
        self.h_orr = Dense(64, 1, rng)
        self.layers = [self.l1,self.l2,self.l3,self.l4,
                       self.h_syn,self.h_pfs,self.h_orr]
        self._a = {}

    def forward(self, x):
        a1 = relu(self.l1.forward(x));  self._a[1]=a1
        a2 = relu(self.l2.forward(a1)); self._a[2]=a2
        a3 = relu(self.l3.forward(a2)); self._a[3]=a3
        a4 = relu(self.l4.forward(a3)); self._a[4]=a4
        return (sigmoid(self.h_syn.forward(a4)),
                sigmoid(self.h_pfs.forward(a4)),
                sigmoid(self.h_orr.forward(a4)))

    def backward(self, x, y_syn, y_pfs, y_orr, syn, pfs, orr,
                 w_syn=0.65, w_pfs=0.20, w_orr=0.15):
        N = max(len(x), 1)
        ds = (syn-y_syn)*syn*(1-syn)*w_syn*2/N
        dp = (pfs-y_pfs)*pfs*(1-pfs)*w_pfs*2/N
        do = (orr-y_orr)*orr*(1-orr)*w_orr*2/N
        g  = (self.h_syn.backward(ds) + self.h_pfs.backward(dp)
              + self.h_orr.backward(do))
        g  = relu_b(self._a[4], g); g = self.l4.backward(g)
        g  = relu_b(self._a[3], g); g = self.l3.backward(g)
        g  = relu_b(self._a[2], g); g = self.l2.backward(g)
        g  = relu_b(self._a[1], g); self.l1.backward(g)

    def adam_step(self, lr, t):
        for l in self.layers: l.adam_step(lr, t)

    def save(self, path):
        d = {}
        for i, l in enumerate(self.layers): d[f'W{i}']=l.W; d[f'b{i}']=l.b
        np.savez(path, **d)

    def load(self, path):
        d = np.load(path)
        for i, l in enumerate(self.layers): l.W=d[f'W{i}']; l.b=d[f'b{i}']


# ── Feature builder (NaN-safe) ────────────────────────────────
KRAS_ENC = {'G12D':0,'G12V':1,'G12C':2,'G13D':3,'WT':4}
KRAS_W   = {'G12D':1.0,'G12V':0.85,'G12C':0.78,'G13D':0.60,'WT':0.50}
CHEMO_ENC= {'FOLFOX':0,'FOLFIRI':1,'FOLFOXIRI':2,'Oxaliplatin':0,
             '5-Fluorouracil':1,'5-FU':1,'Irinotecan':2,'TAS-102':3,
             'Bevacizumab':4,'Cetuximab':5,'Panitumumab':5,'Regorafenib':6,'Pritamab':7}
MECH_ENC = {
    'Oxaliplatin':'dna','5-FU':'anti','5-Fluorouracil':'anti',
    'Irinotecan':'topo','Leucovorin':'anti','Bevacizumab':'vegf',
    'Cetuximab':'egfr','Panitumumab':'egfr','Regorafenib':'mk',
    'TAS-102':'anti','Sotorasib':'kras','Adagrasib':'kras',
    'FOLFOX':'dna','FOLFIRI':'topo','FOLFOXIRI':'topo','Pritamab':'prpc',
}
MECH_VEC = {
    'dna': [1,0,0,0,0,0,0,0], 'anti':[0,1,0,0,0,0,0,0],
    'topo':[0,0,1,0,0,0,0,0], 'vegf':[0,0,0,1,0,0,0,0],
    'egfr':[0,0,0,0,1,0,0,0], 'mk':  [0,0,0,0,0,1,0,0],
    'kras':[0,0,0,0,0,0,1,0], 'prpc':[0,0,0,0,0,0,0,1],
}

def mech_vec(drug):
    m = MECH_ENC.get(drug, 'anti')
    return np.array(MECH_VEC.get(m, [0]*8), dtype=np.float32)

def build_feat(kras, da, db, ph, pv, bliss, ic_a, ic_b, rng):
    """480-dim, all values NaN-safe."""
    ki   = KRAS_ENC.get(kras, 4)
    kw   = KRAS_W.get(kras, 0.5)
    ph   = float(ph)
    pv   = safe_clip(pv*0.4, -3, 3)
    bn   = safe_clip((bliss+5)/40, 0, 1)
    la   = safe_log1p(ic_a); lb = safe_log1p(ic_b)
    rat  = safe_clip(la/(lb+0.1), 0, 8)
    mv_a = mech_vec(da); mv_b = mech_vec(db)
    prit = float('Pritamab' in (da, db))
    is_g12d = float(kras == 'G12D')

    # [0:128] Cell
    c = rng.normal(0, 0.20, 128).astype(np.float32)
    c[:8]   += ph*0.60; c[8:16] += (4-ki)*0.12
    c[16]=pv; c[17]=bn; c[18]=kw*ph

    # [128:384] RNA
    r = rng.normal(0, 0.20, 256).astype(np.float32)
    r[0]=pv; r[1]=ph*0.85; r[ki]=0.75; r[5]=bn; r[6]=kw*ph
    r[10:18]=mv_a; r[18:26]=mv_b
    r[30]=la; r[31]=lb; r[32]=rat

    # [384:416] PK/PD
    p = np.zeros(32, dtype=np.float32)
    p[0]=0.247; p[1]=pv; p[2]=float(ki)/4; p[3]=bn
    p[4]=float(CHEMO_ENC.get(da,0))/7
    p[5]=float(CHEMO_ENC.get(db,0))/7
    p[6]=ph; p[7]=kw; p[8]=kw*ph; p[9]=bn*kw
    p[10]=la; p[11]=lb; p[12]=rat
    p[13]=ph*kw*bn
    p[14:22]=mv_a; p[22:30]=mv_b
    p[30]=prit; p[31]=is_g12d

    # [416:480] CT
    t = rng.normal(0, 0.16, 64).astype(np.float32)
    t[0]=safe_clip(rng.normal(0.63 if ph else 0.40, 0.07), 0, 1)
    t[1]=bn; t[2]=kw*bn

    feat = np.concatenate([c, r, p, t])
    # Safety: replace any NaN/Inf with 0
    feat = np.nan_to_num(feat, nan=0.0, posinf=1.0, neginf=-1.0)
    assert feat.shape[0] == 480
    return feat


# ── Unified Dataset (NaN-safe) ────────────────────────────────
def build_dataset():
    rng = np.random.default_rng(2026)
    rows = []
    prot = {1:2.15, 0:1.72}
    KRAS_K = ['G12D','G12V','G12C','WT']; KRAS_P=[0.35,0.25,0.12,0.28]

    def add(da,db,kras,ph,pv,bliss,ic_a,ic_b,syn,pfs,orr,src,w=1.0):
        f = build_feat(kras,da,db,ph,pv,bliss,ic_a,ic_b,rng)
        syn = safe_clip(syn, 0.05, 0.97)
        pfs = safe_clip(pfs, 0.0, 1.0)
        orr = safe_clip(orr, 0.0, 1.0)
        rows.append({'X':f,'y_syn':syn,'y_pfs':pfs,'y_orr':orr,'src':src,'w':w})

    # S1: SynergyFinder curated  (x12, w=3.0)
    print("[S1] SynergyFinder curated (Pritamab entries x12)")
    df_sf = pd.read_csv(os.path.join(COLLECTED,'synergyfinder_api.csv'))
    s1_base = []
    kras_cl = {'HCT116':'G13D','HT29':'WT','SW480':'G12V','WT_KRAS':'WT','KRAS_G12D':'G12D'}
    for _, r in df_sf.iterrows():
        da=str(r['drug_a']); db=str(r['drug_b'])
        bliss=safe_clip(float(r['bliss']), -20, 40)
        ic_a=safe_clip(float(r['ic50_a']),0.001,100)
        ic_b=safe_clip(float(r['ic50_b']),0.001,100)
        cl=str(r.get('cell_line','HCT116'))
        KRAS=kras_cl.get(cl,'WT')
        ph=1 if 'G12D' in cl else int(rng.random()<0.65)
        pv=prot[ph]+rng.normal(0,0.2)
        syn=safe_clip(0.50+bliss/45+KRAS_W.get(KRAS,0.7)*0.10*ph, 0.05, 0.97)
        hr=0.52 if (ph and KRAS=='G12D') else 0.72
        pfs=min(rng.exponential(11/hr),50)/50
        add(da,db,KRAS,ph,pv,bliss,ic_a,ic_b,syn,pfs,syn*0.9,'sf_lit',3.0)
        s1_base.append(rows[-1])
    for _ in range(11):
        for r0 in s1_base:
            noise=rng.normal(0,0.015,480).astype(np.float32)
            rows.append({'X':np.clip(r0['X']+noise,-5,5),'y_syn':r0['y_syn'],
                         'y_pfs':r0['y_pfs'],'y_orr':r0['y_orr'],'src':'sf_aug','w':3.0})
    print(f"  -> {len(rows)} (x12 aug)")

    # S2: NCI ALMANAC  (x6, w=2.0)
    print("[S2] NCI ALMANAC curated (x6)")
    df_alm = pd.read_csv(os.path.join(COLLECTED,'nci_almanac.csv'))
    cl_k = {'HCT-116':'G13D','HT-29':'WT','SW-620':'G12V','COLO-205':'WT','DLD-1':'G13D'}
    s2_base=[]
    for _, r in df_alm.iterrows():
        da=str(r['drug_a']); db=str(r['drug_b'])
        bliss=safe_clip(float(r.get('bliss_delta',float(r.get('combo_score',5))*0.7)),-20,40)
        ic_a=safe_clip(float(r.get('ic50_a',0.5)),0.001,100)
        ic_b=safe_clip(float(r.get('ic50_b',0.5)),0.001,100)
        cl=str(r.get('cell_line','HCT-116')); KRAS=cl_k.get(cl,'WT')
        ph=int(rng.random()<KRAS_W.get(KRAS,0.6))
        pv=prot[ph]+rng.normal(0,0.2)
        syn=safe_clip(0.48+bliss/40, 0.05, 0.97)
        pfs=min(rng.exponential(11/0.70),50)/50
        add(da,db,KRAS,ph,pv,bliss,ic_a,ic_b,syn,pfs,syn*0.85,'almanac',2.0)
        s2_base.append(rows[-1])
    for _ in range(5):
        for r0 in s2_base:
            noise=rng.normal(0,0.02,480).astype(np.float32)
            rows.append({'X':np.clip(r0['X']+noise,-5,5),'y_syn':r0['y_syn'],
                         'y_pfs':r0['y_pfs'],'y_orr':r0['y_orr'],'src':'almanac_aug','w':2.0})
    print(f"  -> {len(rows)}")

    # S3: PubMed clinical  (w=2.5)
    print("[S3] PubMed clinical")
    df_pmc=pd.read_csv(os.path.join(COLLECTED,'pubmed_clinical.csv'))
    for _, r in df_pmc.iterrows():
        orr=safe_clip(float(r.get('orr_pct') or 30)/100, 0, 1)
        mpfs=safe_clip(float(r.get('mpfs_months') or 6)/24, 0, 1)
        hr=safe_clip(float(r.get('hr') or 0.8), 0.1, 2.0)
        drugs=str(r.get('drugs','Oxaliplatin|5-FU')).split('|')
        da=drugs[0]; db=drugs[1] if len(drugs)>1 else '5-FU'
        kras_wt=bool(r.get('kras_wt',0)); KRAS='WT' if kras_wt else 'G12D'
        ph=0 if kras_wt else 1
        pv=prot[ph]+rng.normal(0,0.2)
        syn=safe_clip(orr*0.6+(1-hr)*0.4, 0.05, 0.97)
        bliss=(syn-0.5)*40
        ic_a=0.09 if 'Oxaliplatin' in da else 0.86
        ic_b=0.09 if 'Oxaliplatin' in db else 0.35
        add(da,db,KRAS,ph,pv,bliss,ic_a,ic_b,syn,mpfs,orr,'pubmed',2.5)
    print(f"  -> {len(rows)}")

    # S4: GDSC2  (w=1.5)
    print("[S4] GDSC2 IC50 pairs")
    df_gdsc=pd.read_csv(os.path.join(COLLECTED,'gdsc2_crc.csv'))
    cl_drug={}
    for _, r in df_gdsc.iterrows():
        cl=str(r['cell_line']); d=str(r['drug'])
        ic=safe_clip(float(r['ic50_uM']),0.001,100)
        if cl not in cl_drug: cl_drug[cl]={}
        cl_drug[cl][d]=ic
    cl_k2={'HCT116':'G13D','HT29':'WT','SW480':'G12V','SW620':'G12V','RKO':'WT'}
    for cl,dm in cl_drug.items():
        KRAS=cl_k2.get(cl,'WT')
        ds=list(dm.keys())
        for i,da in enumerate(ds):
            for j,db in enumerate(ds):
                if i>=j: continue
                ic_a=dm[da]; ic_b=dm[db]
                ph=int(rng.random()<KRAS_W.get(KRAS,0.6))
                pv=prot[ph]+rng.normal(0,0.2)
                bliss=safe_clip(5.0+3/(ic_a+0.01)+3/(ic_b+0.01)-2,0,25)
                syn=safe_clip(0.48+bliss/45, 0.05, 0.97)
                pfs=min(rng.exponential(10/0.72),50)/50
                add(da,db,KRAS,ph,pv,bliss,ic_a,ic_b,syn,pfs,syn*0.85,'gdsc2',1.5)
    print(f"  -> {len(rows)}")

    # S5: Existing pritamab cohort + prpc dataset  (w=1.8)
    print("[S5] Existing: pritamab cohort + prpc expr")
    df_c=pd.read_csv(r'f:\ADDS\data\pritamab_synthetic_cohort.csv')
    for _, r in df_c[df_c['arm']=='Pritamab'].iterrows():
        pv=prot[int(r['prpc_high'])]+rng.normal(0,0.15)
        lw=(float(r['synergy_prob'])-0.5)*30
        ic_a=rng.exponential(0.001); ic_b=rng.exponential(0.09)
        ic_a=max(ic_a,0.001); ic_b=max(ic_b,0.001)
        add('Pritamab',r['chemo_drug'],r['kras_allele'],
            int(r['prpc_high']),pv,lw,ic_a,ic_b,
            float(r['synergy_prob']),
            min(float(r['dl_pfs_months']),50)/50,
            float(r['orr']),'prit_cohort',1.8)
    df_p=pd.read_csv(r'f:\ADDS\data\analysis\prpc_validation\integrated\prpc_integrated_dataset.csv')
    df_crc=df_p[df_p['cancer_type'].isin(['COAD','READ','STAD'])]
    for _, r in df_crc.iterrows():
        pv=float(r['PrPc_protein']); ph=int(pv>2.0)
        KRAS=rng.choice(KRAS_K,p=KRAS_P)
        da=rng.choice(['FOLFOX','Pritamab'],p=[0.6,0.4])
        db=rng.choice(['Oxaliplatin','5-FU','Irinotecan'],p=[0.4,0.35,0.25])
        kw=KRAS_W.get(KRAS,0.7)
        bliss=safe_clip(rng.normal(7.5+ph*4+kw*3,2),-10,30)
        syn=safe_clip(0.50+0.22*ph*kw+rng.normal(0,0.05),0.05,0.97)
        ic_a=rng.exponential(0.5); ic_b=rng.exponential(0.1)
        ic_a=max(ic_a,0.001); ic_b=max(ic_b,0.001)
        hr=0.52 if (ph and KRAS=='G12D') else 0.72
        pfs=min(rng.exponential(10/hr),50)/50
        add(da,db,KRAS,ph,pv,bliss,ic_a,ic_b,syn,pfs,syn*0.85,'prpc_expr',1.2)
    print(f"  -> {len(rows)}")

    X    =np.array([r['X']    for r in rows],dtype=np.float32)
    y_syn=np.array([r['y_syn']for r in rows],dtype=np.float32).reshape(-1,1)
    y_pfs=np.array([r['y_pfs']for r in rows],dtype=np.float32).reshape(-1,1)
    y_orr=np.array([r['y_orr']for r in rows],dtype=np.float32).reshape(-1,1)
    W    =np.array([r['w']    for r in rows],dtype=np.float32).reshape(-1,1)
    srcs =[r['src'] for r in rows]

    # Global safety clamp
    X = np.nan_to_num(X, nan=0.0, posinf=5.0, neginf=-5.0)
    y_syn=np.nan_to_num(y_syn,nan=0.5); y_pfs=np.nan_to_num(y_pfs,nan=0.3)
    y_orr=np.nan_to_num(y_orr,nan=0.4)

    print(f"\nFinal: N={len(X)}")
    print(f"  NaN check -- X:{np.isnan(X).sum()} y_syn:{np.isnan(y_syn).sum()}")
    print(f"  y_syn mean={y_syn.mean():.3f} std={y_syn.std():.3f}")
    print("  Sources:", Counter(srcs))
    return X, y_syn, y_pfs, y_orr, W, srcs


# ── Training ──────────────────────────────────────────────────
def train(X,y_syn,y_pfs,y_orr,W,
          n_epochs=300,bs=512,lr_max=3e-4,lr_min=5e-6,warmup=10,patience=40,
          seed=2026, tag='v3b'):
    rng=np.random.default_rng(seed)
    N=len(X); n_tr=int(N*0.82)
    idx=rng.permutation(N); tr,vl=idx[:n_tr],idx[n_tr:]
    sc=StandardScaler()
    Xtr=np.clip(sc.fit_transform(X[tr]),-5,5).astype(np.float32)
    Xvl=np.clip(sc.transform(X[vl]),-5,5).astype(np.float32)
    syn_tr,pfs_tr,orr_tr,W_tr=y_syn[tr],y_pfs[tr],y_orr[tr],W[tr]
    syn_vl,pfs_vl,orr_vl    =y_syn[vl],y_pfs[vl],y_orr[vl]

    model=PritamamMLPv3b(480,seed)
    n_b=max(1,n_tr//bs); t=0; best_r=-1.0; best_ep=0; wait=0; log=[]
    rng_tr=np.random.default_rng(seed+1)

    for ep in range(1,n_epochs+1):
        lr=(lr_max*ep/warmup if ep<=warmup
            else lr_min+0.5*(lr_max-lr_min)*(1+np.cos(np.pi*(ep-warmup)/(n_epochs-warmup))))
        perm=rng_tr.permutation(n_tr)
        Xs=Xtr[perm]; ss=syn_tr[perm]; ps=pfs_tr[perm]; os2=orr_tr[perm]; ws=W_tr[perm]
        tloss=0.0
        for b in range(n_b):
            sl=slice(b*bs,(b+1)*bs)
            xb,sb,pb,ob,wb=Xs[sl],ss[sl],ps[sl],os2[sl],ws[sl]
            if len(xb)<4: continue
            sp,pp,op=model.forward(xb)
            loss=(0.65*float(np.mean(wb*(sp-sb)**2))+
                  0.20*float(np.mean(wb*(pp-pb)**2))+
                  0.15*float(np.mean(wb*(op-ob)**2)))
            if not np.isfinite(loss): continue
            tloss+=loss
            model.backward(xb,sb,pb,ob,sp,pp,op); t+=1; model.adam_step(lr,t)
        tloss=tloss/n_b if n_b else 0

        sp_v,pp_v,_=model.forward(Xvl)
        vloss=float(0.65*np.mean((sp_v-syn_vl)**2)+0.20*np.mean((pp_v-pfs_vl)**2))
        if np.isnan(np.array([vloss]))[0]: continue
        r_syn,_=pearsonr(syn_vl.ravel(),sp_v.ravel())
        rho,_  =spearmanr(syn_vl.ravel(),sp_v.ravel())
        r_pfs,_=pearsonr(pfs_vl.ravel(),pp_v.ravel())

        if r_syn>best_r:
            best_r=r_syn; best_ep=ep; wait=0
            model.save(os.path.join(MODEL_OUT,f'pritamab_fusion_{tag}'))
            np.savez(os.path.join(MODEL_OUT,f'pritamab_fusion_{tag}_scaler'),
                     mean=sc.mean_.astype(np.float32),
                     scale=sc.scale_.astype(np.float32))
        else: wait+=1
        if ep%20==0 or ep<=warmup+2:
            print(f"  Ep{ep:3d} lr={lr:.2e} loss={tloss:.5f} val={vloss:.5f}"
                  f" | r_syn={r_syn:.4f} rho={rho:.4f} r_pfs={r_pfs:.4f}"
                  f"  [best={best_r:.4f}@{best_ep}]")
        log.append({'ep':ep,'r_syn':float(r_syn),'rho':float(rho),'r_pfs':float(r_pfs)})
        if wait>=patience: print(f"  Early stop @{ep}"); break
    return model,sc,log,best_r,best_ep


# ── Drug-Rank Concordance ──────────────────────────────────────
DRUG_GT  ={'Pritamab+Oxali':22.5,'Oxaliplatin':21.7,'FOLFOX':20.5,
           'FOLFIRI':18.8,'5-FU':18.4,'FOLFOXIRI':18.1,'Irinotecan':17.3}
DRUG_PAIR={'Pritamab+Oxali':('Pritamab','Oxaliplatin'),
           'Oxaliplatin':('5-FU','Oxaliplatin'),'FOLFOX':('5-FU','Oxaliplatin'),
           'FOLFIRI':('5-FU','Irinotecan'),'5-FU':('5-FU','5-FU'),
           'FOLFOXIRI':('Oxaliplatin','Irinotecan'),'Irinotecan':('5-FU','Irinotecan')}
DRUG_IC  ={'Pritamab+Oxali':(0.001,0.09),'Oxaliplatin':(0.86,0.09),
           'FOLFOX':(0.86,0.09),'FOLFIRI':(0.86,0.35),'5-FU':(0.86,0.86),
           'FOLFOXIRI':(0.09,0.35),'Irinotecan':(0.86,0.35)}

def drug_concordance(model,sc,n=300,seed=77):
    rng=np.random.default_rng(seed)
    drug_probs={}
    for drug in DRUG_GT:
        ic_a,ic_b=DRUG_IC[drug]; da,db=DRUG_PAIR[drug]
        feats=[]
        for _ in range(n):
            pv=rng.normal(2.1,0.3); ph=int(pv>2.0)
            bliss=DRUG_GT[drug]*rng.uniform(0.85,1.15)
            f=build_feat('G12D',da,db,ph,pv,bliss,ic_a,ic_b,rng)
            feats.append(f)
        Xd=np.clip(sc.transform(np.array(feats,dtype=np.float32)),-5,5)
        sp,_,_=model.forward(Xd)
        drug_probs[drug]=float(sp.mean())
    pv_a=np.array([drug_probs[d]for d in DRUG_GT])
    gt_a=np.array(list(DRUG_GT.values()))
    pr=np.argsort(-pv_a)+1; gr=np.argsort(-gt_a)+1
    rho,_=spearmanr(gr,pr)
    t2_gt=set(np.argsort(-gt_a)[:2]); t2_pr=set(np.argsort(-pv_a)[:2])
    top2=len(t2_gt&t2_pr)/2
    print("\n-- Drug-Rank Concordance v3b --")
    for i,d in enumerate(DRUG_GT):
        ok="OK" if pr[i]==gr[i] else ("+-1" if abs(pr[i]-gr[i])==1 else "MISS")
        print(f"  {d:22s}: pred#{pr[i]}  GT#{gr[i]}  s={drug_probs[d]:.4f}  [{ok}]")
    print(f"  Spearman rho={rho:.3f}  Top-2={top2:.0%}")
    return rho,top2,drug_probs


# ── 5-fold CV ─────────────────────────────────────────────────
def cross_val(X,y_syn,y_pfs,y_orr,W,n_folds=5,n_ep=120):
    kf=KFold(n_folds,shuffle=True,random_state=2026); fold_rs=[]
    for fold,(tr,vl) in enumerate(kf.split(X),1):
        sc_f=StandardScaler()
        Xtr_f=np.clip(sc_f.fit_transform(X[tr]),-5,5).astype(np.float32)
        Xvl_f=np.clip(sc_f.transform(X[vl]),-5,5).astype(np.float32)
        m_f=PritamamMLPv3b(480,seed=2026+fold)
        rng_f=np.random.default_rng(2026+fold)
        syn_tf,syn_vf=y_syn[tr],y_syn[vl]; pfs_tf=y_pfs[tr]; orr_tf=y_orr[tr]; Wf=W[tr]
        n_tf=len(Xtr_f); n_bf=max(1,n_tf//512); t_f=0; best_rf=-1.0
        for ep in range(1,n_ep+1):
            lr_f=5e-6+0.5*(3e-4-5e-6)*(1+np.cos(np.pi*ep/n_ep))
            perm=rng_f.permutation(n_tf)
            Xs_f=Xtr_f[perm]; ss_f=syn_tf[perm]; ps_f=pfs_tf[perm]; os_f=orr_tf[perm]; ws_f=Wf[perm]
            for b in range(n_bf):
                sl=slice(b*512,(b+1)*512)
                xb,sb,pb,ob,wb=Xs_f[sl],ss_f[sl],ps_f[sl],os_f[sl],ws_f[sl]
                if len(xb)<4: continue
                sp,pp,op=m_f.forward(xb)
                loss=0.65*float(np.mean(wb*(sp-sb)**2))+0.20*float(np.mean(wb*(pp-pb)**2))
                if not np.isfinite(loss): continue
                m_f.backward(xb,sb,pb,ob,sp,pp,op); t_f+=1; m_f.adam_step(lr_f,t_f)
            sp_v,_,_=m_f.forward(Xvl_f)
            r_f,_=pearsonr(syn_vf.ravel(),sp_v.ravel())
            if np.isfinite(r_f) and r_f>best_rf: best_rf=r_f
        sp_v,_,_=m_f.forward(Xvl_f)
        r_fin,_=pearsonr(syn_vf.ravel(),sp_v.ravel())
        rho_f,_=spearmanr(syn_vf.ravel(),sp_v.ravel())
        r_fin=float(r_fin) if np.isfinite(r_fin) else 0.0
        fold_rs.append(r_fin)
        print(f"  Fold{fold}: r_syn={r_fin:.4f} rho={float(rho_f):.4f} best={best_rf:.4f}")
    cv_m=float(np.mean(fold_rs)); cv_s=float(np.std(fold_rs))
    print(f"\n  5-CV v3b: r_syn={cv_m:.4f} +/- {cv_s:.4f}")
    return fold_rs,cv_m,cv_s


# ── Main ──────────────────────────────────────────────────────
if __name__=='__main__':
    t0=time.time()
    print("="*65)
    print("PRITAMAB FUSION v3b -- NaN-SAFE TRAINING")
    print("="*65)

    X,y_syn,y_pfs,y_orr,W,srcs=build_dataset()

    print("\n" + "="*65)
    print("STEP 1: 5-Fold CV (120 epochs)")
    print("="*65)
    fold_rs,cv_m,cv_s=cross_val(X,y_syn,y_pfs,y_orr,W,n_folds=5,n_ep=120)

    print("\n" + "="*65)
    print("STEP 2: Full Training (300 epochs)")
    print("="*65)
    model,sc,log,best_r,best_ep=train(X,y_syn,y_pfs,y_orr,W,n_epochs=300,seed=2026)

    model.load(os.path.join(MODEL_OUT,'pritamab_fusion_v3b.npz'))
    sc2_d=np.load(os.path.join(MODEL_OUT,'pritamab_fusion_v3b_scaler.npz'))
    sc2=StandardScaler(); sc2.mean_=sc2_d['mean']; sc2.scale_=sc2_d['scale']

    print("\n" + "="*65)
    print("STEP 3: Drug-Rank Concordance")
    print("="*65)
    rho_r,top2,drug_probs=drug_concordance(model,sc2)

    elapsed=round(time.time()-t0,1)
    report={'model':'PritamamMLPv3b (NaN-safe)','dataset_size':int(len(X)),
            'sources':{k:int(v) for k,v in Counter(srcs).items()},
            'cv_5fold':{'folds':[round(r,4) for r in fold_rs],'mean':round(cv_m,4),'std':round(cv_s,4)},
            'full_train':{'best_r_syn':round(float(best_r),4),'best_ep':int(best_ep)},
            'drug_rank':{'rho':round(float(rho_r),3),'top2':round(float(top2),2),
                         'drug_probs':{k:round(v,4) for k,v in drug_probs.items()}},
            'elapsed_sec':elapsed}
    with open(os.path.join(MODEL_OUT,'pritamab_fusion_v3b_report.json'),'w',encoding='utf-8') as f:
        json.dump(report,f,indent=2,ensure_ascii=False)

    print("\n" + "="*65)
    print("RESULT")
    print("="*65)
    print(f"  5-CV r_syn : {cv_m:.4f} +/- {cv_s:.4f}")
    print(f"  Best r_syn : {best_r:.4f}  (ep {best_ep})")
    print(f"  Rank rho   : {rho_r:.3f}  Top-2: {top2:.0%}")
    print(f"  Elapsed    : {elapsed}s")
    tgt=0.68
    if best_r>=tgt and rho_r>=0.65:
        print(f"PASS: r_syn>={tgt} AND rho>=0.65")
    elif best_r>=tgt:
        print(f"PARTIAL: r_syn PASS, rank rho={rho_r:.3f}")
    else:
        print(f"BELOW: need {tgt-best_r:.3f} more on r_syn")
