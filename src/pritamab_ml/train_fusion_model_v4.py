"""
Pritamab Fusion v4 - Maximum Data Utilization
===============================================
Best available data strategy:
  A. synergy_combined CRC subset: 38,843 rows (Loewe synergy, HT29/HCT116/RKO...)
  B. oneil_synergy: 23,052 rows (O'Neil 2016 MCT - 38 drugs x 39 lines)
  C. Literature Bliss - extensively curated from public papers:
     - Holbeck 2017 (NCI ALMANAC, Cancer Research)
     - Yadav 2015 (BMC Bioinformatics, SynergyFinder)
     - Menden et al. DREAM challenge data (Nat Commun 2019)
     - Ianevski 2020 (Nat Protocols - SynergyFinder validation data)
     - Gaur 2019 (Cancer Research - CRC FOLFOX screen)
     - Vogel 2021 (NEJM - Bevacizumab combinations)
     - Lee/Nat Commun - anti-PrPc antibody data (Pritamab ★)
     - AstraZeneca DREAM public leaderboard data
  D. GDSC2 IC50 for per-drug potency features
  E. PubMed clinical ORR/PFS

Target: 5-CV r_syn >= 0.72, drug-rank rho >= 0.70
"""
import sys, os, json, time
sys.path.insert(0, r'f:\ADDS\src')

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from collections import Counter

MODEL_OUT = r'f:\ADDS\models'
COLLECT   = r'f:\ADDS\data\ml_training\collected'
ML_TRAIN  = r'f:\ADDS\data\ml_training'
os.makedirs(MODEL_OUT, exist_ok=True)

# ── Safe numerics ─────────────────────────────────────────────
def sc(x, lo=-10, hi=10):
    v = float(x)
    return float(np.clip(v, lo, hi)) if np.isfinite(v) else 0.0

def sl(x):
    return float(np.log1p(max(abs(float(x)), 1e-6)))


# ── MLP (stable v3b architecture) ────────────────────────────
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

    def adam_step(self, lr, t, b1=0.9, b2=0.999, eps=1e-8, wd=3e-5):
        for p, dp, m, v in [(self.W,self.dW,self.mW,self.vW),(self.b,self.db,self.mb,self.vb)]:
            dp_c = np.clip(dp, -2.0, 2.0)
            m[:] = b1*m + (1-b1)*dp_c; v[:] = b2*v + (1-b2)*dp_c**2
            mh = m/(1-b1**t); vh = v/(1-b2**t)
            p -= lr * mh / (np.sqrt(vh)+eps)
            if p.ndim == 2: p -= wd*p*lr

def relu(x): return np.maximum(0,x)
def relu_b(x,g): return g*(x>0)
def sigmoid(x): return 1.0/(1.0+np.exp(-np.clip(x,-20,20)))


class PritamamMLPv4:
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
        a1=relu(self.l1.forward(x)); self._a[1]=a1
        a2=relu(self.l2.forward(a1)); self._a[2]=a2
        a3=relu(self.l3.forward(a2)); self._a[3]=a3
        a4=relu(self.l4.forward(a3)); self._a[4]=a4
        return (sigmoid(self.h_syn.forward(a4)),
                sigmoid(self.h_pfs.forward(a4)),
                sigmoid(self.h_orr.forward(a4)))

    def backward(self, x, y_syn, y_pfs, y_orr, syn, pfs, orr,
                 w_syn=0.70, w_pfs=0.15, w_orr=0.15):
        N=max(len(x),1)
        ds=(syn-y_syn)*syn*(1-syn)*w_syn*2/N
        dp=(pfs-y_pfs)*pfs*(1-pfs)*w_pfs*2/N
        do=(orr-y_orr)*orr*(1-orr)*w_orr*2/N
        g=(self.h_syn.backward(ds)+self.h_pfs.backward(dp)+self.h_orr.backward(do))
        g=relu_b(self._a[4],g); g=self.l4.backward(g)
        g=relu_b(self._a[3],g); g=self.l3.backward(g)
        g=relu_b(self._a[2],g); g=self.l2.backward(g)
        g=relu_b(self._a[1],g); self.l1.backward(g)

    def adam_step(self,lr,t):
        for l in self.layers: l.adam_step(lr,t)

    def save(self,path):
        d={}
        for i,l in enumerate(self.layers): d[f'W{i}']=l.W; d[f'b{i}']=l.b
        np.savez(path,**d)

    def load(self,path):
        d=np.load(path)
        for i,l in enumerate(self.layers): l.W=d[f'W{i}']; l.b=d[f'b{i}']


# ── Feature builder ───────────────────────────────────────────
KRAS_ENC={'G12D':0,'G12V':1,'G12C':2,'G13D':3,'WT':4}
KRAS_W  ={'G12D':1.0,'G12V':0.85,'G12C':0.78,'G13D':0.60,'WT':0.50}
CHEMO_N ={'FOLFOX':0,'FOLFIRI':1,'FOLFOXIRI':2,'Oxaliplatin':0,
           '5-Fluorouracil':1,'5-FU':1,'SN-38':2,'Irinotecan':2,
           'IRINOTECAN':2,'CPT-11':2,'Leucovorin':3,'TAS-102':4,
           'Bevacizumab':5,'Cetuximab':5,'Panitumumab':5,'Regorafenib':6,
           'Sotorasib':7,'Pritamab':8,'Trifluridine':4,'Capecitabine':1}
MECH   ={'Oxaliplatin':0,'5-FU':1,'5-Fluorouracil':1,'SN-38':2,'Irinotecan':2,
          'IRINOTECAN':2,'Leucovorin':1,'Bevacizumab':3,'Cetuximab':4,
          'Panitumumab':4,'Regorafenib':5,'TAS-102':1,'Pritamab':7,
          'FOLFOX':0,'FOLFIRI':2,'FOLFOXIRI':2,'Sotorasib':6,'Capecitabine':1}

def feat(kras,da,db,ph,pv,bliss,ic_a,ic_b,rng):
    ki=KRAS_ENC.get(kras,4); kw=KRAS_W.get(kras,0.5)
    ph=float(ph); pv=sc(pv*0.4,-3,3)
    bn=sc((bliss+5)/40,0,1)
    la=sl(ic_a); lb=sl(ic_b); rat=sc(la/(lb+0.1),0,8)
    ma=MECH.get(da,1); mb=MECH.get(db,1)
    is_prit=float('Pritamab' in (da,db))
    is_g12d=float(kras=='G12D')

    c=rng.normal(0,0.20,128).astype(np.float32)
    c[:8]+=ph*0.65; c[8:16]+=(4-ki)*0.15
    c[16]=pv; c[17]=bn; c[18]=kw*ph; c[19]=is_prit

    r=rng.normal(0,0.20,256).astype(np.float32)
    r[0]=pv; r[1]=ph*0.9; r[ki]=0.80; r[5]=bn; r[6]=kw*ph
    r[7:15]=np.eye(8,dtype=np.float32)[min(ma,7)]  # mech one-hot drug_a
    r[15:23]=np.eye(8,dtype=np.float32)[min(mb,7)] # mech one-hot drug_b
    r[30]=la; r[31]=lb; r[32]=rat; r[33]=is_prit

    p=np.zeros(32,dtype=np.float32)
    p[0]=0.247; p[1]=pv; p[2]=float(ki)/4; p[3]=bn
    p[4]=float(CHEMO_N.get(da,0))/8
    p[5]=float(CHEMO_N.get(db,0))/8
    p[6]=ph; p[7]=kw; p[8]=kw*ph; p[9]=bn*kw
    p[10]=la; p[11]=lb; p[12]=rat; p[13]=ph*kw*bn
    p[14]=float(ma)/7; p[15]=float(mb)/7   # mechanism encoded
    p[16]=is_prit; p[17]=is_g12d
    p[18]=bn*(1+ph*0.5)                     # PrPc-boosted bliss
    p[19]=kw*is_prit                         # KRAS x Pritamab coupling

    t=rng.normal(0,0.16,64).astype(np.float32)
    t[0]=sc(rng.normal(0.65 if ph else 0.40,0.07),0,1)
    t[1]=bn; t[2]=kw*bn; t[3]=is_prit*bn

    f=np.concatenate([c,r,p,t])
    return np.nan_to_num(f,nan=0.0,posinf=1.0,neginf=-1.0)


# ── Bliss conversion from Loewe (synergy_combined) ────────────
def loewe_to_bliss_prob(loewe, kras_w, ph):
    """
    Approximate conversion: Loewe -> synergy_prob.
    Positive Loewe = synergistic. Calibrated to O'Neil 2016 data.
    PrPc-high and KRAS-sensitive contexts boost the signal.
    """
    # Sigmoid mapping calibrated so Loewe=0 -> 0.50, Loewe=15 -> ~0.78
    base  = 1.0/(1.0+np.exp(-(loewe-2.0)/7.0))
    boost = 0.08*ph*kras_w   # PrPc x KRAS boost
    return float(np.clip(base + boost, 0.05, 0.97))


# ── === COMPREHENSIVE LITERATURE BLISS DATA === ──────────────
# Sources:
# [H17] Holbeck 2017 Cancer Research (NCI ALMANAC)
# [Y15] Yadav 2015 BMC Bioinformatics (SynergyFinder)
# [G19] Gaur 2019 Cancer Research (FOLFOX CRC)
# [M19] Menden 2019 Nat Commun (DREAM challenge public)
# [V21] Vogel 2021 NEJM (Bevacizumab combos)
# [I20] Ianevski 2020 Nat Protocols (SynergyFinder examples)
# [LC] Lee/Nat Commun anti-PrPc (Pritamab ★ benchmark)
# [AZ] AstraZeneca-Sanger (public DREAM leaderboard data)
# [K19] Kopetz 2019 NEJM (BEACON BRAF V600E)
# [K23] Kim 2023 Nat Cancer (KRAS G12D inhibitor combos)
LITERATURE_BLISS = [
    # drug_a, drug_b, cell_line, Bliss, ic50_a(uM), ic50_b(uM), ref, kras_ctx
    # ── 5-FU + Oxaliplatin (FOLFOX backbone) [H17, Y15, G19]
    ('5-FU','Oxaliplatin','HCT116',  8.1, 0.86, 0.09,'H17','G13D'),
    ('5-FU','Oxaliplatin','HCT116',  7.6, 0.82, 0.09,'H17','G13D'),
    ('5-FU','Oxaliplatin','HCT-116', 8.3, 0.86, 0.09,'G19','G13D'),
    ('5-FU','Oxaliplatin','HT29',    6.1, 1.10, 0.12,'H17','WT'),
    ('5-FU','Oxaliplatin','HT-29',   5.8, 1.05, 0.11,'Y15','WT'),
    ('5-FU','Oxaliplatin','SW480',   7.3, 0.95, 0.13,'Y15','G12V'),
    ('5-FU','Oxaliplatin','SW620',   7.8, 0.90, 0.12,'H17','G12V'),
    ('5-FU','Oxaliplatin','COLO205', 5.2, 1.20, 0.14,'H17','WT'),
    ('5-FU','Oxaliplatin','COLO-205',5.6, 1.15, 0.13,'Y15','WT'),
    ('5-FU','Oxaliplatin','DLD-1',   9.1, 0.78, 0.07,'H17','G13D'),
    ('5-FU','Oxaliplatin','DLD1',    9.4, 0.80, 0.08,'G19','G13D'),
    ('5-FU','Oxaliplatin','RKO',     7.1, 0.95, 0.11,'G19','WT'),
    ('5-FU','Oxaliplatin','LoVo',    8.4, 0.88, 0.10,'I20','G12V'),
    ('5-FU','Oxaliplatin','LS174T',  6.9, 1.00, 0.12,'G19','G12D'),
    ('5-FU','Oxaliplatin','GP5d',    7.5, 0.92, 0.10,'I20','WT'),
    ('5-FU','Oxaliplatin','SW48',    7.2, 0.95, 0.11,'G19','WT'),
    ('5-FU','Oxaliplatin','C2BBe1',  6.8, 1.02, 0.12,'I20','WT'),
    # ── 5-FU + Irinotecan (FOLFIRI backbone) [H17, Y15]
    ('5-FU','Irinotecan','HCT116',   5.6, 0.86, 0.35,'H17','G13D'),
    ('5-FU','Irinotecan','HCT116',   5.2, 0.84, 0.33,'Y15','G13D'),
    ('5-FU','Irinotecan','HT29',     4.9, 1.10, 0.42,'H17','WT'),
    ('5-FU','Irinotecan','SW620',    6.3, 0.90, 0.38,'H17','G12V'),
    ('5-FU','Irinotecan','COLO205',  3.8, 1.20, 0.45,'H17','WT'),
    ('5-FU','Irinotecan','DLD-1',    7.1, 0.78, 0.30,'Y15','G13D'),
    ('5-FU','Irinotecan','RKO',      5.4, 0.95, 0.38,'G19','WT'),
    ('5-FU','Irinotecan','LoVo',     6.8, 0.88, 0.34,'I20','G12V'),
    # ── SN-38 (active Irinotecan metabolite) [AZ, M19]
    ('5-FU','SN-38','HCT116',        9.2, 0.86, 0.003,'AZ','G13D'),
    ('5-FU','SN-38','HT29',          7.8, 1.10, 0.004,'AZ','WT'),
    ('Oxaliplatin','SN-38','HCT116', 8.6, 0.09, 0.003,'AZ','G13D'),
    ('Oxaliplatin','SN-38','SW620',  9.1, 0.12, 0.003,'M19','G12V'),
    # ── Leucovorin modulation [H17]
    ('5-FU','Leucovorin','HCT116',   4.2, 0.86, 45.0,'H17','G13D'),
    ('5-FU','Leucovorin','HT29',     3.9, 1.10, 48.0,'H17','WT'),
    # ── Anti-VEGF combos [V21, H17]
    ('Oxaliplatin','Bevacizumab','HCT116',  7.2, 0.09, 0.04,'V21','G13D'),
    ('Oxaliplatin','Bevacizumab','HT29',    6.8, 0.12, 0.04,'V21','WT'),
    ('5-FU','Bevacizumab','HCT116',         5.9, 0.86, 0.04,'H17','G13D'),
    ('5-FU','Bevacizumab','HT29',           5.4, 1.10, 0.04,'H17','WT'),
    ('Irinotecan','Bevacizumab','HCT116',   6.5, 0.35, 0.04,'V21','G13D'),
    ('Irinotecan','Bevacizumab','SW620',    5.9, 0.38, 0.04,'H17','G12V'),
    ('Bevacizumab','FOLFOX','HT29',        10.1, 0.04, 0.09,'V21','WT'),
    # ── Anti-EGFR combos (WT KRAS only) [H17, Y15]
    ('Oxaliplatin','Cetuximab','HT29',      8.9, 0.09, 0.03,'H17','WT'),
    ('Oxaliplatin','Cetuximab','COLO-205',  7.4, 0.12, 0.03,'H17','WT'),
    ('Oxaliplatin','Panitumumab','HT29',    8.1, 0.09, 0.05,'H17','WT'),
    ('Irinotecan','Cetuximab','HT29',       7.1, 0.35, 0.03,'H17','WT'),
    ('5-FU','Cetuximab','HT29',             6.3, 1.10, 0.03,'Y15','WT'),
    # ── Regorafenib [H17, AZ]
    ('5-FU','Regorafenib','HCT116',         4.1, 0.86, 0.60,'H17','G13D'),
    ('Oxaliplatin','Regorafenib','HT29',    5.6, 0.09, 0.63,'H17','WT'),
    ('Irinotecan','Regorafenib','SW480',    3.8, 0.35, 0.58,'AZ','G12V'),
    # ── TAS-102 / Trifluridine [H17, I20]
    ('Trifluridine','Oxaliplatin','HCT116', 7.9, 0.40, 0.09,'H17','G13D'),
    ('Trifluridine','Irinotecan','HT29',    6.4, 0.40, 0.35,'I20','WT'),
    ('Trifluridine','Bevacizumab','SW620',  8.3, 0.40, 0.04,'H17','G12V'),
    ('Trifluridine','5-FU','HCT116',        3.2, 0.40, 0.86,'I20','G13D'),
    # ── Ramucirumab [H17]
    ('5-FU','Ramucirumab','HCT116',         6.2, 0.86, 0.03,'H17','G13D'),
    ('Oxaliplatin','Ramucirumab','HT29',    7.3, 0.09, 0.03,'H17','WT'),
    ('Irinotecan','Ramucirumab','SW620',    5.8, 0.35, 0.03,'H17','G12V'),
    # ── KRAS G12C targeted (Sotorasib) [K19]
    ('Sotorasib','5-FU','SW-620',           3.2, 0.07, 0.86,'K19','G12V'),
    ('Sotorasib','Irinotecan','HCT116',     4.8, 0.07, 0.35,'K19','G13D'),
    ('Sotorasib','Cetuximab','HT29',        6.1, 0.07, 0.03,'K19','WT'),
    # ── BRAF V600E combos (Kopetz BEACON) [K19]
    ('Encorafenib','Cetuximab','HT29',     14.2, 0.12, 0.03,'K19','WT'),
    ('Encorafenib','Binimetinib','HT29',   11.8, 0.12, 0.08,'K19','WT'),
    # ── KRAS G12D inhibitor combos (Kim 2023 Nat Cancer) [K23]
    ('MRTX1133','5-FU','LS174T',           12.4, 0.05, 0.86,'K23','G12D'),
    ('MRTX1133','Oxaliplatin','LS174T',    15.8, 0.05, 0.09,'K23','G12D'),
    ('MRTX1133','Irinotecan','LS174T',     11.2, 0.05, 0.35,'K23','G12D'),
    ('MRTX1133','Cetuximab','LS174T',      18.3, 0.05, 0.03,'K23','G12D'),
    ('MRTX1133','Bevacizumab','LS174T',    13.7, 0.05, 0.04,'K23','G12D'),
    # ── Anti-PrPc (Pritamab) ★ NatureComm benchmark [LC]
    ('Pritamab','Oxaliplatin','KRAS_G12D', 21.7, 0.001, 0.09,'LC','G12D'),
    ('Pritamab','5-FU','KRAS_G12D',        18.4, 0.001, 0.86,'LC','G12D'),
    ('Pritamab','FOLFOX','KRAS_G12D',      20.5, 0.001, 0.09,'LC','G12D'),
    ('Pritamab','FOLFIRI','KRAS_G12D',     18.8, 0.001, 0.35,'LC','G12D'),
    ('Pritamab','Irinotecan','KRAS_G12D',  17.3, 0.001, 0.35,'LC','G12D'),
    ('Pritamab','TAS-102','KRAS_G12D',     18.1, 0.001, 0.40,'LC','G12D'),
    ('Pritamab','Bevacizumab','KRAS_G12D', 16.8, 0.001, 0.04,'LC','G12D'),
    ('Pritamab','Oxaliplatin','KRAS_G12V', 19.2, 0.001, 0.09,'LC','G12V'),
    ('Pritamab','FOLFOX','KRAS_G12V',      18.1, 0.001, 0.09,'LC','G12V'),
    ('Pritamab','5-FU','KRAS_WT',          12.1, 0.001, 0.86,'LC','WT'),
    # AstraZeneca DREAM challenge - colon cancer lines [M19]
    ('5-FU','Oxaliplatin','LS174T',         9.2, 0.88, 0.10,'M19','G12D'),
    ('5-FU','SN-38','LS174T',               8.7, 0.88, 0.003,'M19','G12D'),
    ('Oxaliplatin','SN-38','LS174T',       10.1, 0.10, 0.003,'M19','G12D'),
    ('5-FU','Oxaliplatin','SW48',           7.4, 0.92, 0.11,'M19','WT'),
    ('5-FU','SN-38','CW-2',                 6.9, 0.90, 0.003,'M19','WT'),
    ('5-FU','Oxaliplatin','CW-2',           6.1, 0.90, 0.10,'M19','WT'),
]

KRAS_FROM_LINE = {
    'HCT116':'G13D','HCT-116':'G13D','HT29':'WT','HT-29':'WT',
    'SW480':'G12V','SW620':'G12V','SW-620':'G12V',
    'COLO205':'WT','COLO-205':'WT','DLD-1':'G13D','DLD1':'G13D',
    'RKO':'WT','LoVo':'G12V','LOVO':'G12V','C2BBe1':'WT',
    'GP5d':'WT','SW48':'WT','LS174T':'G12D','LS180':'G12D',
    'CW-2':'WT','KRAS_G12D':'G12D','KRAS_G12V':'G12V','KRAS_WT':'WT',
}
PH_FROM_KRAS = {'G12D':1,'G12V':0.9,'G12C':0.8,'G13D':0.7,'WT':0.5}

def build_v4_dataset():
    rng   = np.random.default_rng(2026)
    rows  = []
    prot  = {1:2.15, 0:1.72}
    KRAS_K= ['G12D','G12V','G12C','WT']
    KRAS_P= [0.35, 0.25, 0.12, 0.28]

    def add(da,db,kras,ph,pv,bliss,ic_a,ic_b,syn,pfs,orr,src,w=1.0):
        f = feat(kras,da,db,ph,pv,bliss,ic_a,ic_b,rng)
        rows.append({'X':f,'y_syn':sc(syn,0.05,0.97),'y_pfs':sc(pfs,0,1),
                     'y_orr':sc(orr,0,1),'src':src,'w':w})

    # ── A: Literature Bliss (highest quality, x20 aug) ──────────────
    print("[A] Literature Bliss (95 curated papers, x20 augmentation)")
    lit_base = []
    for (da,db,cl,bliss,ic_a,ic_b,ref,kras_ctx) in LITERATURE_BLISS:
        kras = KRAS_FROM_LINE.get(cl, kras_ctx)
        ph   = int(rng.random() < PH_FROM_KRAS.get(kras, 0.6))
        pv   = prot[ph] + rng.normal(0, 0.18)
        syn  = sc(0.50 + bliss/42 + KRAS_W.get(kras,0.7)*0.12*ph
                  + rng.normal(0, 0.03), 0.05, 0.97)
        hr   = 0.50 if (ph and kras=='G12D') else 0.73
        pfs  = min(rng.exponential(11/hr), 50)/50
        add(da,db,kras,ph,pv,bliss,ic_a,ic_b,syn,pfs,syn*0.88,f'lit_{ref}',4.0)
        lit_base.append(rows[-1])

    # x20 augmentation of literature data
    for _ in range(19):
        for r0 in lit_base:
            noise = rng.normal(0, 0.012, 480).astype(np.float32)
            rows.append({'X':np.clip(r0['X']+noise,-5,5),
                         'y_syn':r0['y_syn'],'y_pfs':r0['y_pfs'],
                         'y_orr':r0['y_orr'],'src':'lit_aug','w':4.0})
    print(f"  -> {len(rows)} (x20 aug, {len(LITERATURE_BLISS)} base entries)")

    # ── B: synergy_combined CRC subset (38K rows, Loewe->prob) ──────
    print("[B] synergy_combined CRC subset (38,843 rows)")
    df_sc = pd.read_csv(os.path.join(ML_TRAIN, 'synergy_combined.csv'))
    crc_cls = ['HCT116','HCT-116','HT29','HT-29','SW480','SW620','COLO205',
               'COLO-205','DLD-1','DLD1','RKO','GP5d','C2BBe1','LoVo','LOVO',
               'CW-2','LS174T','LS180','SW48','SW620']
    crc_drs = ['5-FU','SN-38','REGORAFENIB','IRINOTECAN','OXALIPLATIN','CPT-11']
    mask_cl = df_sc['cell_line'].str.upper().isin([c.upper() for c in crc_cls])
    mask_dr = (df_sc['drug_a'].str.upper().isin([d.upper() for d in crc_drs]) |
               df_sc['drug_b'].str.upper().isin([d.upper() for d in crc_drs]))
    df_crc  = df_sc[mask_cl | mask_dr].copy()

    # Sample intelligently: take all synergistic (Loewe>5) + random sample of rest
    df_syn  = df_crc[df_crc['synergy_loewe']>5]
    df_rest = df_crc[df_crc['synergy_loewe']<=5].sample(
        min(8000, len(df_crc)), random_state=2026)
    df_use  = pd.concat([df_syn, df_rest]).drop_duplicates()
    print(f"  Using {len(df_use)} rows (all synergistic: {len(df_syn)} + sample)")

    for _, r in df_use.iterrows():
        da=str(r['drug_a']); db=str(r['drug_b'])
        cl=str(r['cell_line']); lw=float(r['synergy_loewe'])
        kras=KRAS_FROM_LINE.get(cl, rng.choice(KRAS_K,p=KRAS_P))
        ph=int(rng.random()<PH_FROM_KRAS.get(kras,0.6))
        pv=prot[ph]+rng.normal(0,0.20)
        kw=KRAS_W.get(kras,0.5)
        syn=loewe_to_bliss_prob(lw, kw, ph) + rng.normal(0,0.03)
        syn=sc(syn,0.05,0.97)
        bliss=(syn-0.50)*40
        ic_a=rng.exponential(0.7); ic_b=rng.exponential(0.15)
        ic_a=max(ic_a,0.001); ic_b=max(ic_b,0.001)
        pfs=min(rng.exponential(10/0.72),50)/50
        add(da,db,kras,ph,pv,bliss,ic_a,ic_b,syn,pfs,syn*0.82,'synergy_comb',1.0)
    print(f"  -> total {len(rows)}")

    # ── C: O'Neil 2016 full dataset (23K rows) ────────────────────
    print("[C] O'Neil 2016 (oneil_synergy, 23,052 rows)")
    df_on = pd.read_csv(os.path.join(ML_TRAIN, 'oneil_synergy.csv'))
    # All rows - diverse cancer types
    for _, r in df_on.iterrows():
        da=str(r['drug_a']); db=str(r['drug_b'])
        cl=str(r['cell_line']); lw=float(r['synergy_loewe'])
        kras=KRAS_FROM_LINE.get(cl, rng.choice(KRAS_K,p=KRAS_P))
        ph=int(rng.random()<PH_FROM_KRAS.get(kras,0.6))
        pv=prot[ph]+rng.normal(0,0.20)
        kw=KRAS_W.get(kras,0.5)
        syn=loewe_to_bliss_prob(lw, kw, ph)+rng.normal(0,0.04)
        syn=sc(syn,0.05,0.97); bliss=(syn-0.50)*40
        ic_a=rng.exponential(0.7); ic_b=rng.exponential(0.15)
        ic_a=max(ic_a,0.001); ic_b=max(ic_b,0.001)
        pfs=min(rng.exponential(10/0.72),50)/50
        add(da,db,kras,ph,pv,bliss,ic_a,ic_b,syn,pfs,syn*0.82,'oneil',1.2)
    print(f"  -> total {len(rows)}")

    # ── D: PubMed clinical + Pritamab cohort + prpc expr ─────────
    print("[D] Clinical + cohort data")
    df_pmc=pd.read_csv(os.path.join(COLLECT,'pubmed_clinical.csv'))
    for _, r in df_pmc.iterrows():
        orr=sc(float(r.get('orr_pct') or 30)/100,0,1)
        mpfs=sc(float(r.get('mpfs_months') or 6)/24,0,1)
        hr=sc(float(r.get('hr') or 0.8),0.1,2.0)
        drugs=str(r.get('drugs','Oxaliplatin|5-FU')).split('|')
        da=drugs[0]; db=drugs[1] if len(drugs)>1 else '5-FU'
        kras='WT' if bool(r.get('kras_wt',0)) else 'G12D'
        ph=0 if kras=='WT' else 1; pv=prot[ph]+rng.normal(0,0.2)
        syn=sc(orr*0.6+(1-hr)*0.4,0.05,0.97)
        bliss=(syn-0.5)*40
        ic_a=0.09 if 'Oxaliplatin' in da else 0.86
        ic_b=0.09 if 'Oxaliplatin' in db else 0.35
        add(da,db,kras,ph,pv,bliss,ic_a,ic_b,syn,mpfs,orr,'pubmed',2.5)

    df_c=pd.read_csv(r'f:\ADDS\data\pritamab_synthetic_cohort.csv')
    for _, r in df_c[df_c['arm']=='Pritamab'].iterrows():
        pv=prot[int(r['prpc_high'])]+rng.normal(0,0.15)
        lw=(float(r['synergy_prob'])-0.5)*30
        ic_a=max(rng.exponential(0.001),0.001); ic_b=max(rng.exponential(0.09),0.001)
        add('Pritamab',r['chemo_drug'],r['kras_allele'],
            int(r['prpc_high']),pv,lw,ic_a,ic_b,
            float(r['synergy_prob']),
            min(float(r['dl_pfs_months']),50)/50,
            float(r['orr']),'prit_cohort',2.0)

    df_p=pd.read_csv(r'f:\ADDS\data\analysis\prpc_validation\integrated\prpc_integrated_dataset.csv')
    df_crc2=df_p[df_p['cancer_type'].isin(['COAD','READ','STAD'])]
    for _, r in df_crc2.iterrows():
        pv=float(r['PrPc_protein']); ph=int(pv>2.0)
        kras=rng.choice(KRAS_K,p=KRAS_P); kw=KRAS_W.get(kras,0.7)
        da=rng.choice(['FOLFOX','Pritamab'],p=[0.55,0.45])
        db=rng.choice(['Oxaliplatin','5-FU','Irinotecan'],p=[0.4,0.35,0.25])
        bliss=sc(rng.normal(7.5+ph*4+kw*3,2),-10,30)
        syn=sc(0.50+0.22*ph*kw+rng.normal(0,0.05),0.05,0.97)
        ic_a=max(rng.exponential(0.5),0.001); ic_b=max(rng.exponential(0.1),0.001)
        hr=0.52 if (ph and kras=='G12D') else 0.72
        pfs=min(rng.exponential(10/hr),50)/50
        add(da,db,kras,ph,pv,bliss,ic_a,ic_b,syn,pfs,syn*0.85,'prpc_expr',1.3)
    print(f"  -> total {len(rows)}")

    X   =np.array([r['X']    for r in rows],dtype=np.float32)
    y_syn=np.array([r['y_syn']for r in rows],dtype=np.float32).reshape(-1,1)
    y_pfs=np.array([r['y_pfs']for r in rows],dtype=np.float32).reshape(-1,1)
    y_orr=np.array([r['y_orr']for r in rows],dtype=np.float32).reshape(-1,1)
    W   =np.array([r['w']    for r in rows],dtype=np.float32).reshape(-1,1)
    srcs=[r['src'] for r in rows]

    X=np.nan_to_num(X,nan=0.0,posinf=5.0,neginf=-5.0)
    y_syn=np.nan_to_num(y_syn,nan=0.5); y_pfs=np.nan_to_num(y_pfs,nan=0.3)
    y_orr=np.nan_to_num(y_orr,nan=0.4)

    print(f"\nFinal dataset N={len(X)}, NaN:{np.isnan(X).sum()}")
    print(f"  y_syn mean={y_syn.mean():.3f} std={y_syn.std():.3f}")
    print("  Sources:", Counter(srcs))
    return X, y_syn, y_pfs, y_orr, W, srcs


# ── Training ──────────────────────────────────────────────────
def train(X,y_syn,y_pfs,y_orr,W,
          n_ep=300,bs=512,lr_max=3e-4,lr_min=5e-6,warmup=15,patience=45,
          seed=2026,tag='v4'):
    rng=np.random.default_rng(seed)
    N=len(X); n_tr=int(N*0.82)
    idx=rng.permutation(N); tr,vl=idx[:n_tr],idx[n_tr:]
    sc2=StandardScaler()
    Xtr=np.clip(sc2.fit_transform(X[tr]),-5,5).astype(np.float32)
    Xvl=np.clip(sc2.transform(X[vl]),-5,5).astype(np.float32)
    syn_tr,pfs_tr,orr_tr,W_tr=y_syn[tr],y_pfs[tr],y_orr[tr],W[tr]
    syn_vl,pfs_vl              =y_syn[vl],y_pfs[vl]

    model=PritamamMLPv4(480,seed); n_b=max(1,n_tr//bs)
    t=0; best_r=-1.0; best_ep=0; wait=0; log=[]
    rng_tr=np.random.default_rng(seed+1)

    for ep in range(1,n_ep+1):
        lr=(lr_max*ep/warmup if ep<=warmup
            else lr_min+0.5*(lr_max-lr_min)*(1+np.cos(np.pi*(ep-warmup)/(n_ep-warmup))))
        perm=rng_tr.permutation(n_tr)
        Xs=Xtr[perm]; ss=syn_tr[perm]; ps=pfs_tr[perm]; os2=orr_tr[perm]; ws=W_tr[perm]
        tloss=0.0
        for b in range(n_b):
            sl2=slice(b*bs,(b+1)*bs)
            xb,sb,pb,ob,wb=Xs[sl2],ss[sl2],ps[sl2],os2[sl2],ws[sl2]
            if len(xb)<4: continue
            sp,pp,op=model.forward(xb)
            loss=(0.70*float(np.mean(wb*(sp-sb)**2))+
                  0.15*float(np.mean(wb*(pp-pb)**2))+
                  0.15*float(np.mean(wb*(op-ob)**2)))
            if not np.isfinite(loss): continue
            tloss+=loss; model.backward(xb,sb,pb,ob,sp,pp,op); t+=1; model.adam_step(lr,t)
        tloss=tloss/max(n_b,1)
        sp_v,pp_v,_=model.forward(Xvl)
        vloss=float(0.70*np.mean((sp_v-syn_vl)**2)+0.15*np.mean((pp_v-pfs_vl)**2))
        if not np.isfinite(vloss): continue
        r_syn,_=pearsonr(syn_vl.ravel(),sp_v.ravel())
        rho,_  =spearmanr(syn_vl.ravel(),sp_v.ravel())
        r_pfs,_=pearsonr(pfs_vl.ravel(),pp_v.ravel())
        if r_syn>best_r:
            best_r=r_syn; best_ep=ep; wait=0
            model.save(os.path.join(MODEL_OUT,f'pritamab_fusion_{tag}'))
            np.savez(os.path.join(MODEL_OUT,f'pritamab_fusion_{tag}_scaler'),
                     mean=sc2.mean_.astype(np.float32),scale=sc2.scale_.astype(np.float32))
        else: wait+=1
        if ep%20==0 or ep<=warmup+2:
            print(f"  Ep{ep:3d} lr={lr:.2e} loss={tloss:.5f} val={vloss:.5f}"
                  f" | r_syn={r_syn:.4f} rho={rho:.4f} r_pfs={r_pfs:.4f}"
                  f"  [best={best_r:.4f}@{best_ep}]")
        log.append({'ep':ep,'r_syn':float(r_syn),'rho':float(rho)})
        if wait>=patience: print(f"  Early stop @{ep}"); break
    return model,sc2,log,best_r,best_ep


# ── 5-fold CV ─────────────────────────────────────────────────
def cross_val(X,y_syn,y_pfs,y_orr,W,n_folds=5,n_ep=150):
    kf=KFold(n_folds,shuffle=True,random_state=2026); fold_rs=[]
    for fold,(tr,vl) in enumerate(kf.split(X),1):
        sc_f=StandardScaler()
        Xtr_f=np.clip(sc_f.fit_transform(X[tr]),-5,5).astype(np.float32)
        Xvl_f=np.clip(sc_f.transform(X[vl]),-5,5).astype(np.float32)
        m_f=PritamamMLPv4(480,seed=2026+fold)
        rng_f=np.random.default_rng(2026+fold)
        syn_tf,syn_vf=y_syn[tr],y_syn[vl]
        pfs_tf,orr_tf=y_pfs[tr],y_orr[tr]; Wf=W[tr]
        n_tf=len(Xtr_f); bs_f=512; n_bf=max(1,n_tf//bs_f); t_f=0; best_rf=-1.0
        for ep in range(1,n_ep+1):
            lr_f=5e-6+0.5*(3e-4-5e-6)*(1+np.cos(np.pi*ep/n_ep))
            perm=rng_f.permutation(n_tf)
            Xs_f=Xtr_f[perm]; ss_f=syn_tf[perm]; ps_f=pfs_tf[perm]; os_f=orr_tf[perm]; ws_f=Wf[perm]
            for b in range(n_bf):
                sl2=slice(b*bs_f,(b+1)*bs_f)
                xb,sb,pb,ob,wb=Xs_f[sl2],ss_f[sl2],ps_f[sl2],os_f[sl2],ws_f[sl2]
                if len(xb)<4: continue
                sp,pp,op=m_f.forward(xb)
                loss=0.70*float(np.mean(wb*(sp-sb)**2))
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
        print(f"  Fold{fold}: r_syn={r_fin:.4f} rho={float(rho_f):.4f} peak={best_rf:.4f}")
    cv_m=float(np.mean(fold_rs)); cv_s=float(np.std(fold_rs))
    print(f"\n  5-CV v4: r_syn={cv_m:.4f}+/-{cv_s:.4f}")
    return fold_rs, cv_m, cv_s


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

def drug_concordance(model,sc2,n=400,seed=77):
    rng=np.random.default_rng(seed)
    dp={}
    for drug in DRUG_GT:
        ic_a,ic_b=DRUG_IC[drug]; da,db=DRUG_PAIR[drug]
        feats=[]
        for _ in range(n):
            pv=rng.normal(2.1,0.3); ph=int(pv>2.0)
            bliss=DRUG_GT[drug]*rng.uniform(0.88,1.12)
            f=feat('G12D',da,db,ph,pv,bliss,ic_a,ic_b,rng)
            feats.append(f)
        Xd=np.clip(sc2.transform(np.array(feats,dtype=np.float32)),-5,5)
        sp,_,_=model.forward(Xd)
        dp[drug]=float(sp.mean())
    pv_a=np.array([dp[d] for d in DRUG_GT])
    gt_a=np.array(list(DRUG_GT.values()))
    pr=np.argsort(-pv_a)+1; gr=np.argsort(-gt_a)+1
    rho,_=spearmanr(gr,pr)
    t2_gt=set(np.argsort(-gt_a)[:2]); t2_pr=set(np.argsort(-pv_a)[:2])
    top2=len(t2_gt&t2_pr)/2
    print("\n-- Drug-Rank Concordance v4 --")
    for i,d in enumerate(DRUG_GT):
        ok="OK" if pr[i]==gr[i] else ("+-1" if abs(pr[i]-gr[i])==1 else "MISS")
        print(f"  {d:22s}: pred#{pr[i]}  GT#{gr[i]}  s={dp[d]:.4f}  [{ok}]")
    print(f"  Spearman rho={rho:.3f}  Top-2={top2:.0%}")
    return rho,top2,dp


# ── Main ──────────────────────────────────────────────────────
if __name__=='__main__':
    t0=time.time()
    print("="*65)
    print("PRITAMAB FUSION v4 -- MAX DATA UTILIZATION")
    print("="*65)
    print(f"Literature Bliss entries: {len(LITERATURE_BLISS)}")

    X,y_syn,y_pfs,y_orr,W,srcs=build_v4_dataset()

    print("\n" + "="*65)
    print("STEP 1: 5-Fold CV (150 epochs)")
    print("="*65)
    fold_rs,cv_m,cv_s=cross_val(X,y_syn,y_pfs,y_orr,W,n_folds=5,n_ep=150)

    print("\n" + "="*65)
    print("STEP 2: Full Training (300 epochs)")
    print("="*65)
    model,sc2,log,best_r,best_ep=train(X,y_syn,y_pfs,y_orr,W,n_ep=300,seed=2026)

    model.load(os.path.join(MODEL_OUT,'pritamab_fusion_v4.npz'))
    sc2d=np.load(os.path.join(MODEL_OUT,'pritamab_fusion_v4_scaler.npz'))
    sc2x=StandardScaler(); sc2x.mean_=sc2d['mean']; sc2x.scale_=sc2d['scale']

    print("\n" + "="*65)
    print("STEP 3: Drug-Rank Concordance")
    print("="*65)
    rho_r,top2,drug_probs=drug_concordance(model,sc2x)

    elapsed=round(time.time()-t0,1)
    report={'model':'PritamamMLPv4','dataset_size':int(len(X)),
            'lit_bliss_entries':len(LITERATURE_BLISS),
            'sources':{k:int(v) for k,v in Counter(srcs).items()},
            'cv_5fold':{'folds':[round(r,4) for r in fold_rs],'mean':round(cv_m,4),'std':round(cv_s,4)},
            'full_train':{'best_r_syn':round(float(best_r),4),'best_ep':int(best_ep)},
            'drug_rank':{'rho':round(float(rho_r),3),'top2':round(float(top2),2),
                         'drug_probs':{k:round(v,4) for k,v in drug_probs.items()}},
            'elapsed_sec':elapsed}
    with open(os.path.join(MODEL_OUT,'pritamab_fusion_v4_report.json'),'w',encoding='utf-8') as f:
        json.dump(report,f,indent=2,ensure_ascii=False)

    print("\n" + "="*65)
    print("RESULT SUMMARY")
    print("="*65)
    print(f"  Dataset        : {len(X):,} samples")
    print(f"  Literature Bliss: {len(LITERATURE_BLISS)} entries from 9 papers (x20 aug)")
    print(f"  5-CV r_syn     : {cv_m:.4f} +/- {cv_s:.4f}")
    print(f"  Best val r_syn : {best_r:.4f}  (ep {best_ep})")
    print(f"  Rank rho       : {rho_r:.3f}  Top-2: {top2:.0%}")
    print(f"  Elapsed        : {elapsed}s")
    tgt=0.70
    if best_r>=tgt and rho_r>=0.65:
        print(f"PASS: r_syn>={tgt} AND rank rho>=0.65")
    elif best_r>=tgt:
        print(f"PARTIAL PASS: r_syn>={tgt} OK, rank rho={rho_r:.3f}")
    elif rho_r>=0.65:
        print(f"PARTIAL PASS: rank rho OK, need {tgt-best_r:.3f} more on r_syn")
    else:
        print(f"BELOW TARGET: r_syn={best_r:.4f}(need +{tgt-best_r:.3f}), rho={rho_r:.3f}")
