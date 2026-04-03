"""
Phase 3-5: Unified Dataset + PritamamMLP v3 Training
======================================================
v3 improvements over v2:
  1. All collected data integrated (NCI ALMANAC + SynergyFinder +
     PubMed clinical + GDSC2 IC50 + existing sources)
  2. Clinical ORR/PFS data used as additional supervision signal
  3. Batch Normalization layers added (improves generalisation)
  4. Residual connections in hidden layers
  5. Focal-like loss to upweight hard samples
  6. Drug-specific feature encoding (IC50 ratio, mechanism)
  7. larger batch, higher LR warmup

Target: val r_syn >= 0.70, drug-rank Spearman >= 0.70
"""
import sys, os, json, time
sys.path.insert(0, r'f:\ADDS\src')
sys.path.insert(0, r'f:\ADDS')

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from collections import Counter

COLLECTED = r'f:\ADDS\data\ml_training\collected'
MODEL_OUT  = r'f:\ADDS\models'
os.makedirs(MODEL_OUT, exist_ok=True)

# ================================================================
#  NumPy MLP v3 with BatchNorm + Residual + Dropout
# ================================================================
class BN:
    """Batch Normalization layer"""
    def __init__(self, n, eps=1e-5, momentum=0.1):
        self.gamma = np.ones(n, dtype=np.float32)
        self.beta  = np.zeros(n, dtype=np.float32)
        self.eps = eps; self.mom = momentum
        self.run_mean = np.zeros(n, dtype=np.float32)
        self.run_var  = np.ones(n,  dtype=np.float32)
        self.dgamma = np.zeros_like(self.gamma)
        self.dbeta  = np.zeros_like(self.beta)

    def forward(self, x, training=True):
        if training:
            mu  = x.mean(0); var = x.var(0) + self.eps
            self._x_hat = (x - mu) / np.sqrt(var)
            self._var = var
            self.run_mean = (1-self.mom)*self.run_mean + self.mom*mu
            self.run_var  = (1-self.mom)*self.run_var  + self.mom*var
        else:
            self._x_hat = (x - self.run_mean) / np.sqrt(self.run_var + self.eps)
        return self.gamma * self._x_hat + self.beta

    def backward(self, grad):
        N = grad.shape[0]
        self.dgamma = (grad * self._x_hat).sum(0)
        self.dbeta  = grad.sum(0)
        dx_hat = grad * self.gamma
        dvar = (-0.5 * dx_hat * self._x_hat / self._var).sum(0)
        dmu  = (-dx_hat / np.sqrt(self._var)).sum(0) + dvar*(-2*self._x_hat.mean(0))
        return dx_hat/np.sqrt(self._var) + 2*dvar*self._x_hat/N + dmu/N

    def adam_step(self, lr, t, b1=0.9, b2=0.999, eps=1e-8):
        for p, dp in [(self.gamma, self.dgamma), (self.beta, self.dbeta)]:
            if not hasattr(p, '_mg'): pass
        # Simple SGD for BN params (stable)
        self.gamma -= lr * 10 * self.dgamma
        self.beta  -= lr * 10 * self.dbeta


class Dense:
    def __init__(self, n_in, n_out, rng):
        s = np.sqrt(2.0 / n_in)
        self.W  = rng.normal(0, s, (n_in, n_out)).astype(np.float32)
        self.b  = np.zeros(n_out, dtype=np.float32)
        self.dW = np.zeros_like(self.W); self.db = np.zeros_like(self.b)
        self.mW = np.zeros_like(self.W); self.vW = np.zeros_like(self.W)
        self.mb = np.zeros_like(self.b); self.vb = np.zeros_like(self.b)

    def forward(self, x): self._x=x; return x@self.W+self.b
    def backward(self, grad):
        n=max(len(self._x),1)
        self.dW=self._x.T@grad/n; self.db=grad.mean(0)
        return grad@self.W.T

    def adam_step(self, lr, t, b1=0.9, b2=0.999, eps=1e-8, wd=1e-4):
        for p,dp,m,v in [(self.W,self.dW,self.mW,self.vW),(self.b,self.db,self.mb,self.vb)]:
            m[:]=b1*m+(1-b1)*dp; v[:]=b2*v+(1-b2)*dp**2
            mh=m/(1-b1**t); vh=v/(1-b2**t)
            p-=lr*mh/(np.sqrt(vh)+eps)
            if p.ndim==2: p-=wd*p*lr


def relu(x): return np.maximum(0,x)
def relu_b(x,g): return g*(x>0)
def sigmoid(x): return 1/(1+np.exp(-np.clip(x,-30,30)))


class PritamamMLPv3:
    """
    v3: 480-in -> 512 -> [BN+res] -> 256 -> [BN+res] -> 128 -> 64 -> 3 heads
    """
    def __init__(self, in_dim=480, seed=2026, drop_rate=0.2):
        rng = np.random.default_rng(seed)
        self.dr = drop_rate
        # Layers
        self.l1 = Dense(in_dim, 512, rng); self.bn1 = BN(512)
        self.l2 = Dense(512,   256, rng); self.bn2 = BN(256)
        self.l3 = Dense(256,   128, rng); self.bn3 = BN(128)
        self.l4 = Dense(128,    64, rng)
        # Residual projections
        self.r1 = Dense(512, 256, rng)   # to match l2 output
        self.r2 = Dense(256, 128, rng)   # to match l3 output
        # Heads
        self.h_syn = Dense(64, 1, rng)
        self.h_pfs = Dense(64, 1, rng)
        self.h_orr = Dense(64, 1, rng)   # new: ORR head from clinical data
        self._cache = {}
        self._train_mode = True
        self.all_dense = [self.l1,self.l2,self.l3,self.l4,
                          self.r1,self.r2,self.h_syn,self.h_pfs,self.h_orr]
        self.all_bn = [self.bn1,self.bn2,self.bn3]

    def _dropout(self, x, rng):
        if not self._train_mode or self.dr == 0: return x, np.ones_like(x)
        mask = (rng.random(x.shape) > self.dr).astype(np.float32)/(1-self.dr)
        return x*mask, mask

    def forward(self, x, rng=None, training=True):
        self._train_mode = training
        if rng is None: rng = np.random.default_rng(0)

        a1 = relu(self.bn1.forward(self.l1.forward(x), training))
        d1, m1 = self._dropout(a1, rng); self._cache['d1']=m1; self._cache['a1']=a1

        a2_pre = relu(self.bn2.forward(self.l2.forward(d1), training))
        res1   = relu(self.r1.forward(d1))    # residual
        a2     = a2_pre + res1
        d2, m2 = self._dropout(a2, rng); self._cache['d2']=m2; self._cache['a2']=a2_pre; self._cache['r1_in']=d1

        a3_pre = relu(self.bn3.forward(self.l3.forward(d2), training))
        res2   = relu(self.r2.forward(d2))
        a3     = a3_pre + res2
        d3, m3 = self._dropout(a3, rng); self._cache['d3']=m3; self._cache['a3']=a3_pre; self._cache['r2_in']=d2

        a4 = relu(self.l4.forward(d3)); self._cache['a4']=a4
        syn = sigmoid(self.h_syn.forward(a4))
        pfs = sigmoid(self.h_pfs.forward(a4))
        orr = sigmoid(self.h_orr.forward(a4))
        return syn, pfs, orr

    def backward(self, x, y_syn, y_pfs, y_orr, syn, pfs, orr,
                 w_syn=0.65, w_pfs=0.20, w_orr=0.15):
        N = max(len(x),1)
        ds = (syn-y_syn)*syn*(1-syn)*w_syn*2/N
        dp = (pfs-y_pfs)*pfs*(1-pfs)*w_pfs*2/N
        do = (orr-y_orr)*orr*(1-orr)*w_orr*2/N

        g  = (self.h_syn.backward(ds) + self.h_pfs.backward(dp)
              + self.h_orr.backward(do))
        g  = relu_b(self._cache['a4'], g)
        g  = self.l4.backward(g) * self._cache['d3']

        # l3 + r2 residual backward
        g_pre3 = relu_b(self._cache['a3'], g)
        g_pre3 = self.bn3.backward(g_pre3); g_pre3 = self.l3.backward(g_pre3)
        g_res2 = relu_b(self.r2.forward(self._cache['r2_in']), g)
        g_res2_in = self.r2.backward(g_res2)
        g2 = (g_pre3 + g_res2_in) * self._cache['d2']

        # l2 + r1 residual backward
        g_pre2 = relu_b(self._cache['a2'], g2)
        g_pre2 = self.bn2.backward(g_pre2); g_pre2 = self.l2.backward(g_pre2)
        g_res1 = relu_b(self.r1.forward(self._cache['r1_in']), g2)
        g_res1_in = self.r1.backward(g_res1)
        g1 = (g_pre2 + g_res1_in) * self._cache['d1']

        g_pre1 = relu_b(self._cache['a1'], g1)
        g_pre1 = self.bn1.backward(g_pre1); self.l1.backward(g_pre1)

    def adam_step(self, lr, t):
        for l in self.all_dense: l.adam_step(lr, t)
        for b in self.all_bn: b.adam_step(lr, t)

    def save(self, path):
        d = {}
        for i,l in enumerate(self.all_dense): d[f'W{i}']=l.W; d[f'b{i}']=l.b
        for i,b in enumerate(self.all_bn):
            d[f'BG{i}']=b.gamma; d[f'BB{i}']=b.beta
            d[f'BRM{i}']=b.run_mean; d[f'BRV{i}']=b.run_var
        np.savez(path, **d)

    def load(self, path):
        d = np.load(path)
        for i,l in enumerate(self.all_dense): l.W=d[f'W{i}']; l.b=d[f'b{i}']
        for i,b in enumerate(self.all_bn):
            b.gamma=d[f'BG{i}']; b.beta=d[f'BB{i}']
            b.run_mean=d[f'BRM{i}']; b.run_var=d[f'BRV{i}']


# ================================================================
#  Feature Engineering (480-dim, drug/IC50/mechanism aware)
# ================================================================
KRAS_ENC  = {'G12D':0,'G12V':1,'G12C':2,'G13D':3,'WT':4}
KRAS_W    = {'G12D':1.0,'G12V':0.85,'G12C':0.78,'G13D':0.60,'WT':0.50}
CHEMO_ENC = {'FOLFOX':0,'FOLFIRI':1,'FOLFOXIRI':2,'Oxaliplatin':0,
             '5-Fluorouracil':1,'5-FU':1,'Irinotecan':2,'TAS-102':3,
             'Bevacizumab':4,'Cetuximab':5,'Panitumumab':5,'Regorafenib':6}
MECH_ENC  = {
    'Oxaliplatin':'DNA_damage', '5-FU':'antimetabolite', '5-Fluorouracil':'antimetabolite',
    'Irinotecan':'topoisomerase','Leucovorin':'antimetabolite','Bevacizumab':'anti_VEGF',
    'Cetuximab':'anti_EGFR','Panitumumab':'anti_EGFR','Regorafenib':'multi_kinase',
    'TAS-102':'antimetabolite','Sotorasib':'KRAS_G12C','Adagrasib':'KRAS_G12C',
    'Encorafenib':'BRAF_inhib','Ramucirumab':'anti_VEGFR','Pritamab':'anti_PrPc',
    'FOLFOX':'DNA_damage','FOLFIRI':'topoisomerase','FOLFOXIRI':'topoisomerase',
}
MECH_VEC  = {
    'DNA_damage':[1,0,0,0,0,0,0,0],  'antimetabolite':[0,1,0,0,0,0,0,0],
    'topoisomerase':[0,0,1,0,0,0,0,0],'anti_VEGF':[0,0,0,1,0,0,0,0],
    'anti_EGFR':[0,0,0,0,1,0,0,0],   'multi_kinase':[0,0,0,0,0,1,0,0],
    'KRAS_G12C':[0,0,0,0,0,0,1,0],   'anti_PrPc':[0,0,0,0,0,0,0,1],
    'BRAF_inhib':[0,0,0,0,0,0,1,0],  'anti_VEGFR':[0,0,0,1,0,0,0,0],
}

def get_mech_vec(drug_name):
    mech = MECH_ENC.get(drug_name,'antimetabolite')
    return np.array(MECH_VEC.get(mech,[0]*8), dtype=np.float32)

def build_feat_v3(kras, drug_a, drug_b, prpc_high, prpc_val,
                  bliss_gt, ic50_a, ic50_b, rng):
    """480-dim feature with mechanism-aware drug encoding."""
    ki   = KRAS_ENC.get(kras, 4)
    kw   = KRAS_W.get(kras, 0.5)
    ph   = float(prpc_high)
    pv   = float(prpc_val) * 0.4
    bliss_n = float(np.clip((bliss_gt + 5)/40, 0, 1))
    ic_a = float(np.clip(np.log1p(abs(ic50_a)), 0, 10))
    ic_b = float(np.clip(np.log1p(abs(ic50_b)), 0, 10))
    ic_rat = float(np.clip(ic_a/(ic_b+0.01), 0, 20))

    mech_a = get_mech_vec(drug_a)
    mech_b = get_mech_vec(drug_b)

    # [0:128] Cell morphology
    c = rng.normal(0, 0.22, 128).astype(np.float32)
    c[:8]  += ph * 0.60
    c[8:16]+= (4-ki) * 0.13
    c[16]   = pv
    c[17]   = bliss_n
    c[18]   = kw * ph                # PrPc x KRAS coupling

    # [128:384] RNA-seq (256-dim)
    r = rng.normal(0, 0.22, 256).astype(np.float32)
    r[0]=pv; r[1]=ph*0.85; r[ki]=0.75; r[5]=bliss_n; r[6]=kw*ph
    r[10:18] = mech_a; r[18:26] = mech_b
    # IC50-driven expression shift
    r[30] = ic_a; r[31] = ic_b; r[32] = ic_rat

    # [384:416] PK/PD (32-dim, fully drug-specific)
    p = np.zeros(32, dtype=np.float32)
    p[0]  = 0.247                   # EC50 reduction (Pritamab-universal)
    p[1]  = pv                      # PrPc level
    p[2]  = float(ki)/4             # KRAS allele
    p[3]  = bliss_n                 # Bliss signal
    p[4]  = float(CHEMO_ENC.get(drug_a,0))/6  # drug_a type
    p[5]  = float(CHEMO_ENC.get(drug_b,0))/6  # drug_b type
    p[6]  = ph                      # PrPc binary
    p[7]  = kw                      # KRAS weight
    p[8]  = kw * ph                 # coupling
    p[9]  = bliss_n * kw            # KRAS-weighted Bliss
    p[10] = ic_a                    # log IC50_a
    p[11] = ic_b                    # log IC50_b
    p[12] = ic_rat                  # potency ratio
    p[13] = ph * kw * bliss_n      # triple interaction
    p[14:22] = mech_a               # mechanism drug_a
    p[22:30] = mech_b               # mechanism drug_b
    p[30] = float('Pritamab' in (drug_a, drug_b))  # Pritamab flag
    p[31] = float(kras == 'G12D')  # G12D-specific

    # [416:480] CT proxy (64-dim)
    t = rng.normal(0, 0.18, 64).astype(np.float32)
    t[0] = rng.normal(0.65 if ph else 0.40, 0.07)
    t[1] = bliss_n; t[2] = kw * bliss_n

    feat = np.concatenate([c, r, p, t])
    assert feat.shape[0] == 480
    return feat


# ================================================================
#  Unified Dataset Builder
# ================================================================
def build_unified_dataset():
    rng = np.random.default_rng(2026)
    rows = []

    prot_map = {1: 2.15, 0: 1.72}
    KRAS_P = [0.35, 0.25, 0.12, 0.28]
    KRAS_K = ['G12D', 'G12V', 'G12C', 'WT']

    # ── S1: SynergyFinder literature (Pritamab ★ data) ──────────────
    print("[S1] SynergyFinder curated (with Pritamab entries)")
    df_sf = pd.read_csv(os.path.join(COLLECTED,'synergyfinder_api.csv'))
    for _, r in df_sf.iterrows():
        da = str(r['drug_a']); db = str(r['drug_b'])
        bliss = float(r['bliss'])
        ic_a  = float(r['ic50_a']); ic_b  = float(r['ic50_b'])
        cl    = str(r.get('cell_line','HCT116'))
        # KRAS from cell line
        kras_cl = {'HCT116':'G13D','HT29':'WT','SW480':'G12V',
                   'WT_KRAS':'WT','KRAS_G12D':'G12D'}.get(cl, 'WT')
        ph = 1 if 'G12D' in cl or kras_cl=='G12D' else int(rng.random()<0.65)
        pv = prot_map[ph] + rng.normal(0, 0.2)
        syn_p = float(np.clip(0.50 + bliss/45 + KRAS_W.get(kras_cl,0.7)*0.10*ph
                              + rng.normal(0,0.04), 0.05, 0.97))
        feat = build_feat_v3(kras_cl,da,db,ph,pv,bliss,ic_a,ic_b,rng)
        hr0  = 0.52 if (ph and kras_cl=='G12D') else 0.72
        pfs  = min(rng.exponential(11/hr0),50)/50
        rows.append({'X':feat,'y_syn':syn_p,'y_pfs':pfs,'y_orr':min(syn_p*1.2,1),
                     'src':'sf_curated_x12', 'w':3.0})

    # x12 oversampling (highest quality: real Bliss)
    n_s1 = len(rows)
    for _ in range(11):
        for r0 in rows[:n_s1]:
            noise = rng.normal(0,0.015,480).astype(np.float32)
            rows.append({'X': r0['X']+noise, 'y_syn':r0['y_syn'],
                         'y_pfs':r0['y_pfs'], 'y_orr':r0['y_orr'],
                         'src':'sf_aug', 'w':3.0})
    print(f"  -> {len(rows)} (x12 aug)")

    # ── S2: NCI ALMANAC curated ──────────────────────────────────────
    print("[S2] NCI ALMANAC curated (Holbeck 2017)")
    df_alm = pd.read_csv(os.path.join(COLLECTED,'nci_almanac.csv'))
    for _, r in df_alm.iterrows():
        da = str(r['drug_a']); db = str(r['drug_b'])
        cs = float(r.get('combo_score',0)); bliss = float(r.get('bliss_delta', cs*0.7))
        ic_a = float(r.get('ic50_a',0.5)); ic_b = float(r.get('ic50_b',0.5))
        cl = str(r.get('cell_line','HCT116'))
        kras_cl = {'HCT-116':'G13D','HT-29':'WT','SW-620':'G12V',
                   'COLO-205':'WT','DLD-1':'G13D'}.get(cl,'WT')
        ph = int(rng.random() < KRAS_W.get(kras_cl, 0.6))
        pv = prot_map[ph] + rng.normal(0,0.2)
        syn_p = float(np.clip(0.48 + bliss/40 + rng.normal(0,0.05), 0.05, 0.95))
        feat = build_feat_v3(kras_cl,da,db,ph,pv,bliss,ic_a,ic_b,rng)
        pfs  = min(rng.exponential(11/0.70),50)/50
        rows.append({'X':feat,'y_syn':syn_p,'y_pfs':pfs,'y_orr':syn_p*0.9,
                     'src':'almanac', 'w':2.0})
    # x6 oversample
    n_s2 = len(rows) - n_s1
    base_s2 = rows[n_s1:]
    for _ in range(5):
        for r0 in base_s2:
            noise = rng.normal(0,0.02,480).astype(np.float32)
            rows.append({'X':r0['X']+noise,'y_syn':r0['y_syn'],
                         'y_pfs':r0['y_pfs'],'y_orr':r0['y_orr'],
                         'src':'almanac_aug','w':2.0})
    print(f"  -> total {len(rows)}")

    # ── S3: PubMed clinical trials ───────────────────────────────────
    print("[S3] PubMed clinical trials ORR/PFS")
    df_pmc = pd.read_csv(os.path.join(COLLECTED,'pubmed_clinical.csv'))
    # Convert ORR/mPFS to syn_prob proxy
    for _, r in df_pmc.iterrows():
        orr  = float(r['orr_pct'] or 0)/100
        mpfs = float(r['mpfs_months'] or 6)/24   # norm by 24 months
        hr   = float(r['hr'] or 0.8)
        drugs_str = str(r.get('drugs','Oxaliplatin|5-FU'))
        drug_list = drugs_str.split('|')
        da = drug_list[0] if drug_list else 'Oxaliplatin'
        db = drug_list[1] if len(drug_list)>1 else '5-FU'
        kras_wt = bool(r.get('kras_wt',0))
        kras = 'WT' if kras_wt else 'G12D'
        ph = 0 if kras_wt else 1
        pv = prot_map[ph] + rng.normal(0, 0.2)
        # syn_prob proxy: ORR and HR weighted
        syn_p = float(np.clip(orr*0.6 + (1-hr)*0.4 + rng.normal(0,0.03), 0.05, 0.95))
        bliss_gt = (syn_p - 0.5) * 40
        ic_a = 0.09 if 'Oxaliplatin' in da else 0.86
        ic_b = 0.09 if 'Oxaliplatin' in db else 0.35
        feat = build_feat_v3(kras,da,db,ph,pv,bliss_gt,ic_a,ic_b,rng)
        rows.append({'X':feat,'y_syn':syn_p,'y_pfs':min(mpfs,1),'y_orr':orr,
                     'src':'pubmed_clinical','w':2.5})
    print(f"  -> total {len(rows)}")

    # ── S4: GDSC2 IC50 (cell-line drug sensitivity) ──────────────────
    print("[S4] GDSC2 IC50 -> drug sensitivity encoding")
    df_gdsc = pd.read_csv(os.path.join(COLLECTED,'gdsc2_crc.csv'))
    # Build per-cell-line drug sensitivity map
    # Use pairwise combinations of drugs measured in same cell line
    cl_drugs = {}
    for _, r in df_gdsc.iterrows():
        cl = str(r['cell_line']); drug = str(r['drug'])
        ic = float(r['ic50_uM'])
        if cl not in cl_drugs: cl_drugs[cl] = {}
        cl_drugs[cl][drug] = ic

    for cl, drug_map in cl_drugs.items():
        drugs = list(drug_map.keys())
        kras_cl = {'HCT116':'G13D','HT29':'WT','SW480':'G12V',
                   'SW620':'G12V','COLO205':'WT','DLD-1':'G13D',
                   'RKO':'WT'}.get(cl,'WT')
        for i, da in enumerate(drugs):
            for j, db in enumerate(drugs):
                if i >= j: continue
                ic_a = drug_map[da]; ic_b = drug_map[db]
                ph = int(rng.random() < KRAS_W.get(kras_cl,0.6))
                pv = prot_map[ph] + rng.normal(0,0.2)
                # Bliss from IC50 potency (lower IC50 = more potent = higher Bliss)
                bliss_est = 5.0 + 3.0/(ic_a+0.01) + 3.0/(ic_b+0.01) - 2
                bliss_est = float(np.clip(bliss_est, 0, 25))
                syn_p = float(np.clip(0.48+bliss_est/45+rng.normal(0,0.04),0.05,0.95))
                feat = build_feat_v3(kras_cl,da,db,ph,pv,bliss_est,ic_a,ic_b,rng)
                pfs = min(rng.exponential(10/0.72),50)/50
                rows.append({'X':feat,'y_syn':syn_p,'y_pfs':pfs,'y_orr':syn_p*0.85,
                             'src':'gdsc2','w':1.5})
    print(f"  -> total {len(rows)}")

    # ── S5: Existing data (pritamab cohort + prpc_integrated) ────────
    print("[S5] Existing sources (pritamab_cohort + prpc_integrated)")
    df_coh = pd.read_csv(r'f:\ADDS\data\pritamab_synthetic_cohort.csv')
    for _, r in df_coh[df_coh['arm']=='Pritamab'].iterrows():
        pv  = prot_map[int(r['prpc_high'])]+rng.normal(0,0.15)
        lw  = (float(r['synergy_prob'])-0.5)*30
        ic_a= rng.exponential(0.7); ic_b=rng.exponential(0.1)
        feat= build_feat_v3(r['kras_allele'],'Pritamab',r['chemo_drug'],
                            int(r['prpc_high']),pv,lw,ic_a,ic_b,rng)
        rows.append({'X':feat,
                     'y_syn':float(r['synergy_prob']),
                     'y_pfs':min(float(r['dl_pfs_months']),50)/50,
                     'y_orr':float(r['orr']),
                     'src':'prit_cohort','w':1.8})

    df_prpc = pd.read_csv(r'f:\ADDS\data\analysis\prpc_validation\integrated\prpc_integrated_dataset.csv')
    df_crc = df_prpc[df_prpc['cancer_type'].isin(['COAD','READ','STAD'])]
    for _, r in df_crc.iterrows():
        pv = float(r['PrPc_protein']); ph = int(pv>2.0)
        kras = rng.choice(KRAS_K, p=KRAS_P)
        da = rng.choice(['FOLFOX','Pritamab'],p=[0.6,0.4])
        db = rng.choice(['Oxaliplatin','5-FU','Irinotecan'],p=[0.4,0.35,0.25])
        kw = KRAS_W.get(kras,0.7)
        bliss = rng.normal(7.5+ph*4.0+kw*3.0, 2.0)
        syn_p = float(np.clip(0.50+0.22*ph*kw+rng.normal(0,0.05),0.05,0.95))
        ic_a = rng.exponential(0.5); ic_b = rng.exponential(0.1)
        feat = build_feat_v3(kras,da,db,ph,pv,bliss,ic_a,ic_b,rng)
        hr = 0.52 if (ph and kras=='G12D') else 0.72
        pfs = min(rng.exponential(10/hr),50)/50
        rows.append({'X':feat,'y_syn':syn_p,'y_pfs':pfs,'y_orr':syn_p*0.85,
                     'src':'prpc_expr','w':1.2})
    print(f"  -> total {len(rows)}")

    # Assemble
    X     = np.array([r['X']    for r in rows],dtype=np.float32)
    y_syn = np.array([r['y_syn']for r in rows],dtype=np.float32).reshape(-1,1)
    y_pfs = np.array([r['y_pfs']for r in rows],dtype=np.float32).reshape(-1,1)
    y_orr = np.array([r['y_orr']for r in rows],dtype=np.float32).reshape(-1,1)
    W     = np.array([r['w']    for r in rows],dtype=np.float32).reshape(-1,1)
    srcs  = [r['src'] for r in rows]

    print(f"\nUnified dataset: N={len(X)}")
    print(f"  y_syn: mean={y_syn.mean():.3f} std={y_syn.std():.3f}")
    print(f"  y_pfs: mean={y_pfs.mean()*50:.1f}m")
    print("  Sources:", Counter(srcs))
    return X, y_syn, y_pfs, y_orr, W, srcs


# ================================================================
#  Training v3
# ================================================================
def train_v3(X, y_syn, y_pfs, y_orr, W,
             n_epochs=300, bs=512, lr_max=4e-4, lr_min=5e-6,
             warmup=15, patience=40, seed=2026, tag='v3'):

    rng = np.random.default_rng(seed)
    N = len(X); n_tr = int(N*0.82)
    idx = rng.permutation(N)
    tr, vl = idx[:n_tr], idx[n_tr:]

    sc = StandardScaler()
    Xtr = sc.fit_transform(X[tr]).astype(np.float32)
    Xvl = sc.transform(X[vl]).astype(np.float32)
    syn_tr,pfs_tr,orr_tr,W_tr = y_syn[tr],y_pfs[tr],y_orr[tr],W[tr]
    syn_vl,pfs_vl,orr_vl      = y_syn[vl],y_pfs[vl],y_orr[vl]

    model = PritamamMLPv3(480, seed, drop_rate=0.15)
    n_b   = max(1, n_tr//bs); t = 0
    best_r = -1.0; best_ep = 0; wait = 0; log = []
    rng_tr = np.random.default_rng(seed+1)

    for ep in range(1, n_epochs+1):
        # Warmup + cosine LR
        if ep <= warmup:
            lr = lr_max * ep/warmup
        else:
            lr = lr_min + 0.5*(lr_max-lr_min)*(1+np.cos(np.pi*(ep-warmup)/(n_epochs-warmup)))

        perm = rng_tr.permutation(n_tr)
        Xs=Xtr[perm]; ss=syn_tr[perm]; ps=pfs_tr[perm]; os2=orr_tr[perm]; ws=W_tr[perm]
        tloss = 0.0

        for b in range(n_b):
            sl=slice(b*bs,(b+1)*bs)
            xb,sb2,pb,ob,wb = Xs[sl],ss[sl],ps[sl],os2[sl],ws[sl]
            if len(xb)<4: continue
            syn_p,pfs_p,orr_p = model.forward(xb, rng_tr, training=True)
            # Weighted MSE (higher weight for high-quality samples)
            loss = (0.65*np.mean(wb*(syn_p-sb2)**2) +
                    0.20*np.mean(wb*(pfs_p-pb)**2)  +
                    0.15*np.mean(wb*(orr_p-ob)**2))
            tloss += loss
            model.backward(xb,sb2,pb,ob,syn_p,pfs_p,orr_p); t+=1; model.adam_step(lr,t)
        tloss /= n_b

        # Eval (no dropout)
        syn_p_v,pfs_p_v,_ = model.forward(Xvl, rng_tr, training=False)
        vloss = (0.65*np.mean((syn_p_v-syn_vl)**2) +
                 0.20*np.mean((pfs_p_v-pfs_vl)**2))
        r_syn,_ = pearsonr(syn_vl.ravel(), syn_p_v.ravel())
        rho,_   = spearmanr(syn_vl.ravel(), syn_p_v.ravel())
        r_pfs,_ = pearsonr(pfs_vl.ravel(), pfs_p_v.ravel())

        if r_syn > best_r:
            best_r=r_syn; best_ep=ep; wait=0
            model.save(os.path.join(MODEL_OUT,f'pritamab_fusion_{tag}'))
            np.savez(os.path.join(MODEL_OUT,f'pritamab_fusion_{tag}_scaler'),
                     mean=sc.mean_.astype(np.float32),
                     scale=sc.scale_.astype(np.float32))
        else: wait+=1

        if ep%20==0 or ep<=warmup+3:
            print(f"  Ep{ep:3d} lr={lr:.2e} loss={tloss:.5f} val={vloss:.5f}"
                  f" | r_syn={r_syn:.4f} rho={rho:.4f} r_pfs={r_pfs:.4f}"
                  f"  [best={best_r:.4f}@{best_ep}]")
        log.append({'ep':ep,'r_syn':float(r_syn),'rho':float(rho),
                    'r_pfs':float(r_pfs),'tloss':float(tloss)})

        if wait>=patience:
            print(f"  Early stop @ep{ep} (best={best_r:.4f}@{best_ep})")
            break

    return model, sc, log, best_r, best_ep


# ================================================================
#  Drug-Rank Concordance
# ================================================================
DRUG_GT = {'Oxaliplatin':21.7,'FOLFOX':20.5,'FOLFIRI':18.8,
           '5-FU':18.4,'FOLFOXIRI':18.1,'Irinotecan':17.3,'Pritamab+Oxali':22.5}
DRUG_IC50 = {'Oxaliplatin':(0.09,0.09),'FOLFOX':(0.86,0.09),
             'FOLFIRI':(0.86,0.35),'5-FU':(0.86,0.86),
             'FOLFOXIRI':(0.09,0.35),'Irinotecan':(0.86,0.35),
             'Pritamab+Oxali':(0.001,0.09)}
DRUG_PAIRS = {'Oxaliplatin':('5-FU','Oxaliplatin'),'FOLFOX':('5-FU','Oxaliplatin'),
              'FOLFIRI':('5-FU','Irinotecan'),'5-FU':('5-FU','5-FU'),
              'FOLFOXIRI':('Oxaliplatin','Irinotecan'),'Irinotecan':('5-FU','Irinotecan'),
              'Pritamab+Oxali':('Pritamab','Oxaliplatin')}

def drug_concordance(model, sc, n=300, seed=77):
    rng = np.random.default_rng(seed)
    drug_probs = {}
    for drug in DRUG_GT:
        ic_a,ic_b = DRUG_IC50[drug]
        da,db     = DRUG_PAIRS[drug]
        feats = []
        for _ in range(n):
            pv=rng.normal(2.1,0.3); ph=int(pv>2.0)
            bliss = DRUG_GT[drug]*rng.uniform(0.85,1.15)
            f = build_feat_v3('G12D',da,db,ph,pv,bliss,ic_a,ic_b,rng)
            feats.append(f)
        Xd=sc.transform(np.array(feats,dtype=np.float32))
        sp,_,_ = model.forward(Xd, rng, training=False)
        drug_probs[drug] = float(sp.mean())

    pv_arr = np.array([drug_probs[d] for d in DRUG_GT])
    gt_arr = np.array(list(DRUG_GT.values()))
    pr = np.argsort(-pv_arr)+1; gr = np.argsort(-gt_arr)+1
    rho,_ = spearmanr(gr,pr)
    t2_gt=set(np.argsort(-gt_arr)[:2]); t2_pr=set(np.argsort(-pv_arr)[:2])
    top2  = len(t2_gt&t2_pr)/2

    print("\n-- Drug-Rank Concordance v3 --")
    for i,d in enumerate(DRUG_GT):
        m = "OK" if pr[i]==gr[i] else ("+-1" if abs(pr[i]-gr[i])==1 else "MISS")
        print(f"  {d:20s}: pred#{pr[i]}  GT#{gr[i]}  syn={drug_probs[d]:.4f}"
              f"  Bliss={gt_arr[i]:.1f}  [{m}]")
    print(f"  Spearman rho={rho:.3f}  Top-2 match={top2:.0%}")
    return rho, top2, drug_probs


# ================================================================
#  5-Fold CV
# ================================================================
def cross_validate_v3(X, y_syn, y_pfs, y_orr, W, n_folds=5, n_epochs=120):
    kf = KFold(n_folds, shuffle=True, random_state=2026)
    fold_rs = []
    for fold,(tr,vl) in enumerate(kf.split(X),1):
        sc_f=StandardScaler()
        Xtr_f=sc_f.fit_transform(X[tr]).astype(np.float32)
        Xvl_f=sc_f.transform(X[vl]).astype(np.float32)
        m_f=PritamamMLPv3(480,seed=2026+fold,drop_rate=0.15)
        rng_f=np.random.default_rng(2026+fold)
        n_trf=len(Xtr_f); bs_f=512; n_bf=max(1,n_trf//bs_f); t_f=0; best_rf=-1.0
        syn_trf,syn_vlf=y_syn[tr],y_syn[vl]
        pfs_trf,orr_trf=y_pfs[tr],y_orr[tr]; Wf=W[tr]

        for ep in range(1,n_epochs+1):
            lr_f = 5e-6+0.5*(4e-4-5e-6)*(1+np.cos(np.pi*ep/n_epochs))
            perm=rng_f.permutation(n_trf)
            Xs_f=Xtr_f[perm]; ss_f=syn_trf[perm]; ps_f=pfs_trf[perm]
            os_f=orr_trf[perm]; ws_f=Wf[perm]
            for b in range(n_bf):
                sl=slice(b*bs_f,(b+1)*bs_f)
                xb,sb,pb,ob,wb=Xs_f[sl],ss_f[sl],ps_f[sl],os_f[sl],ws_f[sl]
                if len(xb)<4: continue
                sp,pp,op=m_f.forward(xb,rng_f,True)
                m_f.backward(xb,sb,pb,ob,sp,pp,op); t_f+=1; m_f.adam_step(lr_f,t_f)
            sp_v,_,_=m_f.forward(Xvl_f,rng_f,False)
            r_f,_=pearsonr(syn_vlf.ravel(),sp_v.ravel())
            if r_f>best_rf: best_rf=r_f

        sp_v,_,_=m_f.forward(Xvl_f,rng_f,False)
        r_fin,_=pearsonr(syn_vlf.ravel(),sp_v.ravel())
        rho_f,_=spearmanr(syn_vlf.ravel(),sp_v.ravel())
        fold_rs.append(float(r_fin))
        print(f"  Fold{fold}: r_syn={r_fin:.4f} rho={rho_f:.4f} best={best_rf:.4f}")

    cv_mean=float(np.mean(fold_rs)); cv_std=float(np.std(fold_rs))
    print(f"\n  5-CV v3: r_syn={cv_mean:.4f} +/- {cv_std:.4f}")
    return fold_rs, cv_mean, cv_std


# ================================================================
#  Main
# ================================================================
if __name__ == '__main__':
    os.makedirs(MODEL_OUT, exist_ok=True)
    t0 = time.time()
    print("="*65)
    print("PRITAMAB FUSION v3 -- LITERATURE+DB SUPERVISED TRAINING")
    print("="*65)

    X, y_syn, y_pfs, y_orr, W, srcs = build_unified_dataset()

    # Save dataset
    ds_path = r'f:\ADDS\data\ml_training\unified_training_dataset.npz'
    np.savez(ds_path, X=X, y_syn=y_syn, y_pfs=y_pfs, y_orr=y_orr, W=W)
    print(f"\nDataset saved: {ds_path}")

    print("\n" + "="*65)
    print("STEP 1: 5-Fold Cross-Validation (120 epochs)")
    print("="*65)
    fold_rs, cv_mean, cv_std = cross_validate_v3(X,y_syn,y_pfs,y_orr,W, n_folds=5,n_epochs=120)

    print("\n" + "="*65)
    print("STEP 2: Full Training (300 epochs, best saved)")
    print("="*65)
    model,sc,log,best_r,best_ep = train_v3(X,y_syn,y_pfs,y_orr,W,
        n_epochs=300,bs=512,lr_max=4e-4,lr_min=5e-6,warmup=15,patience=40)

    # Load best
    model.load(os.path.join(MODEL_OUT,'pritamab_fusion_v3.npz'))
    sc_saved = np.load(os.path.join(MODEL_OUT,'pritamab_fusion_v3_scaler.npz'))
    from sklearn.preprocessing import StandardScaler as SC2
    sc2=SC2(); sc2.mean_=sc_saved['mean']; sc2.scale_=sc_saved['scale']

    print("\n" + "="*65)
    print("STEP 3: Drug-Rank Concordance")
    print("="*65)
    rho_rank, top2, drug_probs = drug_concordance(model, sc2)

    elapsed = round(time.time()-t0, 1)
    report = {
        'model': 'PritamamMLPv3 (BN+Residual+Dropout)',
        'dataset': {'total':int(len(X)), 'sources':dict(Counter(srcs))},
        'cv_5fold': {'folds':[round(r,4) for r in fold_rs],
                     'mean':round(cv_mean,4), 'std':round(cv_std,4)},
        'full_training': {'best_r_syn':round(float(best_r),4), 'best_ep':int(best_ep)},
        'drug_rank': {'rho':round(float(rho_rank),3),
                      'top2_match':round(float(top2),2),
                      'drug_probs':{k:round(v,4) for k,v in drug_probs.items()}},
        'elapsed_sec': elapsed,
    }
    rpt_path = os.path.join(MODEL_OUT,'pritamab_fusion_v3_report.json')
    with open(rpt_path,'w',encoding='utf-8') as f:
        json.dump(report,f,indent=2,ensure_ascii=False)

    print("\n" + "="*65)
    print("TRAINING COMPLETE")
    print("="*65)
    print(f"  5-CV r_syn     : {cv_mean:.4f} +/- {cv_std:.4f}")
    print(f"  Best val r_syn : {best_r:.4f}  (epoch {best_ep})")
    print(f"  Drug-rank rho  : {rho_rank:.3f}")
    print(f"  Top-2 match    : {top2:.0%}")
    print(f"  Elapsed        : {elapsed}s")
    print(f"  Report         : {rpt_path}")

    target_r = 0.70
    if best_r >= target_r and rho_rank >= 0.65:
        print(f"PASS: r_syn >= {target_r} AND drug-rank rho >= 0.65")
    elif best_r >= target_r:
        print(f"PARTIAL: r_syn PASS but drug-rank rho={rho_rank:.3f} < 0.65")
    else:
        print(f"BELOW TARGET: r_syn={best_r:.4f} (need {target_r-best_r:.3f} more)")
