"""
Energy Landscape Synergy v6 — Drug Functional Embedding
========================================================

Key innovation: Replace Morgan FP with Drug Functional Embedding (DFE)
DFE consists of:
1. MoA one-hot encoding from Drug Repurposing Hub (25/38 drugs)
2. Target gene overlap encoding 
3. Energy features from pathway graph (v5 cell-line specific)
4. Combined: DFE captures WHAT the drug DOES, not what it looks like

This tests whether functional annotations beat structural fingerprints.
"""

import json, logging, math, pickle, time
from pathlib import Path
from collections import Counter
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr
from sklearn.model_selection import KFold, GroupKFold
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("F:/ADDS/data/ml_training")
MODEL_DIR = Path("F:/ADDS/models")
device = "cuda" if torch.cuda.is_available() else "cpu"

R = 1.987e-3; T = 310.15; RT = R * T
kB = 3.2996e-24; h_p = 1.5837e-34
kBT_h = kB * T / h_p

def kcat_to_dG(kcat):
    return -RT * math.log(max(kcat, 1e-10) / kBT_h)
def ic50_to_potency(ic50_nM):
    return -math.log10(max(ic50_nM, 0.01) * 1e-9)

# ================================================================
# DRUG FUNCTIONAL EMBEDDING
# ================================================================

# MoA and target data from Drug Repurposing Hub
DRUG_MOA = {
    "BORTEZOMIB":     ["proteasome_inhibitor", "NFkB_inhibitor"],
    "CARBOPLATIN":    ["DNA_alkylating", "DNA_inhibitor"],
    "CYCLOPHOSPHAMIDE":["DNA_alkylating"],
    "DASATINIB":      ["BCR_ABL_inhibitor", "SRC_inhibitor", "KIT_inhibitor"],
    "DEXAMETHASONE":  ["glucocorticoid_agonist"],
    "DINACICLIB":     ["CDK_inhibitor"],
    "DOXORUBICIN":    ["topoisomerase_inhibitor"],
    "ERLOTINIB":      ["EGFR_inhibitor"],
    "ETOPOSIDE":      ["topoisomerase_inhibitor"],
    "GELDANAMYCIN":   ["HSP_inhibitor"],
    "GEMCITABINE":    ["antimetabolite", "RNR_inhibitor"],
    "LAPATINIB":      ["EGFR_inhibitor", "HER2_inhibitor"],
    "METFORMIN":      ["insulin_sensitizer", "AMPK_activator"],
    "METHOTREXATE":   ["DHFR_inhibitor", "antimetabolite"],
    "MK-2206":        ["AKT_inhibitor"],
    "MK-5108":        ["aurora_kinase_inhibitor"],
    "OXALIPLATIN":    ["DNA_alkylating", "DNA_inhibitor"],
    "PACLITAXEL":     ["tubulin_inhibitor", "microtubule_stabilizer"],
    "SN-38":          ["topoisomerase_inhibitor"],
    "SORAFENIB":      ["RAF_inhibitor", "VEGFR_inhibitor", "PDGFR_inhibitor"],
    "SUNITINIB":      ["VEGFR_inhibitor", "KIT_inhibitor", "PDGFR_inhibitor"],
    "TEMOZOLOMIDE":   ["DNA_alkylating"],
    "TOPOTECAN":      ["topoisomerase_inhibitor"],
    "VINBLASTINE":    ["tubulin_inhibitor"],
    "VINORELBINE":    ["tubulin_inhibitor"],
    # Unmatched — assign based on known mechanism
    "5-FU":           ["antimetabolite", "thymidylate_synthase_inhibitor"],
    "ABT-888":        ["PARP_inhibitor"],
    "AZD1775":        ["WEE1_inhibitor"],
    "BEZ-235":        ["PI3K_inhibitor", "mTOR_inhibitor"],
    "L778123":        ["farnesyltransferase_inhibitor"],
    "MITOMYCINE":     ["DNA_alkylating"],
    "MK-4541":        ["androgen_modulator"],
    "MK-4827":        ["PARP_inhibitor"],
    "MK-8669":        ["mTOR_inhibitor"],
    "MK-8776":        ["CHK1_inhibitor"],
    "MRK-003":        ["gamma_secretase_inhibitor"],
    "PD325901":       ["MEK_inhibitor"],
    "ZOLINZA":        ["HDAC_inhibitor"],
}

DRUG_TARGETS = {
    "BORTEZOMIB":     ["PSMA1","PSMB5"],
    "CARBOPLATIN":    [],
    "CYCLOPHOSPHAMIDE":["CYP2B6"],
    "DASATINIB":      ["ABL1","SRC","KIT","EPHA2","FYN","LCK","YES1"],
    "DEXAMETHASONE":  ["NR3C1"],
    "DINACICLIB":     ["CDK1","CDK2","CDK5","CDK9"],
    "DOXORUBICIN":    ["TOP2A"],
    "ERLOTINIB":      ["EGFR"],
    "ETOPOSIDE":      ["TOP2A","TOP2B"],
    "GELDANAMYCIN":   ["HSP90AA1"],
    "GEMCITABINE":    ["RRM1","RRM2","TYMS"],
    "LAPATINIB":      ["EGFR","ERBB2"],
    "METFORMIN":      ["PRKAB1","ACACB"],
    "METHOTREXATE":   ["DHFR"],
    "MK-2206":        ["AKT1","AKT2","AKT3"],
    "MK-5108":        ["AURKA","AURKB"],
    "OXALIPLATIN":    [],
    "PACLITAXEL":     ["TUBA1A","TUBB","BCL2"],
    "SN-38":          ["TOP1"],
    "SORAFENIB":      ["BRAF","FLT3","KIT","KDR","PDGFRB"],
    "SUNITINIB":      ["FLT3","KIT","KDR","PDGFRA","PDGFRB"],
    "TEMOZOLOMIDE":   ["MGMT"],
    "TOPOTECAN":      ["TOP1"],
    "VINBLASTINE":    ["TUBA1A","TUBB"],
    "VINORELBINE":    ["TUBA1A","TUBB"],
    "5-FU":           ["TYMS"],
    "ABT-888":        ["PARP1","PARP2"],
    "AZD1775":        ["WEE1"],
    "BEZ-235":        ["PIK3CA","MTOR"],
    "L778123":        ["FNTA"],
    "MITOMYCINE":     [],
    "MK-4541":        ["AR"],
    "MK-4827":        ["PARP1","PARP2"],
    "MK-8669":        ["MTOR"],
    "MK-8776":        ["CHEK1"],
    "MRK-003":        ["PSEN1"],
    "PD325901":       ["MAP2K1","MAP2K2"],
    "ZOLINZA":        ["HDAC1","HDAC2","HDAC3","HDAC6"],
}


def build_drug_functional_embedding():
    """Build MoA and target-based drug embeddings."""
    # Collect all unique MoA
    all_moa = sorted(set(m for ms in DRUG_MOA.values() for m in ms))
    all_targets = sorted(set(t for ts in DRUG_TARGETS.values() for t in ts))
    
    logger.info("DFE: %d unique MoA, %d unique targets", len(all_moa), len(all_targets))
    
    moa_idx = {m: i for i, m in enumerate(all_moa)}
    tgt_idx = {t: i for i, t in enumerate(all_targets)}
    
    embeddings = {}
    for drug in DRUG_MOA:
        moa_vec = np.zeros(len(all_moa), dtype=np.float32)
        for m in DRUG_MOA.get(drug, []):
            moa_vec[moa_idx[m]] = 1.0
        
        tgt_vec = np.zeros(len(all_targets), dtype=np.float32)
        for t in DRUG_TARGETS.get(drug, []):
            if t in tgt_idx:
                tgt_vec[tgt_idx[t]] = 1.0
        
        embeddings[drug.upper()] = np.concatenate([moa_vec, tgt_vec])
    
    dim = len(all_moa) + len(all_targets)
    logger.info("DFE dimension: %d (MoA=%d + Targets=%d)", dim, len(all_moa), len(all_targets))
    
    return embeddings, dim, all_moa, all_targets


# ================================================================
# PATHWAY GRAPH (same as v5)
# ================================================================

def build_pathway_graph():
    reactions = [
        ("STIMULUS","EGFR",0.16,500),("STIMULUS","HER2",0.12,600),("STIMULUS","IGF1R",0.20,400),
        ("EGFR","SOS_RAS",4.6,120),("HER2","SOS_RAS",3.0,150),("IGF1R","IRS1",2.0,200),
        ("SOS_RAS","RAS",2.0,80),("RAS","RAF",5.0,60),("RAF","MEK",8.0,15),("MEK","ERK",10.2,8),
        ("RAS","RAS_GDP",19.0,45),
        ("EGFR","PI3K",1.5,200),("IRS1","PI3K",2.5,100),("RAS","PI3K",1.2,250),
        ("PI3K","AKT",4.7,30),("AKT","MTORC1",2.0,50),("MTORC1","S6K",1.5,80),
        ("STIMULUS","HSP90",10.0,10),("HSP90","RAF",3.0,30),("HSP90","AKT",2.0,50),("HSP90","EGFR",1.5,70),
        ("EGFR","FAK",6.0,40),("FAK","SRC",4.0,50),
        ("ERK","CCND1",1.5,80),("CCND1","CDK46",3.0,25),("CDK46","RB1",5.0,30),
        ("RB1","PROLIFERATION",8.0,15),
        ("WEE1","CDK1",1.0,50),("CHK1","CDK1",1.0,60),("CDK1","PROLIFERATION",3.0,30),
        ("ERK","PROLIFERATION",1.0,100),("AKT","PROLIFERATION",0.3,350),
        ("AKT","SURVIVAL",3.0,40),("S6K","SURVIVAL",1.0,100),("MTORC1","SURVIVAL",0.8,120),
        ("SRC","MIGRATION",5.0,40),("FAK","MIGRATION",8.5,20),
    ]
    G = nx.DiGraph()
    for s,t,kcat,Km in reactions:
        G.add_edge(s,t,kcat=kcat,Km=Km,dG=kcat_to_dG(kcat),weight=kcat_to_dG(kcat))
    return G


DRUG_IC50 = {
    "ERLOTINIB":[("EGFR",2.0)],"LAPATINIB":[("EGFR",10.8),("HER2",9.2)],
    "SORAFENIB":[("RAF",22.0)],"PD325901":[("MEK",0.33)],
    "BEZ-235":[("PI3K",4.0),("MTORC1",6.0)],"MK-2206":[("AKT",8.0)],
    "MK-8669":[("MTORC1",0.2)],"DINACICLIB":[("CDK46",1.0),("CDK1",3.0)],
    "MK-1775":[("WEE1",5.2)],"AZD1775":[("WEE1",5.2)],"MK-8776":[("CHK1",3.0)],
    "MK-5108":[("CDK1",13.0)],"5-FU":[("PROLIFERATION",5000.0)],
    "GEMCITABINE":[("PROLIFERATION",50.0)],"OXALIPLATIN":[("PROLIFERATION",1000.0)],
    "DOXORUBICIN":[("PROLIFERATION",100.0)],"ETOPOSIDE":[("PROLIFERATION",1400.0)],
    "TOPOTECAN":[("PROLIFERATION",6.0)],"SN-38":[("PROLIFERATION",1.4)],
    "TEMOZOLOMIDE":[("PROLIFERATION",200000.0)],"METHOTREXATE":[("PROLIFERATION",21.0)],
    "CARBOPLATIN":[("PROLIFERATION",10000.0)],"MITOMYCINE":[("PROLIFERATION",500.0)],
    "CYCLOPHOSPHAMIDE":[("PROLIFERATION",1000000.0)],
    "PACLITAXEL":[("MIGRATION",4.0),("CDK1",100.0)],
    "VINBLASTINE":[("MIGRATION",2.0),("CDK1",50.0)],
    "VINORELBINE":[("MIGRATION",3.0),("CDK1",80.0)],
    "ABT-888":[("PROLIFERATION",5.2)],"MK-4827":[("PROLIFERATION",3.8)],
    "BORTEZOMIB":[("SURVIVAL",3.0),("PROLIFERATION",10.0)],
    "ZOLINZA":[("PROLIFERATION",1000.0),("SURVIVAL",2000.0)],
    "GELDANAMYCIN":[("HSP90",1.2)],"DASATINIB":[("SRC",0.55),("FAK",100.0)],
    "SUNITINIB":[("EGFR",880.0)],"MRK-003":[("PROLIFERATION",1500.0)],
    "L778123":[("RAS",2000.0)],"MK-4541":[("PROLIFERATION",500.0),("SURVIVAL",800.0)],
    "METFORMIN":[("MTORC1",200000.0)],"DEXAMETHASONE":[("SURVIVAL",10.0),("PROLIFERATION",100.0)],
}

PHENOTYPES = ["PROLIFERATION","SURVIVAL","MIGRATION"]
FEEDBACK_LOOPS = [("ERK","SOS_RAS",0.4),("S6K","IRS1",0.3),("MTORC1","PI3K",0.2),("AKT","RAF",0.3)]
SYNTHETIC_LETHALITY = [
    ({"RAF"},{"MEK"},2.0),({"MEK"},{"PI3K"},3.0),({"EGFR"},{"PI3K"},2.5),
    ({"RAF"},{"PI3K"},2.5),({"PROLIFERATION"},{"SURVIVAL"},4.0),
]

def perturb_graph(G_base, drug_name):
    G = G_base.copy()
    drug_up = drug_name.upper().strip()
    if drug_up not in DRUG_IC50: return G, set()
    blocked = set()
    for tgt, ic50 in DRUG_IC50[drug_up]:
        pot = ic50_to_potency(ic50); barrier = RT * pot; blocked.add(tgt)
        if tgt in G:
            for _,s,d in G.edges(tgt,data=True): d['weight'] = d.get('dG',d['weight'])+barrier
            if tgt in PHENOTYPES:
                for p,_,d in G.in_edges(tgt,data=True): d['weight'] = d.get('dG',d['weight'])+barrier
    return G, blocked

def apply_feedback(G,blocked):
    for fs,ft,st in FEEDBACK_LOOPS:
        if fs in blocked and ft in G:
            rel = RT*st*10
            for _,s,d in G.edges(ft,data=True): d['weight'] = max(d['weight']-rel, d.get('dG',0)*0.5)
    return G

def hill(x,K=0.5,n=3): return x**n/(K**n+x**n)

def sl_bonus(ba,bb):
    b=0.0
    both=ba|bb
    for sa,sb,v in SYNTHETIC_LETHALITY:
        a1=bool(sa&ba);b1=bool(sb&bb);a2=bool(sa&bb);b2=bool(sb&ba)
        if(a1 and b1)or(a2 and b2): b+=v
    return b

def compute_flow(G,source="STIMULUS"):
    results={}
    for target in PHENOTYPES:
        try:
            paths=list(nx.all_simple_paths(G,source,target,cutoff=8))
            if not paths: results[target]={'dG':999,'n':0,'cap':0,'mean':999,'std':0,'path':[]}; continue
            energies=[sum(G[p[i]][p[i+1]]['weight'] for i in range(len(p)-1)) for p in paths]
            mE=min(energies); Z=sum(math.exp(-(E-mE)/RT) for E in energies)
            cap=math.exp(-mE/RT)*Z; idx=int(np.argmin(energies))
            results[target]={'dG':mE,'n':len(paths),'cap':cap,'mean':np.mean(energies),
                            'std':np.std(energies) if len(energies)>1 else 0,'path':paths[idx]}
        except: results[target]={'dG':999,'n':0,'cap':0,'mean':999,'std':0,'path':[]}
    return results

def extract_energy_features(G_base, drug_a, drug_b):
    da,db = drug_a.upper().strip(), drug_b.upper().strip()
    f0=compute_flow(G_base)
    Ga,ba=perturb_graph(G_base,da); Ga=apply_feedback(Ga,ba)
    Gb,bb=perturb_graph(G_base,db); Gb=apply_feedback(Gb,bb)
    Gab,bab1=perturb_graph(G_base,da); Gab,bab2=perturb_graph(Gab,db)
    Gab=apply_feedback(Gab,bab1|bab2)
    fa=compute_flow(Ga);fb=compute_flow(Gb);fab=compute_flow(Gab)
    feats=[]
    for ph in PHENOTYPES:
        n=f0.get(ph,{});a=fa.get(ph,{});b=fb.get(ph,{});ab=fab.get(ph,{})
        dA=a.get('dG',0)-n.get('dG',0);dB=b.get('dG',0)-n.get('dG',0)
        dAB=ab.get('dG',0)-n.get('dG',0);syn=dAB-(dA+dB)
        cn=max(n.get('cap',1),1e-100)
        rA=a.get('cap',0)/cn;rB=b.get('cap',0)/cn;rAB=ab.get('cap',0)/cn
        bliss=rA*rB;csyn=bliss-rAB
        hA=hill(rA);hB=hill(rB);hAB=hill(rAB);hsyn=hA*hB-hAB
        nb=max(n.get('n',1),1);ploss=1.0-ab.get('n',0)/nb
        rswitch=1.0 if ab.get('path',[])!=n.get('path',[]) else 0.0
        feats.extend([dA,dB,dAB,syn,csyn,hsyn,ploss,rswitch,rAB,ab.get('mean',0)-n.get('mean',0)])
    total_loss=sum(1.0-fab.get(p,{}).get('cap',0)/max(f0.get(p,{}).get('cap',1),1e-100) for p in PHENOTYPES)
    feats.append(total_loss)
    sbl=sl_bonus(ba,bb);feats.append(sbl)
    ta={t for t,_ in DRUG_IC50.get(da,[])};tb={t for t,_ in DRUG_IC50.get(db,[])}
    ov=len(ta&tb);tot=len(ta|tb)
    pa=sum(ic50_to_potency(ic) for _,ic in DRUG_IC50.get(da,[("",1000)]))
    pb=sum(ic50_to_potency(ic) for _,ic in DRUG_IC50.get(db,[("",1000)]))
    feats.extend([float(ov),float(tot),1.0 if ov>0 else 0.0,pa,pb,pa*pb/100,float(sbl>0)])
    return np.array(feats,dtype=np.float32)


# ================================================================
# Model
# ================================================================

class SynergyDNN(nn.Module):
    def __init__(self, dim, hidden=[512,256,128]):
        super().__init__()
        layers=[]
        prev=dim
        for h in hidden:
            layers.extend([nn.Linear(prev,h),nn.LayerNorm(h),nn.SiLU(),nn.Dropout(0.3)])
            prev=h
        layers.append(nn.Linear(prev,1))
        self.net=nn.Sequential(*layers)
    def forward(self,x): return self.net(x).squeeze(-1)


def train_eval(X, y, groups, dim, n_epochs=250, label=""):
    scaler=StandardScaler(); X_s=scaler.fit_transform(X)
    results={}
    for cv,kf in [("random",KFold(5,shuffle=True,random_state=42)),
                   ("drug_pair",GroupKFold(5))]:
        rs=[]
        splits=kf.split(X_s,y,groups) if cv=="drug_pair" else kf.split(X_s)
        for fold,(ti,vi) in enumerate(splits):
            Xt=torch.FloatTensor(X_s[ti]).to(device); yt=torch.FloatTensor(y[ti]).to(device)
            Xv=torch.FloatTensor(X_s[vi]).to(device)
            model=SynergyDNN(dim).to(device)
            opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
            sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=n_epochs)
            best_r,patience,best_state=-1,0,None
            for ep in range(n_epochs):
                model.train()
                perm=torch.randperm(len(Xt))
                for s in range(0,len(Xt),2048):
                    idx=perm[s:s+2048]; opt.zero_grad()
                    loss=F.mse_loss(model(Xt[idx]),yt[idx]); loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
                sched.step()
                if(ep+1)%10==0:
                    model.eval()
                    with torch.no_grad(): vp=model(Xv).cpu().numpy()
                    r=pearsonr(y[vi],vp)[0]
                    if r>best_r: best_r=r;patience=0;best_state={k:v.clone() for k,v in model.state_dict().items()}
                    else: patience+=1
                    if patience>=5: break
            if best_state: model.load_state_dict(best_state)
            model.eval()
            with torch.no_grad(): vp=model(Xv).cpu().numpy()
            r=pearsonr(y[vi],vp)[0]; rs.append(r)
            logger.info("  %s %s fold %d: r=%.4f",label,cv,fold+1,r)
        avg=np.mean(rs);std=np.std(rs)
        logger.info("  %s %s: r=%.4f +/- %.4f",label,cv,avg,std)
        results[cv]={"r":round(float(avg),4),"std":round(float(std),4)}
    return results


def main():
    t0=time.time()
    logger.info("="*60)
    logger.info("ENERGY SYNERGY v6 — Drug Functional Embedding")
    logger.info("="*60)
    
    G_base = build_pathway_graph()
    dfe_embeddings, dfe_dim, all_moa, all_targets = build_drug_functional_embedding()
    
    # Show embedding examples
    for drug in ["ERLOTINIB","PACLITAXEL","DOXORUBICIN","BORTEZOMIB"]:
        emb = dfe_embeddings.get(drug.upper())
        if emb is not None:
            active_moa = [all_moa[i] for i in range(len(all_moa)) if emb[i]>0]
            active_tgt = [all_targets[i] for i in range(len(all_targets)) if emb[len(all_moa)+i]>0]
            logger.info("  %s: MoA=%s, Targets=%s", drug, active_moa, active_tgt[:5])
    
    # Compute drug pair DFE similarity
    logger.info("\n--- Drug Pair Functional Similarity ---")
    similar_pairs = [
        ("ERLOTINIB","LAPATINIB"),  # both EGFR inhibitors
        ("VINBLASTINE","VINORELBINE"),  # both tubulin
        ("DOXORUBICIN","ETOPOSIDE"),  # both topoisomerase
        ("ERLOTINIB","PACLITAXEL"),  # different MoA
    ]
    for da,db in similar_pairs:
        ea=dfe_embeddings.get(da.upper(),np.zeros(dfe_dim))
        eb=dfe_embeddings.get(db.upper(),np.zeros(dfe_dim))
        cos = np.dot(ea,eb)/(np.linalg.norm(ea)*np.linalg.norm(eb)+1e-10)
        logger.info("  %s + %s: cosine=%.3f", da, db, cos)
    
    # Load data
    df = pd.read_csv(DATA_DIR/"synergy_combined.csv", low_memory=False)
    df = df[df.source=="oneil"]
    
    smiles = {}
    for p in [MODEL_DIR/"synergy"/"drug_smiles.json", MODEL_DIR/"synergy"/"drug_smiles_extended.json"]:
        if p.exists():
            with open(p) as f: smiles.update(json.load(f))
    from rdkit import Chem; from rdkit.Chem import AllChem
    fps = {}
    for name,smi in smiles.items():
        mol=Chem.MolFromSmiles(smi)
        if mol: fps[name]=np.array(AllChem.GetMorganFingerprintAsBitVect(mol,2,1024),dtype=np.float32)
    
    embed_data=None
    ep=DATA_DIR/"depmap"/"cellline_embedding_v2.pkl"
    if ep.exists():
        with open(ep,"rb") as f: embed_data=pickle.load(f)
    
    def norm_cl(n): return str(n).upper().replace("-","").replace("_","").replace(" ","").replace(".","")
    
    # Build features
    logger.info("\nBuilding dataset...")
    X_energy,X_fp,X_dfe,X_cl,y_list,groups=[],[],[],[],[],[]
    feat_cache={}
    
    for i,(_,row) in enumerate(df.iterrows()):
        da,db=str(row["drug_a"]),str(row["drug_b"])
        score=float(row["synergy_loewe"])
        cl=norm_cl(str(row["cell_line"]))
        if np.isnan(score) or da not in fps or db not in fps: continue
        
        pair_key=tuple(sorted([da.upper(),db.upper()]))
        if pair_key not in feat_cache:
            feat_cache[pair_key]=extract_energy_features(G_base,da,db)
        
        X_energy.append(feat_cache[pair_key])
        X_fp.append(np.concatenate([fps[da],fps[db]]))
        
        # DFE: concatenate drug_a DFE + drug_b DFE
        ea=dfe_embeddings.get(da.upper(),np.zeros(dfe_dim,dtype=np.float32))
        eb=dfe_embeddings.get(db.upper(),np.zeros(dfe_dim,dtype=np.float32))
        X_dfe.append(np.concatenate([ea,eb]))
        
        if embed_data:
            X_cl.append(embed_data["embeddings"].get(cl,np.zeros(embed_data["dim"],dtype=np.float32)))
        y_list.append(score); groups.append(pair_key)
    
    X_energy=np.array(X_energy,dtype=np.float32)
    X_fp=np.array(X_fp,dtype=np.float32)
    X_dfe=np.array(X_dfe,dtype=np.float32)
    y=np.array(y_list,dtype=np.float32)
    
    unique_pairs=list(set(groups))
    pair_to_id={p:i for i,p in enumerate(unique_pairs)}
    group_ids=np.array([pair_to_id[g] for g in groups])
    
    logger.info("Dataset: %d samples, %d pairs",len(y),len(unique_pairs))
    logger.info("Energy: %d, FP: %d, DFE: %d",X_energy.shape[1],X_fp.shape[1],X_dfe.shape[1])
    
    all_results={}
    
    logger.info("\n"+"="*40)
    logger.info("MODEL EVALUATION — DFE vs FP comparison")
    logger.info("="*40)
    
    # A: Energy only (baseline)
    logger.info("\n--- A: Energy only ---")
    r=train_eval(X_energy,y,group_ids,X_energy.shape[1],label="Energy")
    all_results['energy_only']=r
    
    # B: Energy + DFE (NEW: replaces FP)
    logger.info("\n--- B: Energy + DFE (NEW) ---")
    X_edfe=np.concatenate([X_energy,X_dfe],axis=1)
    r=train_eval(X_edfe,y,group_ids,X_edfe.shape[1],label="E+DFE")
    all_results['energy_dfe']=r
    
    # C: Energy + FP (old)
    logger.info("\n--- C: Energy + FP (old) ---")
    X_efp=np.concatenate([X_energy,X_fp],axis=1)
    r=train_eval(X_efp,y,group_ids,X_efp.shape[1],label="E+FP")
    all_results['energy_fp']=r
    
    if embed_data and X_cl:
        X_cl_arr=np.array(X_cl,dtype=np.float32)
        
        # D: Full with DFE (NEW)
        logger.info("\n--- D: Full DFE (Energy + DFE + CL) ---")
        X_full_dfe=np.concatenate([X_energy,X_dfe,X_cl_arr],axis=1)
        r=train_eval(X_full_dfe,y,group_ids,X_full_dfe.shape[1],label="Full-DFE")
        all_results['full_dfe']=r
        
        # E: Full with FP (old)
        logger.info("\n--- E: Full FP (Energy + FP + CL) ---")
        X_full_fp=np.concatenate([X_energy,X_fp,X_cl_arr],axis=1)
        r=train_eval(X_full_fp,y,group_ids,X_full_fp.shape[1],label="Full-FP")
        all_results['full_fp']=r
        
        # F: Full with DFE + FP combined
        logger.info("\n--- F: Full ALL (Energy + DFE + FP + CL) ---")
        X_all=np.concatenate([X_energy,X_dfe,X_fp,X_cl_arr],axis=1)
        r=train_eval(X_all,y,group_ids,X_all.shape[1],label="Full-ALL")
        all_results['full_all']=r
    
    elapsed=time.time()-t0
    
    logger.info("\n"+"="*60)
    logger.info("FINAL v6: DFE vs FP Comparison")
    logger.info("="*60)
    logger.info("  %-20s %-15s %-15s","Model","Random","Drug-pair")
    logger.info("  "+"-"*50)
    for name,res in all_results.items():
        rr=res.get('random',{}).get('r',0);rp=res.get('drug_pair',{}).get('r',0)
        logger.info("  %-20s r=%.4f        r=%.4f",name,rr,rp)
    logger.info("\n  v3/v5 Full (ref):  r=0.7164        r=0.641")
    logger.info("  Phase5 MLP (ref):  r=0.7030        r=0.620")
    logger.info("\n  Time: %.1f seconds",elapsed)
    
    with open(MODEL_DIR/"energy_synergy_v6_results.json","w") as f:
        json.dump(all_results,f,indent=2,default=float)
    logger.info("Saved: energy_synergy_v6_results.json")


if __name__=="__main__":
    main()
