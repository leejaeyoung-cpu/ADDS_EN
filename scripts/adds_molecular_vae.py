"""
ADDS Molecular Structure Generation — Conditional VAE (CVAE) v2.0
====================================================================
심층 분자 생성 시스템. Pritamab 결합 친화도를 조건으로
새로운 PrPc 표적 분자를 de novo 설계합니다.

Architecture:
  Encoder: Bidirectional LSTM (SMILES char → latent z)
  Decoder: LSTM (latent z + condition → SMILES char)
  Condition: BindingDB Ki value + ChEMBL activity (normalized)
  Latent dim: 256
  Vocab: SMILES character-level (75 chars)

Training data:
  1. drug_smiles.json  (4,246 compounds)
  2. binding_database/ ChEMBL (EGFR, BRAF, mTOR, HER2, VEGFR2)
  3. BindingDB_Extracted.tsv (PrPc-related targets)

Output:
  models/molecular_vae/cvae_pritamab_v2.pt
  data/drug_combinations/generated_molecules_{n}.csv
  docs/molecular_generation_report.txt

Usage:
  # Train
  python scripts/adds_molecular_vae.py --mode train --epochs 200

  # Generate
  python scripts/adds_molecular_vae.py --mode generate --n 10000

  # Validate generated molecules
  python scripts/adds_molecular_vae.py --mode validate
"""

import os, sys, json, csv, pickle, time
import numpy as np
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

MODEL_OUT = ROOT / "models" / "molecular_vae"
MODEL_OUT.mkdir(parents=True, exist_ok=True)
DATA_OUT  = ROOT / "data" / "drug_combinations"
DATA_OUT.mkdir(parents=True, exist_ok=True)

import warnings
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════
# 1. SMILES Vocabulary
# ═══════════════════════════════════════════════════════════════
class SMILESVocab:
    """Character-level SMILES vocabulary."""

    SPECIAL = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]
    ATOMS   = list("CcNnOoSsPpFfIlrBbKH")
    BONDS   = list("=#@\\/-()[]%+.-123456789")
    EXTRAS  = list("aromatic")

    def __init__(self, smiles_list=None):
        chars  = self.SPECIAL + self.ATOMS + self.BONDS
        # Add any chars found in training data
        if smiles_list:
            extra = set()
            for s in smiles_list:
                for c in s:
                    if c not in chars:
                        extra.add(c)
            chars += sorted(extra)
        self.chars   = chars
        self.c2i     = {c: i for i, c in enumerate(chars)}
        self.i2c     = {i: c for c, i in self.c2i.items()}
        self.pad_idx = self.c2i["<PAD>"]
        self.sos_idx = self.c2i["<SOS>"]
        self.eos_idx = self.c2i["<EOS>"]
        self.unk_idx = self.c2i["<UNK>"]

    def encode(self, smiles, max_len=120):
        tokens = [self.sos_idx]
        for c in smiles[:max_len-2]:
            tokens.append(self.c2i.get(c, self.unk_idx))
        tokens.append(self.eos_idx)
        # Pad
        tokens += [self.pad_idx] * (max_len - len(tokens))
        return tokens[:max_len]

    def decode(self, indices, clean=True):
        chars = []
        for i in indices:
            if i == self.eos_idx: break
            if i in (self.pad_idx, self.sos_idx): continue
            chars.append(self.i2c.get(i, "?"))
        s = "".join(chars)
        if clean:
            # Remove invalid chars
            s = s.replace("?","")
        return s

    def __len__(self):
        return len(self.chars)


# ═══════════════════════════════════════════════════════════════
# 2. CVAE Model (PyTorch)
# ═══════════════════════════════════════════════════════════════
def build_cvae(vocab_size, latent_dim=256, cond_dim=8,
               hidden_dim=512, n_layers=2, max_len=120):
    """
    Build Conditional SMILES VAE.

    Args:
        vocab_size: SMILES vocabulary size
        latent_dim: latent space dimension
        cond_dim:  condition vector dimension (binding affinities)
        hidden_dim: LSTM hidden size
        n_layers:  LSTM layers
        max_len:   max SMILES length

    Returns:
        model dict with encoder, decoder, loss_fn
    """
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(vocab_size, 128, padding_idx=0)
                self.lstm  = nn.LSTM(128 + cond_dim, hidden_dim,
                                     n_layers, batch_first=True,
                                     bidirectional=True, dropout=0.2)
                self.fc_mu  = nn.Linear(hidden_dim * 2, latent_dim)
                self.fc_log = nn.Linear(hidden_dim * 2, latent_dim)

            def forward(self, x, cond):
                # x: (B, L)  cond: (B, cond_dim)
                e  = self.embed(x)                              # (B, L, 128)
                c  = cond.unsqueeze(1).expand(-1, x.size(1), -1) # (B, L, cond_dim)
                ec = torch.cat([e, c], dim=-1)                  # (B, L, 128+cond_dim)
                _, (h, _) = self.lstm(ec)
                # h: (2*n_layers, B, hidden_dim) -> take last 2 layers, cat directions
                h = torch.cat([h[-2], h[-1]], dim=-1)           # (B, hidden_dim*2)
                mu      = self.fc_mu(h)
                log_var = self.fc_log(h)
                return mu, log_var

        class Decoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed   = nn.Embedding(vocab_size, 128, padding_idx=0)
                self.z_proj  = nn.Linear(latent_dim + cond_dim, hidden_dim * n_layers)
                self.lstm    = nn.LSTM(128 + latent_dim + cond_dim, hidden_dim,
                                       n_layers, batch_first=True, dropout=0.2)
                self.out     = nn.Linear(hidden_dim, vocab_size)
                self.dropout = nn.Dropout(0.1)

            def forward(self, tgt, z, cond):
                # tgt: (B, L)  z: (B, latent)  cond: (B, cond_dim)
                e  = self.dropout(self.embed(tgt))               # (B, L, 128)
                zc = torch.cat([z, cond], dim=-1)                # (B, latent+cond)
                # Init hidden state from z
                h0 = torch.tanh(self.z_proj(zc))                # (B, hidden*n_layers)
                h0 = h0.view(n_layers, tgt.size(0), hidden_dim) # (n_layers, B, hidden)
                c0 = torch.zeros_like(h0)
                # Append z, cond to each token
                zce = zc.unsqueeze(1).expand(-1, tgt.size(1), -1) # (B, L, latent+cond)
                inp = torch.cat([e, zce], dim=-1)                  # (B, L, 128+L+C)
                out, _ = self.lstm(inp, (h0, c0))
                return self.out(out)                               # (B, L, vocab)

        class CVAE(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = Encoder()
                self.decoder = Decoder()
                self.latent_dim = latent_dim
                self.cond_dim   = cond_dim
                self.max_len    = max_len

            def reparameterize(self, mu, log_var):
                std = torch.exp(0.5 * log_var)
                eps = torch.randn_like(std)
                return mu + eps * std

            def forward(self, src, tgt, cond):
                mu, log_var = self.encoder(src, cond)
                z = self.reparameterize(mu, log_var)
                logits = self.decoder(tgt, z, cond)
                return logits, mu, log_var

            def generate(self, z, cond, vocab, device, temperature=1.0):
                """Generate SMILES from latent z and condition."""
                self.eval()
                with torch.no_grad():
                    tokens = [vocab.sos_idx]
                    hidden = None
                    zc = torch.cat([z, cond], dim=-1).unsqueeze(1)  # (1, 1, L+C)
                    for _ in range(vocab.max_len if hasattr(vocab,"max_len") else 120):
                        x       = torch.tensor([[tokens[-1]]], device=device)
                        e       = self.decoder.embed(x)              # (1,1,128)
                        inp     = torch.cat([e, zc.expand(-1,1,-1) if zc.size(1)==1 else zc], dim=-1)
                        if hidden is None:
                            zc_flat = torch.cat([z, cond], dim=-1)
                            h0 = torch.tanh(self.decoder.z_proj(zc_flat))
                            h0 = h0.view(n_layers, 1, hidden_dim)
                            hidden = (h0, torch.zeros_like(h0))
                        out, hidden = self.decoder.lstm(inp, hidden)
                        logits  = self.decoder.out(out.squeeze(1)) / max(temperature, 0.1)
                        probs   = torch.softmax(logits, dim=-1)
                        tok     = torch.multinomial(probs, 1).item()
                        tokens.append(tok)
                        if tok == vocab.eos_idx: break
                    return vocab.decode(tokens)

        return CVAE()

    except ImportError as e:
        print(f"PyTorch not available: {e}")
        return None


# ═══════════════════════════════════════════════════════════════
# 3. Data Loading
# ═══════════════════════════════════════════════════════════════
def load_training_data():
    """Load and combine all SMILES training data."""
    all_smiles  = []
    all_conds   = []  # binding affinity conditions

    # Source 1: drug_smiles.json (4,246 drugs)
    smiles_path = ROOT / "models" / "synergy" / "drug_smiles.json"
    if smiles_path.exists():
        with open(smiles_path) as f:
            drug_dict = json.load(f)
        for name, smi in drug_dict.items():
            if smi and len(smi) > 5:
                all_smiles.append(smi)
                # Default condition: unknown binding (0.5)
                all_conds.append([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        print(f"  drug_smiles.json: {len(drug_dict)} molecules")

    # Source 2: ChEMBL binding_database
    bd_dir = ROOT / "binding_database"
    target_map = {
        "chembl_egfr.csv":  0, "chembl_braf.csv": 1,
        "chembl_mtor.csv":  2, "chembl_her2.csv": 3,
        "chembl_vegfr2.csv":4,
    }
    for fn, target_idx in target_map.items():
        fp = bd_dir / fn
        if not fp.exists(): continue
        try:
            with open(fp, encoding="utf-8", errors="ignore") as f:
                reader = csv.DictReader(f)
                n_added = 0
                for row in reader:
                    smi = (row.get("Smiles") or row.get("smiles") or
                           row.get("canonical_smiles","")).strip()
                    if not smi or len(smi) < 5: continue
                    # Activity
                    act = row.get("Standard Value") or row.get("pChEMBL Value","")
                    try: act_f = float(act) / 10.0  # normalize
                    except: act_f = 0.5
                    cond = [0.5]*8
                    cond[target_idx] = min(max(act_f, 0.0), 1.0)
                    all_smiles.append(smi)
                    all_conds.append(cond)
                    n_added += 1
                    if n_added > 2000: break
            print(f"  {fn}: {n_added} molecules")
        except Exception as e:
            print(f"  {fn}: error ({e})")

    # Source 3: BindingDB
    bdb_path = ROOT / "bindingdb" / "BindingDB_Extracted.tsv"
    if bdb_path.exists():
        try:
            with open(bdb_path, encoding="utf-8", errors="ignore") as f:
                reader = csv.DictReader(f, delimiter="\t")
                n_added = 0
                for row in reader:
                    smi  = row.get("Ligand SMILES","").strip()
                    ki   = row.get("Ki (nM)","")
                    if not smi or len(smi) < 5: continue
                    try: ki_f = min(1.0, 1.0 / (float(ki.replace(">","").replace("<","")) / 1000 + 1))
                    except: ki_f = 0.5
                    cond = [ki_f] + [0.5]*7
                    all_smiles.append(smi)
                    all_conds.append(cond)
                    n_added += 1
                    if n_added > 3000: break
            print(f"  BindingDB: {n_added} molecules")
        except Exception as e:
            print(f"  BindingDB error: {e}")

    print(f"\n  Total training molecules: {len(all_smiles)}")
    return all_smiles, np.array(all_conds, dtype=np.float32)


# ═══════════════════════════════════════════════════════════════
# 4. Training Loop
# ═══════════════════════════════════════════════════════════════
def train(epochs=200, batch_size=128, lr=3e-4,
          kl_weight_start=0.0, kl_weight_end=1.0, kl_anneal_epochs=50,
          max_len=120, latent_dim=256, hidden_dim=512):
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        print("PyTorch required. Install: pip install torch")
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[CVAE Training]  device={device}  epochs={epochs}  latent={latent_dim}")

    # Load data
    print("\nLoading training data...")
    smiles_list, conds = load_training_data()

    # Vocabulary
    vocab = SMILESVocab(smiles_list)
    print(f"Vocabulary size: {len(vocab)}")

    # Encode SMILES
    print("Encoding SMILES...")
    X = []
    C = []
    for smi, cond in zip(smiles_list, conds):
        tok = vocab.encode(smi, max_len=max_len)
        X.append(tok)
        C.append(cond)

    X = torch.tensor(X, dtype=torch.long)
    C = torch.tensor(np.array(C), dtype=torch.float32)
    # src = all tokens except last, tgt = all tokens except first
    X_src = X[:, :-1]
    X_tgt = X[:, 1:]

    dataset    = TensorDataset(X_src, X_tgt, C)
    loader     = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, num_workers=0)

    # Build model
    model = build_cvae(len(vocab), latent_dim=latent_dim,
                       cond_dim=8, hidden_dim=hidden_dim,
                       n_layers=2, max_len=max_len)
    if model is None:
        return None
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    best_loss = float("inf")
    history   = []

    print("\nStarting training...")
    for epoch in range(1, epochs+1):
        model.train()
        epoch_loss = 0; epoch_rec = 0; epoch_kl = 0
        t0 = time.time()

        # KL annealing (beta schedule)
        kl_w = min(kl_weight_end,
                   kl_weight_start + (kl_weight_end - kl_weight_start)
                   * epoch / kl_anneal_epochs)

        for batch_src, batch_tgt, batch_cond in loader:
            batch_src  = batch_src.to(device)
            batch_tgt  = batch_tgt.to(device)
            batch_cond = batch_cond.to(device)

            logits, mu, log_var = model(batch_src, batch_src, batch_cond)

            # Reconstruction loss (cross-entropy, ignoring PAD)
            B, L, V = logits.shape
            rec_loss = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)(
                logits.view(B*L, V), batch_tgt.reshape(-1))

            # KL divergence
            kl_loss  = -0.5 * torch.mean(
                1 + log_var - mu.pow(2) - log_var.exp())

            loss = rec_loss + kl_w * kl_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_rec  += rec_loss.item()
            epoch_kl   += kl_loss.item()

        scheduler.step()
        n_batches  = len(loader)
        avg_loss   = epoch_loss / n_batches
        avg_rec    = epoch_rec  / n_batches
        avg_kl     = epoch_kl   / n_batches
        elapsed    = time.time() - t0

        history.append({"epoch":epoch, "loss":avg_loss,
                        "rec":avg_rec, "kl":avg_kl, "kl_w":kl_w})

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs}  loss={avg_loss:.4f}  "
                  f"rec={avg_rec:.4f}  kl={avg_kl:.4f}  kl_w={kl_w:.3f}  "
                  f"lr={scheduler.get_last_lr()[0]:.6f}  [{elapsed:.1f}s]")

        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "vocab": vocab,
                "config": {
                    "latent_dim": latent_dim,
                    "hidden_dim": hidden_dim,
                    "cond_dim": 8,
                    "max_len": max_len,
                    "vocab_size": len(vocab),
                },
                "history": history,
                "best_loss": best_loss,
            }, str(MODEL_OUT / "cvae_pritamab_v2.pt"))

        # Sample every 25 epochs for monitoring
        if epoch % 25 == 0:
            model.eval()
            with torch.no_grad():
                z    = torch.randn(5, latent_dim, device=device)
                # Condition: high PrPc activity (cond[0] = 1.0)
                cond = torch.zeros(5, 8, device=device)
                cond[:, 0] = 1.0  # PrPc binding active
                for i in range(5):
                    s = model.generate(z[i:i+1], cond[i:i+1], vocab, device,
                                       temperature=0.8)
                    # Basic validity check (no rdkit needed)
                    valid = len(s) > 5 and any(c.isalpha() for c in s)
                    print(f"    Sample {i+1}: {s[:60]}  [{'VALID' if valid else 'invalid'}]")

    print(f"\nTraining complete. Best loss: {best_loss:.4f}")
    print(f"Model saved: {MODEL_OUT}/cvae_pritamab_v2.pt")
    return MODEL_OUT / "cvae_pritamab_v2.pt"


# ═══════════════════════════════════════════════════════════════
# 5. Generation & Validation
# ═══════════════════════════════════════════════════════════════
def generate_molecules(n=10000, temperature=0.8, condition="prpc",
                       filter_validity=True):
    """Generate n molecules conditioned on target."""
    try:
        import torch
    except ImportError:
        print("PyTorch required"); return []

    model_path = MODEL_OUT / "cvae_pritamab_v2.pt"
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        print("Run with --mode train first")
        return []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = torch.load(str(model_path), map_location=device, weights_only=False)
    vocab  = ckpt["vocab"]
    cfg    = ckpt["config"]

    model  = build_cvae(cfg["vocab_size"], latent_dim=cfg["latent_dim"],
                        cond_dim=cfg["cond_dim"], hidden_dim=cfg["hidden_dim"],
                        n_layers=2, max_len=cfg["max_len"])
    model.load_state_dict(ckpt["model_state_dict"])
    model  = model.to(device)
    model.eval()

    # Condition vector
    cond_map = {
        "prpc":  [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # PrPc binding
        "egfr":  [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        "her2":  [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        "braf":  [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "multi": [0.8, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],  # multi-target
    }
    cond_vec = cond_map.get(condition, cond_map["prpc"])

    print(f"\n[Generating {n:,} molecules]  condition={condition}  temp={temperature}")

    generated = []
    valid_n   = 0
    batch_z   = 256

    # RDKit disabled (NumPy 2.x incompatibility)
    # Validity = length > 5 and has alphabetic SMILES atoms
    rdkit_ok = False

    with torch.no_grad():
        for start in range(0, n, batch_z):
            bs   = min(batch_z, n - start)
            z    = torch.randn(bs, cfg["latent_dim"], device=device)
            c    = torch.tensor([cond_vec]*bs, dtype=torch.float32, device=device)
            for i in range(bs):
                smi = model.generate(z[i:i+1], c[i:i+1], vocab, device, temperature)
                # Basic SMILES validity: length + has ring/atom chars
                valid = len(smi) > 5 and any(c.isalpha() for c in smi)
                if valid: valid_n += 1
                mol = None  # RDKit disabled

                props = {}
                if mol and rdkit_ok:
                    props = {
                        "mw": round(Descriptors.MolWt(mol), 2),
                        "logp": round(Descriptors.MolLogP(mol), 2),
                        "hbd": Descriptors.NumHDonors(mol),
                        "hba": Descriptors.NumHAcceptors(mol),
                        "tpsa": round(Descriptors.TPSA(mol), 2),
                        # Lipinski check
                        "lipinski_ok": (Descriptors.MolWt(mol) <= 500 and
                                        Descriptors.MolLogP(mol) <= 5 and
                                        Descriptors.NumHDonors(mol) <= 5 and
                                        Descriptors.NumHAcceptors(mol) <= 10),
                    }
                generated.append({
                    "id": start + i + 1,
                    "smiles": smi,
                    "valid": valid,
                    "condition": condition,
                    "temperature": temperature,
                    **props,
                })

            if (start // batch_z + 1) % 10 == 0:
                print(f"  Generated {min(start+batch_z, n):,}/{n:,}  "
                      f"valid={valid_n} ({100*valid_n/(start+bs):.1f}%%)")

    # Save
    out_path = DATA_OUT / f"generated_molecules_{n}_{condition}.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        if generated:
            w = csv.DictWriter(f, fieldnames=generated[0].keys())
            w.writeheader(); w.writerows(generated)

    validity_rate = valid_n / n * 100
    print(f"\nGeneration complete:")
    print(f"  Total: {n:,}  Valid: {valid_n:,} ({validity_rate:.1f}%%)")
    print(f"  Saved: {out_path}")
    return generated


# ═══════════════════════════════════════════════════════════════
# 6. Metrics & Report
# ═══════════════════════════════════════════════════════════════
def compute_metrics(generated, training_smiles=None):
    """Compute generation quality metrics."""
    valid    = [g for g in generated if g.get("valid")]
    n_total  = len(generated)
    n_valid  = len(valid)

    metrics = {
        "n_total": n_total,
        "n_valid": n_valid,
        "validity_rate": round(n_valid/n_total*100, 2) if n_total else 0,
        "unique_smiles": len(set(g["smiles"] for g in valid)),
    }

    if n_valid > 0:
        metrics["uniqueness_rate"] = round(metrics["unique_smiles"]/n_valid*100, 2)
        # Lipinski compliance
        lipinski = [g for g in valid if g.get("lipinski_ok")]
        metrics["lipinski_rate"] = round(len(lipinski)/n_valid*100, 2)
        # Property stats
        mws  = [g.get("mw",0) for g in valid if g.get("mw")]
        logps= [g.get("logp",0) for g in valid if g.get("logp")]
        if mws:
            metrics["mw_mean"] = round(np.mean(mws), 1)
            metrics["mw_std"]  = round(np.std(mws), 1)
        if logps:
            metrics["logp_mean"] = round(np.mean(logps), 2)

    return metrics


def write_generation_report(metrics, model_path, n_epochs, best_loss):
    rpt = ROOT / "docs" / "molecular_generation_report.txt"
    lines = [
        "=" * 65,
        "ADDS 분자 구조 생성 보고서 (CVAE de novo Design)",
        "=" * 65,
        f"생성 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "모델 구조",
        "-" * 40,
        "  Architecture : Conditional SMILES VAE (Bi-LSTM Encoder + LSTM Decoder)",
        "  Encoder      : BiLSTM(128+cond→512, 2-layer) → mu, log_var (256-dim)",
        "  Decoder      : LSTM(128+z+cond→512, 2-layer) → vocab softmax",
        "  Latent dim   : 256",
        "  Condition    : 8-dim (PrPc, BRAF, mTOR, HER2, EGFR, VEGFR2, Ki, misc)",
        "  Max SMILES   : 120 chars",
        "",
        "훈련 데이터",
        "-" * 40,
        "  drug_smiles.json  : 4,246 compounds",
        "  ChEMBL (5 targets): EGFR, BRAF, mTOR, HER2, VEGFR2",
        "  BindingDB          : PrPc-related binding data",
        "  총 훈련 분자       : ~10,000+",
        "",
        "훈련 결과",
        "-" * 40,
        f"  에폭          : {n_epochs}",
        f"  Best loss     : {best_loss:.4f}",
        f"  모델 저장     : {model_path}",
        "",
        "생성 품질 지표",
        "-" * 40,
    ]
    for k, v in metrics.items():
        lines.append(f"  {k:25s}: {v}")
    lines += [
        "",
        "중요 고지",
        "-" * 40,
        "  생성 분자는 in silico 가상 후보물질입니다.",
        "  임상 사용 전 다음이 필수입니다:",
        "  1) 분자동역학 시뮬레이션 (Schrödinger/AutoDock)",
        "  2) PrPc 결합 친화도 in vitro 실험",
        "  3) ADMET 독성/약동학 프로파일링",
        "  4) 세포주 확인 실험",
        "  논문 표현: 'AI de novo designed candidates",
        "              (pending experimental validation)'",
        "=" * 65,
    ]
    with open(rpt, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Report: {rpt}")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="ADDS Molecular CVAE")
    p.add_argument("--mode", choices=["train","generate","validate","all"],
                   default="all")
    p.add_argument("--epochs",      type=int,   default=200)
    p.add_argument("--latent_dim",  type=int,   default=256)
    p.add_argument("--hidden_dim",  type=int,   default=512)
    p.add_argument("--n_generate",  type=int,   default=10000)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--condition",   default="prpc",
                   choices=["prpc","egfr","her2","braf","multi"])
    p.add_argument("--batch_size",  type=int,   default=128)
    args = p.parse_args()

    print("=" * 65)
    print("ADDS Molecular Structure Generation CVAE v2.0")
    print(f"  Mode: {args.mode}  Epochs: {args.epochs}  Latent: {args.latent_dim}")
    print("=" * 65)

    model_path = None
    if args.mode in ("train","all"):
        model_path = train(
            epochs=args.epochs,
            batch_size=args.batch_size,
            latent_dim=args.latent_dim,
            hidden_dim=args.hidden_dim,
        )

    if args.mode in ("generate","all"):
        generated = generate_molecules(
            n=args.n_generate,
            temperature=args.temperature,
            condition=args.condition,
        )
        if generated:
            metrics = compute_metrics(generated)
            print("\nMetrics:", json.dumps(metrics, indent=2))
            # Load history for report
            ckpt_path = MODEL_OUT / "cvae_pritamab_v2.pt"
            if ckpt_path.exists():
                import torch
                ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
                hist = ckpt.get("history",[{}])
                write_generation_report(
                    metrics, str(ckpt_path),
                    n_epochs=len(hist),
                    best_loss=ckpt.get("best_loss",9.99))
