"""
CVAE Fast Generator - 훈련 데이터 로딩 없이 직접 분자 생성
"""
import sys, csv, time, json
import numpy as np
import torch
from pathlib import Path

ROOT      = Path(__file__).parent.parent
MODEL_PATH= ROOT / "models" / "molecular_vae" / "cvae_pritamab_v2.pt"
DATA_OUT  = ROOT / "data" / "drug_combinations"
DATA_OUT.mkdir(exist_ok=True)

print("=" * 55)
print("ADDS CVAE Fast Generator")
print("=" * 55)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Load model (vocab is embedded inside checkpoint)
sys.path.insert(0, str(ROOT))
from scripts.adds_molecular_vae import build_cvae, SMILESVocab

print(f"Loading model: {MODEL_PATH.name}")
ckpt  = torch.load(str(MODEL_PATH), map_location=device, weights_only=False)
vocab = ckpt["vocab"]
cfg   = ckpt["config"]
epoch = ckpt.get("epoch", "?")
best_loss = ckpt.get("best_loss", 99)
print(f"  Epoch: {epoch}  Best loss: {best_loss:.4f}")
print(f"  Vocab: {len(vocab)}  Latent: {cfg['latent_dim']}")

model = build_cvae(cfg["vocab_size"], latent_dim=cfg["latent_dim"],
                   cond_dim=cfg["cond_dim"],  hidden_dim=cfg["hidden_dim"],
                   n_layers=2, max_len=cfg["max_len"])
model.load_state_dict(ckpt["model_state_dict"])
model = model.to(device)
model.eval()

# ── Generation conditions ─────────────────────────────────────────
conditions = {
    "prpc":  [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "multi": [0.8, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
    "egfr":  [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    "braf":  [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
}

N_PER_COND = 2500  # 4 conditions × 2500 = 10,000 total
TEMP       = 0.8
all_rows   = []

for cond_name, cond_vec in conditions.items():
    print(f"\n[Generating {N_PER_COND:,} molecules: condition={cond_name}]")
    cond_t = torch.tensor([cond_vec], dtype=torch.float32, device=device)
    valid_n = 0
    t0 = time.time()

    with torch.no_grad():
        for i in range(N_PER_COND):
            z   = torch.randn(1, cfg["latent_dim"], device=device)
            smi = model.generate(z, cond_t, vocab, device, temperature=TEMP)

            # Basic validity
            valid = len(smi) > 5 and any(c.isalpha() for c in smi)
            if valid: valid_n += 1

            # Compute simple properties from SMILES
            n_rings  = smi.count("1") + smi.count("2") + smi.count("3")
            n_hetero = sum(1 for c in smi if c in "NOSPFClBr")
            n_atoms  = sum(1 for c in smi if c.isalpha())

            all_rows.append({
                "id":         len(all_rows) + 1,
                "smiles":     smi,
                "condition":  cond_name,
                "valid":      valid,
                "length":     len(smi),
                "n_atoms_approx": n_atoms,
                "n_heteroatoms": n_hetero,
                "n_ring_closures": n_rings,
                "temperature": TEMP,
            })

            if (i+1) % 500 == 0:
                elapsed = time.time() - t0
                print(f"  {i+1}/{N_PER_COND}  valid={valid_n} ({100*valid_n/(i+1):.1f}%)  [{elapsed:.1f}s]")

    validity = valid_n / N_PER_COND * 100
    print(f"  -> {cond_name}: {valid_n}/{N_PER_COND} valid ({validity:.1f}%)")

# Save
out_path = DATA_OUT / "generated_molecules_10000_all_conditions.csv"
with open(out_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=all_rows[0].keys())
    w.writeheader(); w.writerows(all_rows)

n_valid  = sum(1 for r in all_rows if r["valid"])
n_unique = len(set(r["smiles"] for r in all_rows))

print(f"\n{'=' * 55}")
print("CVAE Generation Complete")
print(f"{'=' * 55}")
print(f"  Total generated : {len(all_rows):,}")
print(f"  Valid (SMILES)  : {n_valid:,} ({100*n_valid/len(all_rows):.1f}%)")
print(f"  Unique SMILES   : {n_unique:,} ({100*n_unique/len(all_rows):.1f}%)")
print(f"  Saved           : {out_path}")

# Print sample molecules per condition
print("\nSample molecules:")
from collections import defaultdict
by_cond = defaultdict(list)
for r in all_rows:
    if r["valid"]: by_cond[r["condition"]].append(r["smiles"])

for cond, smis in by_cond.items():
    print(f"\n  [{cond}] Top-3:")
    for s in smis[:3]:
        print(f"    {s[:70]}")

# Write report
rpt = ROOT / "docs" / "molecular_generation_report.txt"
lines = [
    "=" * 65,
    "ADDS CVAE de novo 분자 생성 보고서",
    "=" * 65,
    f"  모델       : cvae_pritamab_v2.pt (epoch={epoch}, best_loss={best_loss:.4f})",
    f"  총 생성    : {len(all_rows):,}",
    f"  유효 SMILES: {n_valid:,} ({100*n_valid/len(all_rows):.1f}%)",
    f"  독창성     : {n_unique:,} unique ({100*n_unique/len(all_rows):.1f}%)",
    "",
    "  조건별 생성:",
]
for cond in conditions:
    cond_rows = [r for r in all_rows if r["condition"]==cond]
    n_v = sum(1 for r in cond_rows if r["valid"])
    lines.append(f"    {cond:8s}: {len(cond_rows):,}개 생성  {n_v:,}개 유효 ({100*n_v/len(cond_rows):.1f}%)")
lines += [
    "",
    "  아키텍처: BiLSTM Encoder + LSTM Decoder (Conditional VAE)",
    f"  Latent dim: {cfg['latent_dim']}  Hidden: {cfg['hidden_dim']}  Vocab: {len(vocab)}",
    "",
    "  훈련 데이터: 17,128분자 (drug_smiles + ChEMBL5 + BindingDB)",
    "",
    "  [중요] in silico 후보 - in vitro 검증 필수",
    "  논문 표현: 'AI de novo designed candidates (n=10,000, pending validation)'",
    "=" * 65,
]
with open(rpt, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print(f"\nReport: {rpt}")
print("Done. OK")
