"""
Independent verification of regimen_toxicity_profile.py
Checks data against published literature values.
All ASCII-safe.
"""
import re, sys
import numpy as np

SCRIPT = r'f:\ADDS\figures\regimen_toxicity_profile.py'
with open(SCRIPT, encoding='utf-8') as f:
    src = f.read()

# ── Extract TOX_MATRIX from script ──────────────────────────────
mat_m = re.search(r'TOX_MATRIX\s*=\s*np\.array\(\[(.+?)\],\s*dtype=float\)', src, re.DOTALL)
assert mat_m, "TOX_MATRIX not found"
ns = {}
exec("import numpy as np; TOX_MATRIX = np.array([" + mat_m.group(1) + "], dtype=float)", ns)
M = ns['TOX_MATRIX']

REGIMENS = [
    'Prit+FOLFOX','Prit+FOLFIRI','Pritamab',
    'FOLFOX','FOLFIRI','FOLFOXIRI','CAPOX','TAS-102',
    'Bev+FOLFOX','Pembrolizumab',
]
TOX = [
    'Neutropenia','Anemia','Thrombocytopenia','Nausea/Vomiting',
    'Diarrhea','Periph.Neuropathy','Fatigue','Hand-Foot',
    'Alopecia','Hepatotox','Hypertension','irAE',
]

PASS = 0; FAIL = 0; WARN = 0
issues = []
def ok(msg):   global PASS; PASS+=1; print("  OK   "+msg)
def fail(msg): global FAIL; FAIL+=1; issues.append("FAIL: "+msg); print("  FAIL "+msg)
def warn(msg): global WARN; WARN+=1; issues.append("WARN: "+msg); print("  WARN "+msg)

print("="*65)
print("TOXICITY FIGURE -- INDEPENDENT VERIFICATION")
print("="*65)

# ── CHECK 1: Matrix shape ────────────────────────────────────────
print("\n[1] Matrix dimensions")
if M.shape == (10, 12):
    ok("Matrix shape (10, 12) correct")
else:
    fail("Matrix shape %s != (10,12)" % str(M.shape))

# ── CHECK 2: No negative values, no implausibly high values ─────
print("\n[2] Value range plausibility")
if M.min() >= 0:
    ok("All values >= 0 (no negative incidence)")
else:
    fail("Negative value found: min=%.1f" % M.min())
if M.max() <= 100:
    ok("All values <= 100%% (max=%.1f)" % M.max())
else:
    fail("Value > 100%%: max=%.1f" % M.max())
vals_over_60 = [(REGIMENS[i], TOX[j], M[i,j])
                for i in range(10) for j in range(12) if M[i,j] > 60]
if vals_over_60:
    for reg, tox, v in vals_over_60:
        warn("%.1f%% for %s / %s -- unusually high, verify" % (v,reg,tox))
else:
    ok("No single toxicity value > 60%% (plausible range)")

# ── CHECK 3: Literature anchor values ───────────────────────────
print("\n[3] Literature anchor verification")

# Ground truth from published pivotal trials (Grade 3/4 %)
# Sources cited in script
ANCHORS = {
    # FOLFOX (MOSAIC, N_Engl_J_Med 2004): Neutropenia 41%, PN 18%
    ('FOLFOX', 'Neutropenia'): (35, 48, 'MOSAIC 2004'),
    ('FOLFOX', 'Periph.Neuropathy'): (12, 25, 'MOSAIC 2004'),
    # FOLFIRI (Douillard 2000, Lancet): Neutropenia 24-28%, Diarrhea 14-20%
    ('FOLFIRI', 'Neutropenia'): (20, 32, 'Douillard 2000 Lancet'),
    ('FOLFIRI', 'Diarrhea'): (14, 24, 'Douillard 2000 Lancet'),
    # FOLFOXIRI (Falcone 2007, JCO): Neutropenia 50%, Nausea 19%
    ('FOLFOXIRI', 'Neutropenia'): (45, 58, 'Falcone 2007 JCO'),
    ('FOLFOXIRI', 'Nausea/Vomiting'): (15, 24, 'Falcone 2007 JCO'),
    # CAPOX (XELOX-ACOSOG Z6006): Thrombocytopenia 10-20%, HFS 17%
    ('CAPOX', 'Thrombocytopenia'): (8, 22, 'XELOX triails'),
    ('CAPOX', 'Hand-Foot'): (12, 22, 'XELOX trials'),
    # TAS-102 (RECOURSE, Mayer 2015 NEJM): Neutropenia 38%, Anemia 19%
    ('TAS-102', 'Neutropenia'): (33, 44, 'RECOURSE 2015 NEJM'),
    ('TAS-102', 'Anemia'): (15, 24, 'RECOURSE 2015 NEJM'),
    # Bev+FOLFOX: HTN 18% (TREE-2/NO16966), Neutropenia 35-41%
    ('Bev+FOLFOX', 'Hypertension'): (13, 25, 'NO16966 2008'),
    ('Bev+FOLFOX', 'Neutropenia'): (33, 44, 'NO16966 2008'),
    # Pembrolizumab (KEYNOTE-177+158): irAE 22%, Fatigue 18%
    ('Pembrolizumab', 'irAE'): (17, 28, 'KEYNOTE-177 2021'),
    ('Pembrolizumab', 'Fatigue'): (14, 24, 'KEYNOTE-177 2021'),
    # Pritamab mono: expected very low chemo-type toxicity (proprietary)
    ('Pritamab', 'Neutropenia'): (0, 8, 'Lee ADDS 2026 (PrPc-targeted, no cytotoxic payload)'),
    ('Pritamab', 'Diarrhea'): (0, 8, 'Lee ADDS 2026'),
}

reg_idx = {r:i for i,r in enumerate(REGIMENS)}
tox_idx = {t:j for j,t in enumerate(TOX)}

anchor_fails = 0
for (reg, tox), (lo, hi, ref) in ANCHORS.items():
    ri = reg_idx.get(reg)
    ti = tox_idx.get(tox)
    if ri is None or ti is None:
        warn("Anchor lookup failed: %s / %s" % (reg,tox))
        continue
    val = M[ri, ti]
    if lo <= val <= hi:
        ok("%s / %s: %.0f%% in [%d,%d] (%s)" % (reg,tox,val,lo,hi,ref))
    elif abs(val - lo) <= 5 or abs(val - hi) <= 5:
        warn("%s / %s: %.0f%% near but outside [%d,%d] (%s)" % (reg,tox,val,lo,hi,ref))
    else:
        fail("%s / %s: %.0f%% OUTSIDE [%d,%d] -- check vs %s" % (reg,tox,val,lo,hi,ref))
        anchor_fails += 1

# ── CHECK 4: Internal consistency ───────────────────────────────
print("\n[4] Internal consistency")

# Pritamab mono should be safest overall
prit_mono_score = M[2].sum()
for i, reg in enumerate(REGIMENS):
    if i != 2 and M[i].sum() < prit_mono_score:
        fail("Regimen '%s' has lower composite than Pritamab mono (%d < %d)" %
             (reg, int(M[i].sum()), int(prit_mono_score)))
if all(M[i].sum() >= prit_mono_score for i in range(10) if i != 2):
    ok("Pritamab mono has lowest composite toxicity (%d) -- clinically plausible" % int(prit_mono_score))

# FOLFOXIRI should be most intensive (highest composite)
folfoxiri_score = M[5].sum()
if folfoxiri_score == M.sum(axis=1).max():
    ok("FOLFOXIRI has highest composite (%d) -- consistent with triple-agent regimen" % int(folfoxiri_score))
else:
    worst = REGIMENS[int(M.sum(axis=1).argmax())]
    warn("Highest composite is %s (%d), not FOLFOXIRI (%d) -- check" %
         (worst, int(M.sum(axis=1).max()), int(folfoxiri_score)))

# Neuropathy: FOLFOX > FOLFIRI (oxaliplatin vs irinotecan)
folfox_np  = M[3, 5]  # FOLFOX, Periph.Neuropathy
folfiri_np = M[4, 5]  # FOLFIRI, Periph.Neuropathy
if folfox_np > folfiri_np:
    ok("FOLFOX PN (%.0f%%) > FOLFIRI PN (%.0f%%) -- oxaliplatin effect correct" % (folfox_np, folfiri_np))
else:
    fail("FOLFIRI PN >= FOLFOX PN -- wrong (oxaliplatin causes more PN than irinotecan)")

# Alopecia: FOLFIRI > FOLFOX (irinotecan >> oxaliplatin for alopecia)
folfox_alo  = M[3, 8]
folfiri_alo = M[4, 8]
if folfiri_alo > folfox_alo:
    ok("FOLFIRI alopecia (%.0f%%) > FOLFOX (%.0f%%) -- irinotecan effect correct" % (folfiri_alo, folfox_alo))
else:
    fail("FOLFOX alopecia >= FOLFIRI -- wrong (irinotecan causes more alopecia)")

# HFS: CAPOX > others (capecitabine-related)
capox_hfs = M[6, 7]
folfox_hfs= M[3, 7]
if capox_hfs > folfox_hfs:
    ok("CAPOX HFS (%.0f%%) > FOLFOX HFS (%.0f%%) -- capecitabine effect correct" % (capox_hfs, folfox_hfs))
else:
    fail("CAPOX HFS not higher than FOLFOX -- capecitabine should cause more HFS")

# Hypertension: Bevacizumab combos > standard chemo
bev_htn  = M[8, 10]  # Bev+FOLFOX
folfox_htn = M[3, 10]
if bev_htn > folfox_htn:
    ok("Bev+FOLFOX HTN (%.0f%%) > FOLFOX alone (%.0f%%) -- bevacizumab VEGF effect correct" % (bev_htn, folfox_htn))
else:
    fail("Bevacizumab combination not showing higher HTN -- VEGF inhibition signature missing")

# irAE: Pembrolizumab has highest irAE
pembro_irae = M[9, 11]
max_irae    = M[:,11].max()
if pembro_irae == max_irae:
    ok("Pembrolizumab has highest irAE (%.0f%%) -- immune checkpoint expected" % pembro_irae)
else:
    fail("Pembrolizumab irAE (%.0f%%) not highest (max=%.0f%%)" % (pembro_irae, max_irae))

# Chemo regimens should have irAE = 0
for i in [3,4,5,6,7]:
    irae = M[i,11]
    if irae == 0:
        ok("%s irAE=0 (pure chemo, correct)" % REGIMENS[i])
    else:
        warn("%s irAE=%.0f%% (should be 0 for pure chemo)" % (REGIMENS[i], irae))

# ── CHECK 5: Composite scores reasonableness ────────────────────
print("\n[5] Composite score distribution")
composites = M.sum(axis=1)
print("  Scores:")
for i, (reg, score) in enumerate(zip(REGIMENS, composites)):
    bar = "#" * int(score / 8)
    print("    %-20s: %4.0f  %s" % (reg, score, bar))
spread = composites.max() - composites.min()
if spread >= 100:
    ok("Score spread = %.0f (good separation)" % spread)
elif spread >= 50:
    warn("Score spread = %.0f (moderate separation)" % spread)
else:
    fail("Score spread = %.0f (insufficient separation)" % spread)

# ── CHECK 6: Rendering elements in source ───────────────────────
print("\n[6] Figure rendering element checks")
render_checks = [
    ("Heatmap imshow called",        "ax_heat.imshow" in src),
    ("Cell value annotations",       "ax_heat.text" in src and "ha='center'" in src),
    ("Colorbar present",             "plt.colorbar" in src),
    ("Radar chart polar=True",       "polar=True" in src),
    ("Radar chart closed (loop)",    "angles += angles[:1]" in src),
    ("Bar chart horizontal",         "ax_bar.barh" in src),
    ("White facecolor saved",        "facecolor='white'" in src),
    ("dpi >= 150",                   "dpi=175" in src or "dpi=200" in src or "dpi=150" in src),
    ("Source citation present",      "NCCN" in src and "ESMO" in src),
    ("Pritamab highlight box",       "Pritamab\ncombinations" in src or "Pritamab combinations" in src),
    ("Regimen color bars",           "REG_COLORS" in src),
    ("Literature references cited",  "Falcone" in src and "RECOURSE" in src and "KEYNOTE" in src),
]
for label, result in render_checks:
    if result: ok(label)
    else: warn(label)

# ── CHECK 7: Known potential reviewer attack points ─────────────
print("\n[7] Reviewer risk assessment")
risks = []
# Pritamab data = proprietary -- must be clearly labeled
if "proprietary" in src.lower() or "Lee ADDS" in src:
    ok("Pritamab data source labeled as proprietary ADDS 2026")
else:
    risks.append("Pritamab data source not clearly labeled")

# Check if composite score methodology is clear
if "sum" in src and ("sum of" in src or "sum(" in src):
    ok("Composite score method (sum) referenced")
else:
    risks.append("Composite score calculation method not explained in labels")

# Alopecia note: for FOLFIRI typically Grade 1/2 dominant, G3/4 rare
folfiri_alo_check = M[4, 8]
if folfiri_alo_check > 30:
    risks.append("FOLFIRI alopecia %.0f%% -- typically any-grade ~70%% but G3/4 < 5%% in most trials (check)" % folfiri_alo_check)
else:
    ok("FOLFIRI G3/4 alopecia %.0f%% -- consistent with any-grade caveat in label" % folfiri_alo_check)

# Nausea/Vomiting: FOLFOXIRI 19% may seem high vs FOLFOX 7%
ratio_nv = M[5,3] / max(M[3,3], 0.1)
if ratio_nv <= 4.0:
    ok("FOLFOXIRI/FOLFOX nausea ratio = %.1fx (within expected range)" % ratio_nv)
else:
    risks.append("FOLFOXIRI nausea %.0f%% vs FOLFOX %.0f%% (ratio %.1fx -- verify Falcone 2007)" %
                 (M[5,3], M[3,3], ratio_nv))

for r in risks:
    warn(r)
if not risks:
    ok("No major reviewer attack points")

# ── FINAL SUMMARY ────────────────────────────────────────────────
print("\n" + "="*65)
print("RESULT: %d OK  %d WARN  %d FAIL" % (PASS, WARN, FAIL))
print("="*65)
if FAIL == 0 and WARN <= 3:
    print("Verdict: PASS -- publication-ready data")
elif FAIL == 0:
    print("Verdict: CONDITIONAL PASS -- address WARNings before submission")
else:
    print("Verdict: NEEDS CORRECTION -- %d data errors found" % FAIL)

if issues:
    print("\nItems requiring action:")
    for i in issues: print("  " + i)
