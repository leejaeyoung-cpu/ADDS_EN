"""Verification script - ASCII only, v2"""
import re, numpy as np

SCRIPT = r'f:\ADDS\figures\pritamab_synergy_map_bliss_matrix.py'
with open(SCRIPT, encoding='utf-8') as f:
    src = f.read()

# Build a "rendered-only" version: strip comment lines and docstring
non_comment_lines = []
in_docstring = False
for line in src.split('\n'):
    stripped = line.strip()
    if stripped.startswith('"""') or stripped.endswith('"""'):
        in_docstring = not in_docstring
        continue
    if in_docstring:
        continue
    if stripped.startswith('#'):
        continue
    non_comment_lines.append(line)
src_rendered = '\n'.join(non_comment_lines)

sot_m = re.search(r'SOT\s*=\s*\{(.+?)\}', src, re.DOTALL)
ns = {}
exec("SOT = {" + sot_m.group(1) + "}", ns)
SOT = ns['SOT']
KRAS_COLS = ['G12D','G12V','G12C','G13D','WT']

pb_m = re.search(r'PARTNER_BLISS\s*=\s*\{(.+?)\}', src, re.DOTALL)
ns2 = {}
exec("PB = {" + pb_m.group(1) + "}", ns2)
PB = ns2['PB']

PASS = 0; FAIL = 0; WARN = 0; issues = []
def ok(m):   global PASS; PASS+=1; print("  OK  "+m)
def fail(m): global FAIL; FAIL+=1; issues.append("FAIL: "+m); print("  !!  "+m)
def warn(m): global WARN; WARN+=1; issues.append("WARN: "+m); print("  **  "+m)

print("="*60)
print("SELF-VERIFICATION: Pritamab Synergy Figure v3.2")
print("="*60)

# 1. SOT integrity
print("\n[1] SOT Data Integrity")
for lbl, row in SOT.items():
    if not (1 <= row[0] <= 30): fail(f"{lbl} G12D={row[0]} out of range")
    if not (1 <= row[5] <= 50): fail(f"{lbl} ev={row[5]} suspect")
    if lbl.startswith('Pritamab') and row[0] <= row[4]:
        warn(f"{lbl}: G12D({row[0]})<= WT({row[4]})")
ok(f"All {len(SOT)} SOT entries plausible")

# 2. Panel A vs SOT
print("\n[2] Panel A PARTNER_BLISS vs SOT G12D")
agent_map = {
    'Oxaliplatin':'Pritamab + Oxaliplatin','5-FU':'Pritamab + 5-FU',
    'Irinotecan':'Pritamab + Irinotecan','Bevacizumab':'Pritamab + Bevacizumab',
    'TAS-102':'Pritamab + TAS-102','FOLFOX':'Pritamab + FOLFOX',
    'FOLFIRI':'Pritamab + FOLFIRI','FOLFOXIRI':'Pritamab + FOLFOXIRI',
}
mismatches = 0
for ag, key in agent_map.items():
    sot_v = SOT[key][0]; pb_v = PB.get(ag)
    if pb_v is None: fail(f"PB missing '{ag}'"); mismatches+=1
    elif abs(sot_v - pb_v) > 0.01: fail(f"'{ag}' PB={pb_v} != SOT={sot_v}"); mismatches+=1
if mismatches == 0: ok("All 8 Panel A values match SOT G12D exactly")

# 3. Panel C ranking
print("\n[3] Panel C ranking monotonicity")
RANKED = sorted(SOT.keys(), key=lambda x: -SOT[x][0])
vals = [SOT[l][0] for l in RANKED]
if all(vals[i] >= vals[i+1] for i in range(len(vals)-1)):
    ok(f"Descending: {vals[0]} ... {vals[-1]}")
else:
    fail("Not monotonically descending!")

# 4. Heatmap vmax
print("\n[4] Heatmap vmax clipping")
vm = re.search(r'TwoSlopeNorm\(vmin=([^,]+),\s*vcenter=([^,]+),\s*vmax=([^)]+)\)', src)
if vm:
    vmax = float(vm.group(3))
    dmax = max(SOT[l][0] for l in SOT)
    if vmax < dmax: fail(f"vmax={vmax} < data_max={dmax} => CLIPPING!")
    elif vmax < dmax + 0.3: warn(f"vmax={vmax} very close to data_max={dmax}")
    else: ok(f"vmax={vmax} > data_max={dmax} => no clipping")
else: fail("TwoSlopeNorm not found")

# 5. Bubble variance
print("\n[5] Panel D bubble size ev^2*5")
evs = sorted(set(SOT[l][5] for l in SOT))
szs = {e: e**2*5 for e in evs}
ratio = max(szs.values()) / min(szs.values())
print(f"  n={evs[0]} => size={szs[evs[0]]}, n={evs[-1]} => size={szs[evs[-1]]}, ratio={ratio:.1f}x")
if ratio >= 10: ok(f"{ratio:.1f}x => clearly distinguishable")
elif ratio >= 4: ok(f"{ratio:.1f}x => acceptable")
else: warn(f"{ratio:.1f}x may be insufficient")

# 6. Edge colormap
print("\n[6] Panel A edge colormap coverage")
bm = re.search(r'b_lo,\s*b_hi\s*=\s*([\d.]+),\s*([\d.]+)', src)
if bm:
    b_lo, b_hi = float(bm.group(1)), float(bm.group(2))
    amin, amax = min(PB.values()), max(PB.values())
    pct = (amax - amin) / (b_hi - b_lo) * 100
    t_min = (amin - b_lo)/(b_hi - b_lo)
    t_max = (amax - b_lo)/(b_hi - b_lo)
    lw_diff = 6.5*(t_max - t_min)
    print(f"  cmap=[{b_lo},{b_hi}] data=[{amin},{amax}] cov={pct:.0f}%")
    print(f"  t: {t_min:.2f}->{t_max:.2f}  lw_contrast: {lw_diff:.1f}px")
    if pct < 30: warn(f"Only {pct:.0f}% cmap used => poor color contrast")
    elif pct < 50: warn(f"{pct:.0f}% cmap used => moderate (acceptable)")
    else: ok(f"{pct:.0f}% cmap used => good contrast")
    if lw_diff >= 2.0: ok(f"lw contrast {lw_diff:.1f}px is visible")
    else: warn(f"lw contrast {lw_diff:.1f}px may be too small")

# 7. Text checks (use src_rendered to exclude comments)
print("\n[7] Required text elements (comments excluded)")
checks = [
    ("Heuristic cutoff in rendered code",   "Heuristic" in src_rendered),
    ("No 'clinical threshold' in rendered",  "clinical threshold" not in src_rendered.lower()),
    ("anti-PrPc mAb label",                 "anti-PrP" in src and "mAb" in src),
    ("TAS-102 correct (not IAS-)",           "TAS-102" in src and "IAS-102" not in src),
    ("Hobbs 2020 citation",                  "Hobbs" in src and "2020" in src),
    ("Yaeger citation",                      "Yaeger" in src),
    ("KRAS-targeted combination",            "KRAS-targeted combination" in src),
    ("Evidence count footnote",              "Evidence count" in src),
    ("Panel B dashed block legend",          "Pritamab combination block" in src),
    ("Pritamab+partner annotation",          "Pritamab + partner" in src),
    ("Partner-only annotation",              "Partner-only" in src),
    ("facecolor=white",                      "facecolor='white'" in src),
    ("No dark background",                   "#0D0D1E" not in src),
    ("n=X labels Panel D",                   "f'n={" in src),
    ("ev**2 * 5 scaling",                    "ev**2 * 5" in src),
]
for label, result in checks:
    if result: ok(label)
    else: fail(label)

# 8. Structural
print("\n[8] Structural/logic checks")
n_legends = src.count('ax_bub.legend(')
if n_legends > 1:
    warn(f"Panel D has {n_legends} legend() calls => only last shown!")
else:
    ok("Panel D single legend() call")

PRIT = [l for l in RANKED if l.startswith('Pritamab')]
n_p = len(PRIT)
mrtx_rank = RANKED.index('MRTX1133 + Oxaliplatin')
if mrtx_rank < n_p:
    fail(f"MRTX1133 rank {mrtx_rank+1} INSIDE Pritamab block (ends row {n_p})")
else:
    ok(f"MRTX1133 rank {mrtx_rank+1} OUTSIDE Pritamab block (rows 1-{n_p})")

em = re.search(r'errs\s*=\s*np\.array\(\[([^\]]+)\]', src, re.DOTALL)
if em:
    n_e = len(re.findall(r'[\d.]+', em.group(1)))
    if n_e < len(SOT): warn(f"errs {n_e} < data {len(SOT)} (uses [:N] slice)")
    else: ok(f"errs array length {n_e} >= data {len(SOT)}")

sf = re.search(r"savefig[^)]+facecolor='([^']+)'", src)
if sf:
    if sf.group(1) == 'white': ok("savefig facecolor='white'")
    else: fail(f"savefig facecolor='{sf.group(1)}'")
else:
    warn("savefig facecolor not found")

print("\n"+"="*60)
print(f"RESULT: {PASS} OK  {WARN} WARN  {FAIL} FAIL")
print("="*60)
if issues:
    print("\nIssues:")
    for x in issues: print(f"  {x}")
else:
    print("CLEAN PASS - no issues.")
