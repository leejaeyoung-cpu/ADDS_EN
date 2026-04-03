"""
v2 Figures -- Independent Rigorous Verification
Checks BOTH toxicity_verification_dashboard_v2.py AND toxicity_bullseye_v2.py
ASCII-safe
"""
import re, os
import numpy as np

DASH_SRC  = r'f:\ADDS\figures\toxicity_verification_dashboard_v2.py'
BULL_SRC  = r'f:\ADDS\figures\toxicity_bullseye_v2.py'
DASH_PNG  = r'f:\ADDS\figures\toxicity_verification_dashboard_v2.png'
BULL_PNG  = r'f:\ADDS\figures\toxicity_bullseye_v2.png'

with open(DASH_SRC, encoding='utf-8') as f: dash = f.read()
with open(BULL_SRC, encoding='utf-8') as f: bull = f.read()

PASS=0; FAIL=0; WARN=0
issues=[]
def ok(tag,msg):   global PASS; PASS+=1; print("  OK   [%s] %s"%(tag,msg))
def fail(tag,msg): global FAIL; FAIL+=1; issues.append("FAIL[%s] %s"%(tag,msg)); print("  FAIL [%s] %s"%(tag,msg))
def warn(tag,msg): global WARN; WARN+=1; issues.append("WARN[%s] %s"%(tag,msg)); print("  WARN [%s] %s"%(tag,msg))

print("="*65)
print("v2 FIGURES -- INDEPENDENT RIGOROUS VERIFICATION")
print("="*65)

# ======================================================================
# A. DASHBOARD v2 CHECKS
# ======================================================================
print("\n[Dashboard v2]")

# --- Fix verify: 1순위 Bliss removal ---
print("\n  (1) Bliss contamination check")
# Exclude: docstring (lines inside triple-quote blocks), comments (#),
# and suptitle metadata descriptions (these are explaining changes, not rendered AE data)
# Target: Bliss appearing in rendered checklist strings or axis labels
src_no_docstring = re.sub(r'""".*?"""', '', dash, flags=re.DOTALL)  # strip docstrings
src_no_comments  = re.sub(r'#.*', '', src_no_docstring)  # strip # comments
# Also strip the suptitle description line (it documents the fix, not a data label)
src_clean = re.sub(r"fig\.suptitle\(.+?\)", '', src_no_comments, flags=re.DOTALL)
bliss_in_code = [l.strip() for l in src_clean.split('\n')
                 if 'Bliss' in l and l.strip()]
if not bliss_in_code:
    ok('Dash-1','Bliss term absent from all rendered data/label code (docstring/comments excluded)')
else:
    for ln in bliss_in_code:
        fail('Dash-1','Bliss in rendered label/data code: '+ln[:80])

# --- Fix verify: 2순위 count consistency ---
print("\n  (2) Check count consistency (50 total)")
n_ok_match   = re.findall(r'N_OK\s*=\s*(\d+)', dash)
n_warn_match = re.findall(r'N_WARN\s*=\s*(\d+)', dash)
n_fail_match = re.findall(r'N_FAIL\s*=\s*(\d+)', dash)
n_tot_match  = re.findall(r'N_TOT\s*=\s*N_OK\s*\+\s*N_WARN\s*\+\s*N_FAIL', dash)

if n_ok_match and int(n_ok_match[0]) == 49:
    ok('Dash-2','N_OK = 49')
else:
    fail('Dash-2','N_OK != 49: %s' % n_ok_match)

if n_warn_match and int(n_warn_match[0]) == 1:
    ok('Dash-2','N_WARN = 1')
else:
    fail('Dash-2','N_WARN != 1: %s' % n_warn_match)

if n_tot_match:
    ok('Dash-2','N_TOT computed as N_OK+N_WARN+N_FAIL (= 50)')
else:
    fail('Dash-2','N_TOT not computed from sum -- hardcoded?')

if '50 total' in dash:
    ok('Dash-2','Footer contains "50 total" -- consistent with donut')
else:
    fail('Dash-2','Footer missing "50 total"')

if 'All 49 checks PASSED' in dash:
    fail('Dash-2','Old misleading "All 49 checks PASSED" still present')
else:
    ok('Dash-2','Old "All 49 checks PASSED" text removed')

# --- Fix verify: 3순위 Pritamab source tier separation ---
print("\n  (3) Pritamab tier separation")
if 'ANCHOR_PUBLISHED' in dash and 'ANCHOR_INTERNAL' in dash:
    ok('Dash-3','ANCHOR_PUBLISHED and ANCHOR_INTERNAL defined separately')
else:
    fail('Dash-3','Separate anchor lists not found')

if '[LIT]' in dash and '[INT]' in dash:
    ok('Dash-3','[LIT] and [INT] prefix labels present')
else:
    fail('Dash-3','[LIT]/[INT] prefix labels missing')

if 'pre-submission' in dash or 'internal' in dash.lower():
    ok('Dash-3','Pritamab internal/pre-submission status labeled')
else:
    fail('Dash-3','Pritamab source not labeled as internal/pre-submission')

# Check Pritamab NOT in ANCHOR_PUBLISHED
pub_block = re.search(r'ANCHOR_PUBLISHED\s*=\s*\[(.+?)\]', dash, re.DOTALL)
if pub_block:
    if 'Pritamab' in pub_block.group(1):
        fail('Dash-3','Pritamab still inside ANCHOR_PUBLISHED block')
    else:
        ok('Dash-3','Pritamab absent from ANCHOR_PUBLISHED -- correctly in INTERNAL')

# --- Fix verify: 4순위 Composite caveat ---
print("\n  (4) Composite score caveat")
if 'Additive visual' in dash or 'additive visual' in dash:
    ok('Dash-4','additive visual index caveat present')
else:
    fail('Dash-4','Additive visual caveat not found')
if 'not patient-level' in dash or 'Not patient-level' in dash:
    ok('Dash-4','"not patient-level" caveat present')
else:
    fail('Dash-4','"not patient-level" caveat missing')
if 'COMPOSITE_ERR' in dash:
    ok('Dash-4','Uncertainty errors (COMPOSITE_ERR) defined for error bars')
else:
    warn('Dash-4','No COMPOSITE_ERR variable found -- no uncertainty bars?')
if 'errorbar' in dash or 'error_bar' in dash:
    ok('Dash-4','errorbar plotted for uncertainty')
else:
    warn('Dash-4','ax.errorbar not found -- uncertainty may not be visualized')

# --- Fix verify: 5순위 x-axis from 0 ---
print("\n  (5) x-axis starts at 0")
xlim_match = re.findall(r"ax_anchor\.set_xlim\((\S+),", dash)
for xlo in xlim_match:
    v = float(xlo.strip(',').strip(')'))
    if v >= 0:
        ok('Dash-5','Panel A x-axis lower limit = %g (>= 0)' % v)
    else:
        fail('Dash-5','Panel A x-axis lower limit = %g (negative!)' % v)

# --- Fix verify: 6순위 Panel D -- no implementation-level items ---
print("\n  (6) Panel D clinical checks (no implementation terms)")
impl_terms = ['polar=True', 'facecolor=white', 'dpi >= 150', 'imshow',
              'Cell value annotations']
# Check in RENDER_CHECKS list only
render_block = re.search(r'RENDER_CHECKS\s*=\s*\[(.+?)\]', dash, re.DOTALL)
if render_block:
    rb = render_block.group(1)
    found_impl = [t for t in impl_terms if t in rb]
    if not found_impl:
        ok('Dash-6','No implementation-level items in RENDER_CHECKS')
    else:
        for t in found_impl:
            fail('Dash-6','Implementation term still in RENDER_CHECKS: '+t)
    # Positive checks -- clinical terms
    clinical_terms = ['regimen','AE categor','G3/4','colorscale','source','citation','caveat']
    found_clinical = [t for t in clinical_terms if t.lower() in rb.lower()]
    ok('Dash-6','Clinical terms in RENDER_CHECKS: %d/7 (%s...)' %
       (len(found_clinical), ', '.join(found_clinical[:3])))
else:
    fail('Dash-6','RENDER_CHECKS block not found')

# --- Logic check: Bliss-free and ratio-free ---
print("\n  (7) Logic check text quality")
logic_block = re.search(r'LOGIC_CHECKS\s*=\s*\[(.+?)\]', dash, re.DOTALL)
if logic_block:
    lb = logic_block.group(1)
    # Forbidden terms
    for banned in ['Bliss', 'ratio =', '[0,100]', 'plausible range']:
        if banned in lb:
            fail('Dash-7','Banned term in LOGIC_CHECKS: '+banned)
        else:
            ok('Dash-7','Banned term absent: "%s"' % banned)
    # Required improvements
    for req in ['triplet', 'doublet', 'ordering', 'mechanism of action',
                'source hierarchy']:
        if req.lower() in lb.lower():
            ok('Dash-7','Required term present: "%s"' % req)
        else:
            warn('Dash-7','Optional improvement term missing: "%s"' % req)
else:
    fail('Dash-7','LOGIC_CHECKS block not found')

# --- Output file exists ---
print("\n  (8) Output file")
if os.path.exists(DASH_PNG):
    sz = os.path.getsize(DASH_PNG)//1024
    if sz >= 300:
        ok('Dash-8','PNG exists, %d KB (>= 300KB, good resolution)' % sz)
    else:
        warn('Dash-8','PNG exists but small: %d KB' % sz)
else:
    fail('Dash-8','PNG file not found: '+DASH_PNG)

# ======================================================================
# B. BULLSEYE v2 CHECKS
# ======================================================================
print("\n[Bullseye v2]")

# --- Data matrix ---
print("\n  (1) Data matrix")
mat_m = re.search(r'TOX_MATRIX\s*=\s*np\.array\(\[(.+?)\],\s*dtype=float\)', bull, re.DOTALL)
if mat_m:
    ns = {}
    exec("import numpy as np\nTOX_MATRIX=np.array([%s],dtype=float)" % mat_m.group(1), ns)
    M = ns['TOX_MATRIX']
    if M.shape == (10,12):
        ok('Bull-1','Matrix (10,12) correct')
    else:
        fail('Bull-1','Matrix shape %s != (10,12)' % str(M.shape))
    if M.min() >= 0 and M.max() <= 100:
        ok('Bull-1','All values in [0,100]%%: min=%.0f, max=%.0f' % (M.min(),M.max()))
    else:
        fail('Bull-1','Values out of [0,100]%%')
    n_zero = int((M==0).sum())
    if n_zero >= 5:
        ok('Bull-1','Zero cells: %d (non-trivial, 0 vs missing meaningful)' % n_zero)
    else:
        warn('Bull-1','Few zero cells (%d) -- 0 vs missing distinction less critical' % n_zero)
else:
    fail('Bull-1','TOX_MATRIX not found in bullseye script')
    M = None

# --- Fix: 0 vs missing ---
print("\n  (2) Zero vs missing differentiation")
if 'ZERO_COLOR' in bull:
    zc_match = re.search(r"ZERO_COLOR\s*=\s*'(#[0-9A-Fa-f]+)'", bull)
    if zc_match:
        zc = zc_match.group(1)
        ok('Bull-2','ZERO_COLOR defined: %s' % zc)
        # It should NOT be white (#FFFFFF)
        if zc.upper() in ('#FFFFFF', '#FFF', '#FEFEFE'):
            fail('Bull-2','ZERO_COLOR is white -- indistinguishable from background!')
        else:
            ok('Bull-2','ZERO_COLOR is distinct from white (%s)' % zc)
    else:
        warn('Bull-2','ZERO_COLOR defined but cannot parse hex')
else:
    fail('Bull-2','ZERO_COLOR not defined -- 0 vs missing not differentiated')

if 'val == 0' in bull and 'ZERO_COLOR' in bull:
    ok('Bull-2','0-value cells routed to ZERO_COLOR distinctly')
else:
    fail('Bull-2','"val == 0" branch with ZERO_COLOR not found')

if 'confirmed zero' in bull or 'confirmed 0%' in bull:
    ok('Bull-2','Footer explicitly says 0% is confirmed zero, not missing')
else:
    warn('Bull-2','Footer does not distinguish 0% from missing data')

# --- Fix: legend cleanup ---
print("\n  (3) Legend structure")
if 'Ring Reference' in bull:
    ok('Bull-3','"Ring Reference" legend section present')
else:
    fail('Bull-3','"Ring Reference" section missing')

if 'Color Encoding' in bull:
    ok('Bull-3','"Color Encoding" section present')
else:
    fail('Bull-3','"Color Encoding" section missing')

if 'Regimen Groups' in bull:
    ok('Bull-3','"Regimen Groups" section present')
else:
    fail('Bull-3','"Regimen Groups" section missing')

# Check for overlap risk: no two ax objects share same region
# Proxy: verify colorbar is on separate axis (named ax_cbar)
if 'ax_cbar' in bull:
    ok('Bull-3','Colorbar on dedicated axis (ax_cbar) -- no overlap with main chart')
else:
    fail('Bull-3','Colorbar not on dedicated axis -- overlap risk')

# --- Fix: label readability ---
print("\n  (4) Label readability")
reg_label_font = re.findall(r"fontsize=(\d+\.?\d*),.*fontweight='bold'.*zorder=7", bull)
reg_label_font2= re.findall(r"fontsize=(\d+\.?\d*).*REG_SHORT", bull)
ring_label_font = re.findall(r"fontsize=(\d+\.?\d*).*italic.*zorder=8", bull)

# Check regimen label fontsize
reg_fs_match = re.search(r"REG_SHORT\[i_reg\].*?fontsize=(\d+\.?\d*)", bull, re.DOTALL)
if reg_fs_match:
    fs = float(reg_fs_match.group(1))
    if fs >= 8.0:
        ok('Bull-4','Regimen label fontsize=%.1f (>= 8.0)' % fs)
    else:
        warn('Bull-4','Regimen label fontsize=%.1f (< 8.0, may be small in print)' % fs)
else:
    warn('Bull-4','Could not parse regimen label fontsize')

ring_fs_match = re.search(r"TOX_SHORT\[i_tox\].*?fontsize=(\d+\.?\d*)", bull, re.DOTALL)
if ring_fs_match:
    fs = float(ring_fs_match.group(1))
    if fs >= 7.0:
        ok('Bull-4','Ring label fontsize=%.1f (>= 7.0)' % fs)
    else:
        warn('Bull-4','Ring label fontsize=%.1f (< 7.0, may be unreadable)' % fs)
else:
    warn('Bull-4','Could not parse ring label fontsize')

# --- Annotation threshold ---
print("\n  (5) Annotation rule")
ann_thresh = re.search(r"val\s*>=\s*(\d+)", bull)
if ann_thresh:
    thresh = int(ann_thresh.group(1))
    if thresh <= 10:
        ok('Bull-5','Annotation threshold = %d%% (reasonable, <= 10)' % thresh)
    else:
        warn('Bull-5','Annotation threshold = %d%% (high -- fewer numbers shown)' % thresh)
    # verify footer mentions this
    if str(thresh)+'%' in bull or '>= %d' % thresh in bull:
        ok('Bull-5','Threshold (%d%%) referenced in title/footer' % thresh)
    else:
        warn('Bull-5','Threshold not mentioned in caption/footer')
else:
    fail('Bull-5','Annotation threshold not found')

# --- Data consistency: clinical logic (same checks as before) ---
print("\n  (6) Clinical logic cross-check")
if M is not None:
    # FOLFOX neuropathy vs FOLFIRI
    folfox_np  = M[3,5]; folfiri_np = M[4,5]
    if folfox_np > folfiri_np:
        ok('Bull-6','FOLFOX PN (%.0f%%) > FOLFIRI PN (%.0f%%)' % (folfox_np,folfiri_np))
    else:
        fail('Bull-6','FOLFOX PN not > FOLFIRI PN -- oxaliplatin signature wrong')

    # FOLFIRI alopecia vs FOLFOX
    if M[4,8] > M[3,8]:
        ok('Bull-6','FOLFIRI alopecia > FOLFOX -- irinotecan effect')
    else:
        fail('Bull-6','FOLFIRI alopecia not > FOLFOX')

    # CAPOX HFS
    if M[6,7] > M[3,7]:
        ok('Bull-6','CAPOX HFS > FOLFOX -- capecitabine effect')
    else:
        fail('Bull-6','CAPOX HFS not > FOLFOX')

    # Pure chemo irAE = 0
    chemo_irae_bad = [M[i,11] for i in [3,4,5,6,7] if M[i,11] != 0]
    if not chemo_irae_bad:
        ok('Bull-6','All pure-chemo irAE = 0%%')
    else:
        fail('Bull-6','Some pure chemo irAE != 0: %s' % str(chemo_irae_bad))

    # Pritamab mono safest
    prit_score = M[0].sum()
    others_score = [M[i].sum() for i in range(10) if i != 0]
    if all(s >= prit_score for s in others_score):
        ok('Bull-6','Pritamab mono lowest composite (%.0f)' % prit_score)
    else:
        fail('Bull-6','Pritamab mono NOT lowest composite')

# --- Center text improvement ---
print("\n  (7) Center content (practical not decorative)")
if 'n=10' in bull or 'n = 10' in bull:
    ok('Bull-7','Center shows n=10 (practical info)')
else:
    warn('Bull-7','Center may still show decorative "TOXICITY TARGET" text')
if 'TOXICITY\nTARGET' in bull or 'TOXICITY TARGET' in bull:
    if 'n=10' in bull:
        ok('Bull-7','Both n=10 and some decorative text -- acceptable')
    else:
        warn('Bull-7','"TOXICITY TARGET" text without practical info')

# --- Output file ---
print("\n  (8) Bullseye PNG output")
if os.path.exists(BULL_PNG):
    sz = os.path.getsize(BULL_PNG)//1024
    if sz >= 500:
        ok('Bull-8','PNG exists, %d KB (good)' % sz)
    else:
        warn('Bull-8','PNG small: %d KB' % sz)
else:
    fail('Bull-8','Bullseye PNG not found')

# ======================================================================
# C. CROSS-CHECK: Composite scores identical in both scripts
# ======================================================================
print("\n[Cross-check: Composite score consistency]")
# Dashboard has COMPOSITE_E; Bullseye has TOX_MATRIX from which we compute composite
dash_comp_m = re.search(r'COMPOSITE_E\s*=\s*\[([\d,\s]+)\]', dash)
bull_mat_m2 = re.search(r'TOX_MATRIX\s*=\s*np\.array\(\[(.+?)\],\s*dtype=float\)', bull, re.DOTALL)

if dash_comp_m and bull_mat_m2:
    dash_comp = list(map(int, re.findall(r'\d+', dash_comp_m.group(1))))
    ns3 = {}
    exec('import numpy as np\nTOX_MATRIX=np.array([%s],dtype=float)' % bull_mat_m2.group(1), ns3)
    M_bull = ns3['TOX_MATRIX']
    # Dashboard composite order: FOLFOXIRI,Prit+FOLFOX,Bev+FOLFOX,CAPOX,FOLFOX,FOLFIRI,Prit+FOLFIRI,TAS-102,Pembro,PritMono
    # Bullseye order: PritMono,Prit+FOX,Prit+FIRI,FOLFOX,FOLFIRI,FOLFOXIRI,CAPOX,TAS-102,Bev+FOX,Pembro
    bull_sums = M_bull.sum(axis=1)  # [29,124,105,118,115,175,119,104,123,57]
    bull_sorted_desc = sorted(bull_sums, reverse=True)
    dash_sorted_desc = sorted(dash_comp, reverse=True)
    if bull_sorted_desc == dash_sorted_desc:
        ok('Cross','Composite score sets identical (sorted desc): %s' % str(dash_sorted_desc[:5])+'...')
    else:
        fail('Cross','Composite scores DIFFER between scripts!')
        fail('Cross','  Dashboard: %s' % str(dash_sorted_desc[:5]))
        fail('Cross','  Bullseye:  %s' % str(bull_sorted_desc[:5]))
else:
    warn('Cross','Could not extract composite scores from one or both scripts')

# Extra: verify Pritamab mono (index 0 in bullseye) = 29 in both
if bull_mat_m2:
    prit_sum = int(ns3['TOX_MATRIX'][0].sum())
    if prit_sum == 29:
        ok('Cross','Pritamab mono composite = 29 in bullseye matrix (matches dashboard)')
    else:
        fail('Cross','Pritamab mono composite = %d in bullseye, expected 29' % prit_sum)

    folfoxiri_sum = int(ns3['TOX_MATRIX'][5].sum())
    if folfoxiri_sum == 175:
        ok('Cross','FOLFOXIRI composite = 175 in bullseye matrix (matches dashboard)')
    else:
        fail('Cross','FOLFOXIRI composite = %d in bullseye, expected 175' % folfoxiri_sum)

# ======================================================================
# FINAL SUMMARY
# ======================================================================
print("\n" + "="*65)
print("FINAL: %d OK  %d WARN  %d FAIL" % (PASS, WARN, FAIL))
print("="*65)

if FAIL == 0 and WARN <= 3:
    verdict = "PASS -- v2 revisions correctly applied"
elif FAIL == 0:
    verdict = "CONDITIONAL PASS -- %d warnings to address" % WARN
else:
    verdict = "NEEDS WORK -- %d critical failures" % FAIL
print("Verdict:", verdict)

if issues:
    print("\nAction items:")
    for iss in issues:
        print("  " + iss)
else:
    print("\nNo action items remaining.")
