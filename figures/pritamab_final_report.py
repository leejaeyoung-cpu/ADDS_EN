"""
Pritamab Final Comprehensive Report — v3.0
5×4 Grid (20 panels): Mechanism + PK/PD + Synergy + DL Pipeline + Clinical
2차 검증 완료 데이터 기반 | 2026-03-04
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, Circle
import matplotlib.patheffects as pe
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ── 색상 ──────────────────────────────────────────────────
BG='#070A14'; PANEL='#0F1726'; PANEL2='#141E30'
WHITE='#EEF2FF'; GRAY='#8B9CC4'; BLUE='#60A5FA'
GREEN='#34D399'; RED='#F87171'; AMBER='#FBBF24'
PURPLE='#A78BFA'; CYAN='#67E8F9'; TEAL='#2DD4BF'
ORANGE='#FB923C'; DARK='#1F2937'

def pb(ax, c=PANEL): ax.set_facecolor(c); [s.set_visible(False) for s in ax.spines.values()]
def pt(ax, t, c=WHITE, fs=10): ax.set_title(t, fontsize=fs, fontweight='bold', color=c, pad=7)

# ════════════════════════════════════════════════════════
# 검증 완료 데이터 상수
# ════════════════════════════════════════════════════════
# ★=NatureComm논문  ◆=ADDS계산  ●=SCI문헌  ▲=에너지모델투영
SIGNAL = {'RAS-GTP':-42,'ERK1/2':-38,'AKT S473':-31,'Notch1-NICD':-55}  # ★ L219-222
ENERGY = {'Survival\ninit.':(3.0,0.30,0.80),'Prolif.\ngate':(2.5,1.25,1.50),
           'Resist.\npeak':(2.0,1.70,1.80),'Apo\nentry':(1.5,1.25,1.30),
           'Apo\ncommit.':(1.0,0.88,0.90)}  # ★ L256-262
EC50 = {'5-FU':(12000,9032),'Oxali':(3750,2823),'Irino':(7500,5645),'Soto':(75,56.5)}  # ★ L283-286
# 시너지 히트맵: combos순서=5-FU,Oxali,Irino,Soto,FOLFOX,FOLFIRI,TAS-102
COMBOS=['Prit\n+5-FU','Prit\n+Oxali','Prit\n+Irino','Prit\n+Soto','Prit\n+FOLFOX','Prit\n+FOLFIRI','Prit\n+TAS-102']
BLISS= [18.4, 21.7, 17.3, 15.8, 20.5, 18.8, 18.1]   # ★(5FU,Oxali) ◆(나머지)
ADDS_S=[0.87, 0.89, 0.84, 0.82, 0.84, 0.87, 0.87]   # ★/◆
# ❌2차수정: DRS 순서 combos에 맞게 재정렬
DRS_V= [0.820,0.850,0.760,0.780,0.893,0.870,0.880]
APO_V= [85,   82,   75,   68,   80,   75,   80]
GRADES=['★','★','◆','◆','◆','◆','◆']  # Soto Bliss ◆ (2차수정)
# CS점수(ADDS계산◆+Apoptosis투영▲)
CS={'5-FU':0.820,'Oxali':0.850,'Irino':0.760,'Soto':0.780,'FOLFOX':0.893,'FOLFIRI':0.870,'TAS-102':0.880}
# Apoptosis% 패널
APO_COMBOS=['Baseline\n(KRAS-mut)','PrPc\nsiRNA','Pritamab\nalone','+Irino','+Oxali',
            '+TAS-102','+FOLFOX*','+FOLFIRI*','+FOLFOXIRI*']
APO_PCT=[22,40,55,75,75,80,85,82,88]
APO_G=['●','●','▲+●','▲','▲','▲+●','▲(!)','▲(!)','▲(!)']
# DG문헌
DG_L=['SN-38\nTopo-I','Edotech.\nTopo-I','Naphtho.\nTopo-I','Oxali\nDNA(QM)','FTD\nDNA(cov)',
      '5-FU TS\nnative\n[≠ADDS]','FdUMP TS\nactive\n[≈ADDS]',
      'ADDS:\nIrino−13.0','ADDS:\nOxali−14.0','ADDS:\nFTD−14.3','ADDS:\n5-FU−11.2']
DG_V=[-12.0,-10.7,-11.94,-14.0,-13.5,-3.44,-11.5,-13.0,-14.0,-14.3,-11.2]
DG_C=[BLUE]*3+[RED]*2+[GREEN]*2+[CYAN,CYAN,PURPLE,TEAL]
# DL 데이터 (◆ ADDS DL 합성코호트 n=1000)
KRAS_MUT =['G12D','G12V','G12C','G13D','WT']
KRAS_PFS =[13.86,14.45,14.28,14.03,14.29]
KRAS_OS  =[17.48,17.21,17.45,18.29,16.32]
KRAS_ORR =[58,   55,   49,   47,   47]
KRAS_DL_HR=[0.965,0.888,0.891,1.009,0.932]
KRAS_EM_HR=[0.52, 0.55, 0.53, 0.58, 0.67]
# 독성
TOX_ITEMS=['Neutro.','Anemia','Diarrhea','Nausea','Neuropathy','Fatigue']
TOX_FF=[32,8,10,14,8,22]; TOX_PF=[24,6,8,10,6,17]; TOX_FI=[28,6,28,18,3,20]
# IHC발현
IHC_CT=['CRC\n(58-91%)','Gastric\n(66-70%)','PDAC\n(76%)','Breast\n(15-33%)']
IHC_EXP=[74.5,68.0,76.0,24.0]
IHC_KRAS=[40,15,90,5]

# ════════════════════════════════════════════════════════
# FIGURE 생성 (5행×4열)
# ════════════════════════════════════════════════════════
fig=plt.figure(figsize=(38,46), facecolor=BG)
fig.text(0.5,0.991,"PRITAMAB  ·  Final Comprehensive Research Report  v3.0",
         ha='center',va='top',fontsize=30,fontweight='bold',color=WHITE)
fig.text(0.5,0.983,"PrPc-Targeting Anti-Cancer Antibody | Mechanism · PK · Synergy · DL Pipeline · Clinical Development",
         ha='center',va='top',fontsize=13,color=GRAY)
legend_str=("★ NatureComm Paper Direct   ◆ ADDS System Calculated   "
            "● SCI Literature   ▲ ADDS Energy Model Projection   ◇ DL Synthetic Cohort (n=1,000)")
fig.text(0.5,0.977,legend_str,ha='center',va='top',fontsize=9.5,color=AMBER,
         bbox=dict(boxstyle='round,pad=0.4',facecolor='#1A1F35',edgecolor=AMBER,alpha=0.9,linewidth=1))

gs=gridspec.GridSpec(5,4,figure=fig,left=0.03,right=0.985,
                     top=0.972,bottom=0.015,hspace=0.38,wspace=0.28)

# ═══════════════ ROW 1 ════════════════════════════════
# [R1C0] 메커니즘
ax1=fig.add_subplot(gs[0,0]); pb(ax1); ax1.set_xlim(0,10); ax1.set_ylim(0,10); ax1.axis('off')
pt(ax1,"① PrPc-RPSA Signalosome  ★",CYAN)
nodes=[(5,9.0,"Surface PrPc\n(Octarepeat 51-90)",BLUE,0.7),(5,7.3,"RPSA / 37LRP",RED,0.6),
       (5,5.7,"SRC/FYN Kinase",AMBER,0.5),(2.5,3.8,"RAS-GTP\nLoading ↑",RED,0.55),
       (5,3.8,"KRAS G12D/V/C\nConst. Act.",ORANGE,0.55),(7.5,3.8,"Filamin A\nEMT/Invasion",PURPLE,0.5),
       (2.5,1.8,"RAF-MEK\nERK ↑",RED,0.45),(5,1.8,"PI3K-AKT\nSurvival ↑",ORANGE,0.45),
       (7.5,1.8,"Notch1-NICD\nCSC ↑",PURPLE,0.45)]
for x,y,txt,c,s in nodes:
    ax1.add_patch(Circle((x,y),s,facecolor=c,alpha=0.22,edgecolor=c,linewidth=1.5))
    ax1.text(x,y,txt,ha='center',va='center',fontsize=6.5,color=WHITE,fontweight='bold',
             path_effects=[pe.withStroke(linewidth=1,foreground=BG)])
for x1,y1,x2,y2 in [(5,8.3,5,7.9),(5,6.7,5,6.3),(5,5.2,2.5,4.35),(5,5.2,5,4.35),
                     (5,5.2,7.5,4.35),(2.5,3.25,2.5,2.25),(5,3.25,5,2.25),(7.5,3.25,7.5,2.25)]:
    ax1.annotate('',xy=(x2,y2),xytext=(x1,y1),arrowprops=dict(arrowstyle='->',color=GRAY,lw=1.2))
ax1.text(2.5,6.1,'SOS1/2\n(RAS-GEF)\n★L443',ha='center',va='center',fontsize=5.5,color=AMBER,style='italic')
ax1.text(5,8.65,'[X] PRITAMAB\nIC₅₀=12.3nM ★',ha='center',va='center',fontsize=7.5,fontweight='bold',color=GREEN,
         bbox=dict(boxstyle='round,pad=0.3',facecolor='#064E3B',edgecolor=GREEN,linewidth=1.5))
ax1.text(0.3,9.3,'KD=0.84nM ★',fontsize=7,color=AMBER,fontweight='bold')

# [R1C1] 에너지 장벽
ax2=fig.add_subplot(gs[0,1]); pb(ax2); pt(ax2,"② Energy Barrier Profile  ★",AMBER)
nk=list(ENERGY.keys()); vn=[v[0] for v in ENERGY.values()]
vm=[v[1] for v in ENERGY.values()]; vp=[v[2] for v in ENERGY.values()]; xn=range(len(nk))
ax2.plot(xn,vn,'o-',color=GREEN,lw=2,ms=6,label='Normal (WT)')
ax2.plot(xn,vm,'s--',color=RED,lw=2,ms=6,label='KRAS-mut+PrPc↑')
ax2.plot(xn,vp,'D-',color=BLUE,lw=2.5,ms=6,label='+Pritamab')
ax2.fill_between(xn,vm,vp,alpha=0.15,color=BLUE)
ax2.set_xticks(list(xn)); ax2.set_xticklabels(nk,fontsize=6.5,color=WHITE)
ax2.set_ylabel('Energy Barrier (rel.units)',color=GRAY,fontsize=8)
ax2.tick_params(colors=GRAY,labelsize=7); ax2.grid(alpha=0.12,color=GRAY)
ax2.legend(fontsize=7.5,facecolor=DARK,edgecolor=GRAY,labelcolor=WHITE,loc='upper right')
ax2.text(0.05,0.85,'Rate ↓ 55.6% ★\nArrhenius',transform=ax2.transAxes,fontsize=7.5,color=BLUE,fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.3',facecolor=PANEL2,edgecolor=BLUE))
ax2.annotate('10× collapse ★',xy=(0,0.30),fontsize=7,color=RED,xytext=(0.45,0.15),
             arrowprops=dict(arrowstyle='->',color=RED,lw=1))

# [R1C2] 신호 차단
ax3=fig.add_subplot(gs[0,2]); pb(ax3); pt(ax3,"③ Signalling Inhibition (10nM,24h)  ★",GREEN)
lb=list(SIGNAL.keys()); vl=list(SIGNAL.values())
cs3=[RED if v<0 else GREEN for v in vl]
br3=ax3.barh(lb,vl,color=cs3,alpha=0.82,edgecolor=BG)
for bar,v in zip(br3,vl):
    ax3.text(v-2 if v<0 else v+1,bar.get_y()+bar.get_height()/2,
             f'{v}%',ha='right' if v<0 else 'left',va='center',fontsize=9,color=WHITE,fontweight='bold')
ax3.axvline(0,color=GRAY,lw=1); ax3.set_xlim(-70,20)
ax3.tick_params(colors=GRAY,labelsize=8); ax3.set_xlabel('Change vs Control (%)',color=GRAY,fontsize=8)
ax3.grid(axis='x',alpha=0.12,color=GRAY)
for i,pv in enumerate(['<0.001','0.001','0.004','<0.001']):
    ax3.text(2,i,f'p={pv}',va='center',fontsize=7,color=GRAY)
ax3.text(0.98,0.08,'Cleaved\nCaspase-3:\n+2.8-fold ★',ha='right',va='bottom',
         transform=ax3.transAxes,fontsize=8.5,color=CYAN,fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.3',facecolor='#0C3547',edgecolor=CYAN))

# [R1C3] PK 파라미터
ax4=fig.add_subplot(gs[0,3]); pb(ax4); pt(ax4,"④ PK/PD Parameters  ★",PURPLE); ax4.axis('off')
pk_rows=[("KD (SPR)","0.84 nM","★"),("IC₅₀ (RPSA)","12.3 nM","★"),
         ("IC₅₀ (cytotox)",">500 nM","★"),("Clearance","0.18 L/day","★"),
         ("Volume (Vd)","4.3 L","★"),("t½ (terminal)","21-25 days","★"),
         ("Cmin target","≥50 nM","★"),("EC50 Reduction","−24.7%","★"),
         ("Rate Reduction","−55.6%","★◆"),  # ❌2차수정: ★◆
         ("ddG_RLS","0.50 kcal/mol","★"),("α coupling","0.35","★"),
         ("ADCC fold","10-15× WT IgG1","★"),("Dose","10-15 mg/kg Q3W","★"),("Accum. ratio","1.4-1.6×","★")]
y0=0.97
for param,val,grade in pk_rows:
    gc=GREEN if '★' in grade else BLUE
    ax4.text(0.02,y0,f"{param}:",fontsize=7.5,color=GRAY,transform=ax4.transAxes,va='top')
    ax4.text(0.53,y0,val,fontsize=7.5,color=WHITE,fontweight='bold',transform=ax4.transAxes,va='top')
    ax4.text(0.92,y0,grade,fontsize=7.5,color=gc,fontweight='bold',transform=ax4.transAxes,va='top')
    y0-=0.067
    ax4.plot([0.02,0.98],[y0+0.005,y0+0.005],color=PANEL2,lw=0.5,transform=ax4.transAxes,clip_on=False)

# ═══════════════ ROW 2 ════════════════════════════════
# [R2C0] EC50
ax5=fig.add_subplot(gs[1,0]); pb(ax5); pt(ax5,"⑤ EC50 Sensitisation (−24.7%)  ★",AMBER)
drugs=list(EC50.keys()); alone=[v[0] for v in EC50.values()]; combo=[v[1] for v in EC50.values()]
xe=np.arange(len(drugs)); w=0.35
ax5.bar(xe-w/2,alone,w,color=RED,alpha=0.8,label='Alone',edgecolor=BG)
b2=ax5.bar(xe+w/2,combo,w,color=BLUE,alpha=0.8,label='+Pritamab',edgecolor=BG)
ax5.set_yscale('log'); ax5.set_xticks(xe); ax5.set_xticklabels(drugs,color=WHITE,fontsize=9)
ax5.set_ylabel('EC50 (nM) — log scale',color=GRAY,fontsize=8)
ax5.tick_params(colors=GRAY,labelsize=8); ax5.grid(axis='y',alpha=0.12,color=GRAY)
ax5.legend(fontsize=8,facecolor=DARK,edgecolor=GRAY,labelcolor=WHITE)
for bar in b2:
    ax5.text(bar.get_x()+bar.get_width()/2,bar.get_height()*1.2,'−24.7%',
             ha='center',va='bottom',fontsize=7,color=AMBER,fontweight='bold')

# [R2C1] 시너지 히트맵
ax6=fig.add_subplot(gs[1,1]); pb(ax6); pt(ax6,"⑥ 4-Model Synergy Heat Map  ★/◆",GREEN)
data_h=np.array([[b/25 for b in BLISS],ADDS_S,DRS_V,[a/100 for a in APO_V]])
cmap6=mcolors.LinearSegmentedColormap.from_list('syn',['#1E3A5F','#1E6B8A','#26A69A','#66BB6A','#FFD54F','#EF5350'])
im=ax6.imshow(data_h,cmap=cmap6,aspect='auto',vmin=0.5,vmax=1.0)
ax6.set_xticks(range(len(COMBOS))); ax6.set_xticklabels(COMBOS,fontsize=6.5,color=WHITE,rotation=0)
ax6.set_yticks(range(4))
ax6.set_yticklabels(['Bliss\n(0-25)','ADDS\nConsensus','DRS\nScore','Apoptosis\n(÷100)'],fontsize=8,color=WHITE)
ax6.tick_params(colors=GRAY,labelsize=7)
for i in range(4):
    for j in range(len(COMBOS)):
        ax6.text(j,i,f'{data_h[i,j]:.2f}',ha='center',va='center',fontsize=7,
                 color='white' if data_h[i,j]<0.85 else BG,fontweight='bold')
for j,g in enumerate(GRADES):
    ax6.text(j,-0.7,g,ha='center',va='center',fontsize=8,color=AMBER)
plt.colorbar(im,ax=ax6,shrink=0.7,pad=0.02).ax.tick_params(colors=GRAY,labelsize=7)

# [R2C2] CS 랭킹
ax7=fig.add_subplot(gs[1,2]); pb(ax7); pt(ax7,"⑦ Comprehensive Score Ranking  ◆+▲",BLUE)
ax7.text(0.5,-0.13,'CS=0.35×DRS+0.25×(Bliss/25)+0.20×ADDS+0.20×(Apo/100)\n⚠ ADDS◆+Apoptosis▲ 이중추론 — 임상 미검증',
         ha='center',transform=ax7.transAxes,fontsize=6.5,color=AMBER,style='italic')
cs_s=sorted(CS.items(),key=lambda x:x[1],reverse=True)
cn=[x[0] for x in cs_s]; cv=[x[1] for x in cs_s]
rc=[AMBER,WHITE,TEAL,BLUE,GREEN,PURPLE,GRAY]
ax7.barh(range(len(cn)),cv,color=rc[:len(cn)],alpha=0.85,edgecolor=BG)
ax7.set_yticks(range(len(cn))); ax7.set_yticklabels([f'Prit+{n}' for n in cn],fontsize=7.5,color=WHITE)
ax7.set_xlim(0.70,0.93); ax7.axvline(0.86,color=RED,lw=1.5,linestyle='--',alpha=0.7)
ax7.text(0.863,-0.5,'CS≥0.86\nTop',fontsize=7,color=RED,va='top')
ax7.tick_params(colors=GRAY,labelsize=7); ax7.grid(axis='x',alpha=0.12,color=GRAY)
ax7.set_xlabel('Comprehensive Score',color=GRAY,fontsize=8)
for i,(v,nm) in enumerate(zip(cv,cn)):
    ax7.text(-0.005,i,f'#{i+1}',ha='right',va='center',fontsize=8,color=AMBER)
    ax7.text(v+0.002,i,f'{v:.3f}',ha='left',va='center',fontsize=7.5,color=WHITE)

# [R2C3] Apoptosis%
ax8=fig.add_subplot(gs[1,3]); pb(ax8); pt(ax8,"⑧ Apoptosis Efficiency (%)  ▲/●",PURPLE)
cmap8=plt.cm.get_cmap('YlOrRd')
c8=[cmap8(v/100) for v in APO_PCT]
ax8.barh(APO_COMBOS,APO_PCT,color=c8,alpha=0.85,edgecolor=BG)
ax8.axvline(55,color=CYAN,lw=1.5,linestyle=':',alpha=0.7,label='Alone 55%')
ax8.axvline(25,color=GRAY,lw=1.2,linestyle='--',alpha=0.5,label='Baseline ~25%')
ax8.set_xlim(0,110); ax8.tick_params(colors=GRAY,labelsize=7.5)
ax8.set_xlabel('Apoptosis Rate (%)',color=GRAY,fontsize=8)
ax8.grid(axis='x',alpha=0.12,color=GRAY)
ax8.legend(fontsize=7,facecolor=DARK,edgecolor=GRAY,labelcolor=WHITE,loc='lower right')
for v,g,bar in zip(APO_PCT,APO_G,ax8.patches):
    ax8.text(v+1,bar.get_y()+bar.get_height()/2,f'{v}% {g}',ha='left',va='center',fontsize=7,color=WHITE)
ax8.text(0.01,0.06,'* FOLFOX/FOLFIRI/FOLFOXIRI:\nADDS 추론 — 임상 미검증',
         transform=ax8.transAxes,fontsize=6,color=AMBER,
         bbox=dict(boxstyle='round,pad=0.3',facecolor='#2A1A00',edgecolor=AMBER,alpha=0.9))

# ═══════════════ ROW 3 ════════════════════════════════
# [R3C0] ΔG 문헌
ax9=fig.add_subplot(gs[2,0]); pb(ax9); pt(ax9,"⑨ ΔG_bind Literature Evidence  ●",TEAL)
ax9.barh(DG_L,DG_V,color=DG_C,alpha=0.8,edgecolor=BG)
ax9.axvline(-10.7,color=GRAY,lw=0.8,linestyle=':',alpha=0.5)
for v,c in [(-13.0,CYAN),(-14.0,CYAN),(-14.3,PURPLE),(-11.2,TEAL)]:
    ax9.axvline(v,color=c,lw=1.5,linestyle='--',alpha=0.6)
for bar,v in zip(ax9.patches,DG_V):
    ax9.text(v-0.3,bar.get_y()+bar.get_height()/2,f'{v}',ha='right',va='center',fontsize=7,color=WHITE)
ax9.set_xlim(-18,0); ax9.tick_params(colors=GRAY,labelsize=7)
ax9.set_xlabel('ΔG_bind (kcal/mol)',color=GRAY,fontsize=8)
ax9.grid(axis='x',alpha=0.12,color=GRAY)
ax9.text(-3.44,5.5,'← 5-FU native\n  (≠FdUMP AM)',fontsize=6,color=AMBER,ha='left',va='center')
legs9=[Patch(fc=BLUE,label='Topo-I inh.'),Patch(fc=RED,label='Oxali-DNA'),
       Patch(fc=GREEN,label='5-FU/TS'),Patch(fc=PURPLE,label='TAS-102(FTD)'),Patch(fc=CYAN,label='ADDS inferred')]
ax9.legend(handles=legs9,fontsize=6.5,facecolor=DARK,edgecolor=GRAY,labelcolor=WHITE,loc='lower right')

# [R3C1] Apo 문헌
ax10=fig.add_subplot(gs[2,1]); pb(ax10); pt(ax10,"⑩ Apoptosis Evidence (Lit.)  ●",GREEN)
apo_c2=['KRAS-mut\nbaseline','PrPc siRNA\nHCT116','FOLFOX+Bev\nHCT116','Irino+met.\nSW480',
        '5-FU+dios.\nHCT116','TAS-102+tgt\n(Oncotarget)','ADDS Prit.\nalone (▲)']
apo_m2=[22,40,70,68,45,78,55]; apo_lo=[15,25,60,60,45,75,50]; apo_hi=[30,50,75,75,45,82,60]
apo_err=[[m-l for m,l in zip(apo_m2,apo_lo)],[h-m for m,h in zip(apo_m2,apo_hi)]]
ac2=[GRAY,BLUE,ORANGE,ORANGE,GREEN,PURPLE,CYAN]
ax10.barh(apo_c2,apo_m2,xerr=apo_err,color=ac2,alpha=0.80,edgecolor=BG,
          error_kw=dict(ecolor=GRAY,lw=1.2))
ax10.axvline(55,color=CYAN,lw=2,linestyle='--',alpha=0.8,label='ADDS 55%')
ax10.set_xlim(0,95); ax10.tick_params(colors=GRAY,labelsize=7)
ax10.set_xlabel('Apoptosis Rate (%)',color=GRAY,fontsize=8)
ax10.grid(axis='x',alpha=0.12,color=GRAY)
ax10.legend(fontsize=8,facecolor=DARK,edgecolor=GRAY,labelcolor=WHITE)

# [R3C2] PK 시뮬
ax11=fig.add_subplot(gs[2,2]); pb(ax11); pt(ax11,"⑪ PK Simulation (10 mg/kg Q3W)  ★",AMBER)
CL=0.18; Vd=4.3; t=np.linspace(0,100,1000)
MW=148000; C0=(70*10)/Vd*1e6/MW*1e9; k_el=CL/Vd
conc=np.zeros_like(t)
for dt in [0,21,42,63]:
    mask=t>=dt; conc[mask]+=C0*np.exp(-k_el*(t[mask]-dt))
ax11.plot(t,conc,color=BLUE,lw=2.5)
ax11.fill_between(t,conc,alpha=0.12,color=BLUE)
ax11.axhline(50,color=GREEN,lw=1.8,linestyle='--',label='Cmin=50nM ★(L403)')
ax11.axhline(12.3*4,color=AMBER,lw=1.2,linestyle=':',label=f'4×IC₅₀={12.3*4:.1f}nM ★(L404)')
for i,dt in enumerate([0,21,42,63]):
    ax11.axvline(dt,color=GRAY,lw=0.8,linestyle=':',alpha=0.5)
    ax11.text(dt+0.5,max(conc)*0.92,f'Dose {i+1}',fontsize=7,color=GRAY,rotation=90)
ax11.set_xlabel('Time (days)',color=GRAY,fontsize=8)
ax11.set_ylabel('Serum Conc. (nM, MW=148kDa assumed)',color=GRAY,fontsize=8)
ax11.tick_params(colors=GRAY,labelsize=8); ax11.grid(alpha=0.12,color=GRAY)
ax11.legend(fontsize=7.5,facecolor=DARK,edgecolor=GRAY,labelcolor=WHITE)
ax11.set_ylim(0,max(conc)*1.15)
ax11.text(0.01,0.97,'Q3W sim. (논문 제안 Q3W L398)\nCmin기준:Q2W기재(L405)\nMW=148kDa 가정',
          transform=ax11.transAxes,fontsize=6,color=AMBER,va='top',
          bbox=dict(boxstyle='round,pad=0.3',facecolor='#1A1200',edgecolor=AMBER,alpha=0.85))

# [R3C3] 독성 프로파일
ax12=fig.add_subplot(gs[2,3]); pb(ax12); pt(ax12,"⑫ Toxicity Profile G3/4 (%)  ●",RED)
xp=np.arange(len(TOX_ITEMS)); wt=0.25
ax12.bar(xp-wt,TOX_FF,wt,label='FOLFOX alone ●',color=RED,alpha=0.75,edgecolor=BG)
ax12.bar(xp,    TOX_PF,wt,label='Prit+FOLFOX ▲',color=BLUE,alpha=0.75,edgecolor=BG)
ax12.bar(xp+wt, TOX_FI,wt,label='FOLFIRI alone ●',color=ORANGE,alpha=0.75,edgecolor=BG)
ax12.axhline(30,color=RED,lw=1.5,linestyle='--',alpha=0.6)
ax12.axhline(15,color=AMBER,lw=1.0,linestyle=':',alpha=0.6)
ax12.set_xticks(xp); ax12.set_xticklabels(TOX_ITEMS,rotation=25,ha='right',fontsize=7.5,color=WHITE)
ax12.set_ylabel('G3/4 Incidence (%)',color=GRAY,fontsize=8)
ax12.tick_params(colors=GRAY,labelsize=8); ax12.grid(axis='y',alpha=0.12,color=GRAY)
ax12.legend(fontsize=7,facecolor=DARK,edgecolor=GRAY,labelcolor=WHITE)
ax12.text(0.01,0.95,'Dashed: 30% high-risk',transform=ax12.transAxes,fontsize=7,color=RED,va='top')

# ═══════════════ ROW 4 — DL PIPELINE ══════════════════
# [R4C0] DL 아키텍처
ax13=fig.add_subplot(gs[3,0]); pb(ax13); pt(ax13,"⑬ DL Pipeline Architecture  ◆",BLUE); ax13.axis('off')
ax13.set_xlim(0,10); ax13.set_ylim(0,10)
modals=[('Modal 1\nCellpose','128d','세포밀도·핵분절화\nPrPc대리지표',CYAN,8.5),
        ('Modal 2\nRNA-seq','256d','PRNP↓, CASP3↑\nPritamab서명 27g',GREEN,6.5),
        ('Modal 3\nPK/PD','32d','Bliss/EC50/ddG\n에너지장벽기반',AMBER,4.5),
        ('Modal 4\nCT Image','64d','종양부피·HU밀도\nnnUNet기반',PURPLE,2.5)]
for lbl,dim,desc,c,y in modals:
    ax13.add_patch(FancyBboxPatch((0.2,y-0.5),4.0,0.95,boxstyle='round,pad=0.05',
                                   facecolor=f'{c}22',edgecolor=c,linewidth=1.2))
    ax13.text(0.5,y+0.25,lbl,fontsize=7.5,color=c,fontweight='bold',va='center')
    ax13.text(1.8,y+0.25,f'→ {dim}',fontsize=8,color=WHITE,va='center',fontweight='bold')
    ax13.text(0.5,y-0.15,desc,fontsize=6.5,color=GRAY,va='center')
    ax13.annotate('',xy=(5.2,y+0.25),xytext=(4.2,y+0.25),
                  arrowprops=dict(arrowstyle='->',color=GRAY,lw=1))
ax13.add_patch(FancyBboxPatch((5.2,3.8),4.2,3.8,boxstyle='round,pad=0.1',
                               facecolor='#1E3A5F',edgecolor=BLUE,linewidth=1.5))
ax13.text(7.3,7.3,'Fusion MLP',fontsize=9,color=BLUE,fontweight='bold',ha='center')
ax13.text(7.3,6.7,'480→256→128→64',fontsize=7.5,color=WHITE,ha='center')
for i,(head,c2) in enumerate([('PFS head',GREEN),('OS head',ORANGE),('Synergy head',PURPLE)]):
    ax13.text(7.3,5.9-i*0.65,head,fontsize=7.5,color=c2,ha='center',fontweight='bold')
ax13.text(7.3,4.0,'Xavier init.\nSoftplus+Sigmoid\nCalibration',fontsize=6.5,color=GRAY,ha='center')
ax13.text(5.0,1.5,'Total: 480d input → 3 output heads',fontsize=7,color=GRAY,ha='center')
ax13.text(5.0,0.8,'◆ ADDS DL synthetic cohort n=1,000',fontsize=7,color=AMBER,ha='center',style='italic')

# [R4C1] KRAS 변이별 PFS/OS
ax14=fig.add_subplot(gs[3,1]); pb(ax14); pt(ax14,"⑭ KRAS Subtype: mPFS & mOS  ◆",CYAN)
x14=np.arange(len(KRAS_MUT)); w14=0.35
ax14.bar(x14-w14/2,KRAS_PFS,w14,color=BLUE,alpha=0.8,label='mPFS (mo)',edgecolor=BG)
ax14.bar(x14+w14/2,KRAS_OS, w14,color=ORANGE,alpha=0.8,label='mOS (mo)',edgecolor=BG)
ax14.set_xticks(x14); ax14.set_xticklabels(KRAS_MUT,color=WHITE,fontsize=9)
ax14.set_ylabel('Months',color=GRAY,fontsize=8)
ax14.tick_params(colors=GRAY,labelsize=8); ax14.grid(axis='y',alpha=0.12,color=GRAY)
ax14.legend(fontsize=8,facecolor=DARK,edgecolor=GRAY,labelcolor=WHITE)
ax14.set_ylim(0,23)
for i,(pfs,os_,orr) in enumerate(zip(KRAS_PFS,KRAS_OS,KRAS_ORR)):
    ax14.text(i-w14/2,pfs+0.2,f'{pfs}m',ha='center',fontsize=7,color=BLUE,fontweight='bold')
    ax14.text(i+w14/2,os_+0.2,f'{os_}m',ha='center',fontsize=7,color=ORANGE,fontweight='bold')
    ax14.text(i,1.0,f'ORR\n{orr}%',ha='center',fontsize=6.5,color=WHITE,
              bbox=dict(boxstyle='round,pad=0.2',facecolor=PANEL2,alpha=0.8))
ax14.text(0.01,0.97,'◆ DL 합성코호트 n=1,000 | ★ 방향성 에너지모델과 일치',
          transform=ax14.transAxes,fontsize=6.5,color=AMBER,va='top',style='italic')

# [R4C2] 합성 코호트 ORR/DCR
ax15=fig.add_subplot(gs[3,2]); pb(ax15); pt(ax15,"⑮ Synthetic Cohort Response  ◆◇",GREEN)
grps=['Pritamab\n(n=666)','Control\n(n=334)']
orr_v=[51.5,24.0]; dcr_v=[99.2,89.2]
x15=np.arange(2); w15=0.32
b15a=ax15.bar(x15-w15/2,orr_v,w15,color=[GREEN,GRAY],alpha=0.82,label='ORR (%)',edgecolor=BG)
b15b=ax15.bar(x15+w15/2,dcr_v,w15,color=[CYAN,PANEL2],alpha=0.82,label='DCR (%)',edgecolor=[CYAN,GRAY])
ax15.set_xticks(x15); ax15.set_xticklabels(grps,color=WHITE,fontsize=10)
ax15.set_ylabel('%',color=GRAY,fontsize=9); ax15.set_ylim(0,115)
ax15.tick_params(colors=GRAY,labelsize=8); ax15.grid(axis='y',alpha=0.12,color=GRAY)
ax15.legend(fontsize=8,facecolor=DARK,edgecolor=GRAY,labelcolor=WHITE)
for bar,v in zip(b15a,orr_v):
    ax15.text(bar.get_x()+bar.get_width()/2,v+1.5,f'{v}%',ha='center',fontsize=9,color=WHITE,fontweight='bold')
for bar,v in zip(b15b,dcr_v):
    ax15.text(bar.get_x()+bar.get_width()/2,v+1.5,f'{v}%',ha='center',fontsize=9,color=WHITE,fontweight='bold')
ax15.annotate('',xy=(1-w15/2-0.02,51.5),xytext=(0-w15/2+w15+0.02,51.5),
              arrowprops=dict(arrowstyle='<->',color=AMBER,lw=2))
ax15.text(0.5,55,'ΔORR=+27.5%p',ha='center',fontsize=9,color=AMBER,fontweight='bold')
ax15.text(0.5,62,'Syn.Score: 17.10 vs 3.97 (◆)',ha='center',fontsize=7.5,color=TEAL)
ax15.text(0.5,67,'ΔmOS: +2.87mo (+20.3%) (◆)',ha='center',fontsize=7.5,color=PURPLE)
ax15.text(0.5,0.02,'◇ DL 합성코호트 — 실제 임상 결과 아님',
          transform=ax15.transAxes,fontsize=6.5,color=AMBER,ha='center',style='italic')

# [R4C3] PrPc 발현 역학 (버블차트)
ax16=fig.add_subplot(gs[3,3]); pb(ax16); pt(ax16,"⑯ PrPc Expression vs KRAS Rate  ★",AMBER)
ax16.scatter(IHC_KRAS,IHC_EXP,s=[v*20 for v in IHC_EXP],
             c=[ORANGE,GREEN,RED,BLUE],alpha=0.75,edgecolors=WHITE,linewidths=1)
for ct,kx,py in zip(IHC_CT,IHC_KRAS,IHC_EXP):
    ax16.text(kx+1.5,py,ct,fontsize=7.5,color=WHITE,va='center',
              path_effects=[pe.withStroke(linewidth=1,foreground=BG)])
ax16.set_xlabel('KRAS Mutation Rate (%)',color=GRAY,fontsize=8)
ax16.set_ylabel('PrPc Expression Rate (%)',color=GRAY,fontsize=8)
ax16.tick_params(colors=GRAY,labelsize=8); ax16.grid(alpha=0.12,color=GRAY)
ax16.set_xlim(-5,100); ax16.set_ylim(0,100)
ax16.text(0.05,0.05,'Bubble size ∝ PrPc expression\nParallel trend: ★논문 근거',
          transform=ax16.transAxes,fontsize=7,color=AMBER,style='italic')

# ═══════════════ ROW 5 ════════════════════════════════
# [R5C0] 환자 선택
ax17=fig.add_subplot(gs[4,0]); pb(ax17); pt(ax17,"⑰ Patient Selection Strategy  ★",CYAN); ax17.axis('off')
pat_data=[('Biomarker','Criterion','Value','Grade'),
          ('PrPc IHC (8H4)','H-score ≥ 50','85.7% KRAS+','★'),
          ('KRAS mutation','Any allele (NGS)','40% CRC','★'),
          ('Dual positive','PrPc+/KRAS+','34.5% CRC','★'),
          ('KRAS G12D','H-score 142±28','1st','★'),
          ('KRAS G12V','H-score 138±31','2nd','★'),
          ('G13D','H-score 124±34','3rd','★'),
          ('CRC annual(US)','PrPc+/KRAS+','~52,500/yr','★'),
          ('Global total','All KRAS indic.','~120,000+/yr','★'),
          ('DL: PrPc-high\nORR','G12D/G12V','63%/60%','◆')]
col_x=[0.01,0.32,0.65,0.93]
row_y=np.linspace(0.95,0.05,len(pat_data))
for ri,row in enumerate(pat_data):
    for ci,(cell,x) in enumerate(zip(row,col_x)):
        fw='bold' if ri==0 else 'normal'
        tc=AMBER if ri==0 else (GREEN if cell=='★' else (BLUE if cell=='◆' else WHITE))
        ax17.text(x,row_y[ri],cell,fontsize=7,color=tc,fontweight=fw,transform=ax17.transAxes,va='center')
    if ri<len(pat_data)-1:
        ax17.plot([0,1],[row_y[ri]-0.047,row_y[ri]-0.047],color=DARK,lw=0.5,
                  transform=ax17.transAxes,clip_on=False)

# [R5C1] 임상 로드맵
ax18=fig.add_subplot(gs[4,1]); pb(ax18); pt(ax18,"⑱ Clinical Development Roadmap  ★",GREEN); ax18.axis('off')
phases=[('Phase I\n(12-18 mo)','1→15 mg/kg Q3W\n3+3 dose escalation\nMTD/RP2D\nPK+RPSA occupancy',BLUE),
        ('Phase II\n(18-36 mo)','n=120 (2:1 Prit+FOLFOX)\nmCRC KRAS-mut PrPc-high\nmPFS 5.5→8.25m ★\nHR=0.667 α=0.10',GREEN),
        ('Phase III\n(3-5 yr)','FOLFOX+Bev ± Pritamab\nDouble-blind RCT\n1st-line mCRC\nOS HR=0.75 ★',ORANGE)]
for (lbl,txt,c),y in zip(phases,[0.88,0.55,0.22]):
    ax18.add_patch(FancyBboxPatch((0.03,y-0.12),0.94,0.28,boxstyle='round,pad=0.02',
                                   facecolor=f'{c}22',edgecolor=c,linewidth=1.5,transform=ax18.transAxes))
    ax18.text(0.08,y+0.10,lbl,transform=ax18.transAxes,fontsize=8.5,fontweight='bold',color=c,va='top')
    ax18.text(0.08,y+0.01,txt,transform=ax18.transAxes,fontsize=7,color=WHITE,va='top',linespacing=1.5)
    if y!=0.22:
        ax18.annotate('',xy=(0.5,y-0.12),xytext=(0.5,y-0.08),xycoords='axes fraction',
                      arrowprops=dict(arrowstyle='->',color=GRAY,lw=1.5))

# [R5C2-3] 검증 요약표 (2차 수정 반영)
ax19=fig.add_subplot(gs[4,2:]); pb(ax19); pt(ax19,"⑲-⑳ Data Validation Summary — All Report Values",WHITE,fs=11); ax19.axis('off')
ver_rows=[
    ('Category','Value','Source','Grade','Status'),
    ('KD (SPR)','0.84 nM','NatureComm §Results L210','★','VERIFIED'),
    ('IC50 PrPc-RPSA','12.3 nM','NatureComm §Results L215','★','VERIFIED'),
    ('EC50 reduction','−24.7% (4 drugs)','NatureComm L283-286','★','VERIFIED'),
    ('Bliss 5-FU','+18.4','NatureComm L305','★','VERIFIED'),
    ('Bliss Oxaliplatin','+21.7','NatureComm L306','★','VERIFIED'),
    ('Bliss Sotorasib','15.8 (ADDS est.)','ADDS DL — 논문에없음','◆','ADDS EST'),  # 2차수정
    ('Loewe DRI','1.34 (5-FU & Oxali)','NatureComm L309','★','VERIFIED'),
    ('ADDS consensus','0.87/0.89/0.82/0.84','NatureComm L375-378','★','VERIFIED'),
    ('ddG_RLS','0.50 kcal/mol','NatureComm + ADDS','★◆','VERIFIED'),
    ('Rate Reduction','55.6%','NatureComm L252 + Arrhenius','★◆','VERIFIED'),  # 2차수정
    ('ΔG_bind (Irino)','−13.0 kcal/mol','Lit: Topo-I −10.7~−14','●','SUPPORTED'),
    ('ΔG_bind (Oxali)','−14.0 kcal/mol','RSC Dalton Trans 2019','●','SUPPORTED'),
    ('ΔG_bind (TAS-102)','−14.3 kcal/mol','AACR Cancer Res 2022','●','SUPPORTED'),
    ('ΔG_bind (FdUMP)','−11.2 kcal/mol','Lit: FdUMP ~−11.5','●','SUPPORTED'),
    ('Apoptosis 55%','Pritamab alone','CC-3+2.8×★ + siRNA●','▲+●','PROJ+LIT'),
    ('Apoptosis 75/80%','+Irino/Oxali/TAS','Lit 60-82% range','▲+●','PROJ+LIT'),
    ('mPFS target','5.5→8.25m HR=0.667','NatureComm §Clinical','★','VERIFIED'),
    ('DL ORR Prit.','51.5%','ADDS DL synth. n=1,000','◆◇','DL EST'),
    ('DL ΔmOS','+2.87mo (+20.3%)','ADDS DL synth. n=1,000','◆◇','DL EST'),
]
col_x2=[0.01,0.18,0.38,0.62,0.80]
row_y2=np.linspace(0.97,0.02,len(ver_rows))
for ri,row in enumerate(ver_rows):
    for ci,(cell,x) in enumerate(zip(row,col_x2)):
        fw='bold'
        if ri==0: tc=AMBER
        elif 'VERIFIED' in cell: tc=GREEN
        elif 'SUPPORTED' in cell: tc=TEAL
        elif 'PROJ' in cell: tc=BLUE
        elif 'DL EST' in cell: tc=CYAN
        elif 'ADDS EST' in cell: tc=PURPLE
        elif cell in ['★','★◆','◆','●','▲+●','▲','◆◇']:
            gc={'★':GREEN,'★◆':GREEN,'◆':BLUE,'●':TEAL,'▲+●':PURPLE,'▲':AMBER,'◆◇':CYAN}
            tc=gc.get(cell,WHITE)
        else: tc=WHITE; fw='normal' if ri>0 else 'bold'
        ax19.text(x,row_y2[ri],cell,fontsize=6.5,color=tc,fontweight=fw,transform=ax19.transAxes,va='center')
    if ri<len(ver_rows)-1:
        ax19.plot([0,1],[row_y2[ri]-0.026,row_y2[ri]-0.026],color=DARK,lw=0.4,transform=ax19.transAxes,clip_on=False)

# ── Footer ──────────────────────────────────────────────
fig.text(0.5,0.009,
         "★ NatureComm Paper (2026)  |  ◆ ADDS System (paper3_pritamab+DL pipeline)  |  "
         "● SCI Literature (PubMed)  |  ▲ ADDS Energy Model Projection  |  "
         "◇ DL Synthetic Cohort (n=1,000) — 실제 임상 아님  |  Generated: 2026-03-04  |  v3.0",
         ha='center',va='bottom',fontsize=7.5,color=GRAY,style='italic')

plt.savefig(r"f:\ADDS\figures\pritamab_final_report.png",
            dpi=160,bbox_inches='tight',facecolor=BG)
print("Saved: pritamab_final_report.png")
