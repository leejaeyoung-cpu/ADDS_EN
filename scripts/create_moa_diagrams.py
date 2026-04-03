"""
PrPc-KRAS MOA Pathway Visualization

Generate detailed mechanism of action diagram showing PrPc-RPSA-KRAS signaling cascade
and therapeutic intervention points.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np

def create_prpc_moa_diagram():
    """
    Create comprehensive MOA diagram for PrPc-KRAS signaling pathway
    """
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis('off')
    
    # Color scheme
    colors = {
        'prpc': '#FF6B6B',  # Red - PrPc
        'rpsa': '#4ECDC4',  # Teal - RPSA
        'kras': '#95E1D3',  # Light teal - KRAS
        'downstream': '#F3A683',  # Orange - downstream
        'effects': '#786FA6',  # Purple - cellular effects
        'intervention': '#F8B500',  # Gold - intervention points
        'membrane': '#E8E8E8'  # Gray - membrane
    }
    
    # Title
    ax.text(5, 13.5, 'PrPc-RPSA-KRAS Signaling Cascade in Cancer', 
            ha='center', fontsize=20, fontweight='bold')
    ax.text(5, 13.0, 'Mechanism of Action & Therapeutic Intervention Points',
            ha='center', fontsize=14, style='italic')
    
    # Draw cell membrane
    membrane = FancyBboxPatch((0.5, 8), 9, 0.3, 
                              boxstyle="round,pad=0.05", 
                              edgecolor='black', facecolor=colors['membrane'],
                              linewidth=3, linestyle='--')
    ax.add_patch(membrane)
    ax.text(0.3, 8.15, 'Cell Membrane', fontsize=10, rotation=90, va='center', fontweight='bold')
    
    # === EXTRACELLULAR REGION ===
    ax.text(5, 11.5, 'EXTRACELLULAR', ha='center', fontsize=14, 
            fontweight='bold', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    
    # PrPc
    prpc_box = FancyBboxPatch((1.5, 9.5), 1.5, 1.2,
                              boxstyle="round,pad=0.1",
                              edgecolor='black', facecolor=colors['prpc'],
                              linewidth=2.5)
    ax.add_patch(prpc_box)
    ax.text(2.25, 10.3, 'PrPc', ha='center', fontsize=14, fontweight='bold')
    ax.text(2.25, 9.85, '(Prion Protein)', ha='center', fontsize=9)
    
    # RPSA/37LRP (membrane-anchored)
    rpsa_box = FancyBboxPatch((4.5, 8.5), 1.8, 1.5,
                              boxstyle="round,pad=0.1",
                              edgecolor='black', facecolor=colors['rpsa'],
                              linewidth=2.5)
    ax.add_patch(rpsa_box)
    ax.text(5.4, 9.5, 'RPSA', ha='center', fontsize=14, fontweight='bold')
    ax.text(5.4, 9.15, '(37/67 kDa', ha='center', fontsize=9)
    ax.text(5.4, 8.85, 'Laminin Receptor)', ha='center', fontsize=9)
    
    # PrPc -> RPSA interaction
    arrow1 = FancyArrowPatch((3.0, 10.1), (4.5, 9.3),
                            arrowstyle='->', mutation_scale=30, linewidth=3,
                            color='black')
    ax.add_patch(arrow1)
    ax.text(3.7, 10, 'High affinity\nbinding', ha='center', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # === INTERVENTION POINT 1 ===
    intervention1 = Circle((1, 10.1), 0.4, color=colors['intervention'], 
                          ec='black', linewidth=2, zorder=10)
    ax.add_patch(intervention1)
    ax.text(1, 10.1, '1', ha='center', va='center', fontsize=16, fontweight='bold')
    ax.text(0.5, 11.2, 'Anti-PrPc\nAntibody', ha='center', fontsize=9,
            bbox=dict(boxstyle='round', facecolor=colors['intervention'], alpha=0.7))
    
    # === INTRACELLULAR REGION ===
    ax.text(5, 7.5, 'INTRACELLULAR', ha='center', fontsize=14,
            fontweight='bold', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    
    # KRAS (at membrane)
    kras_box = FancyBboxPatch((4.2, 6.5), 2.4, 1.2,
                              boxstyle="round,pad=0.1",
                              edgecolor='black', facecolor=colors['kras'],
                              linewidth=2.5)
    ax.add_patch(kras_box)
    ax.text(5.4, 7.3, 'KRAS-GTP', ha='center', fontsize=14, fontweight='bold')
    ax.text(5.4, 6.85, '(Activated)', ha='center', fontsize=10, style='italic')
    
    # RPSA -> KRAS interaction (through membrane)
    arrow2 = FancyArrowPatch((5.4, 8.5), (5.4, 7.7),
                            arrowstyle='->', mutation_scale=30, linewidth=3,
                            color='black')
    ax.add_patch(arrow2)
    ax.text(6.2, 8.0, 'Supports RAS\nactivation', ha='center', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # === INTERVENTION POINT 2 ===
    intervention2 = Circle((7.5, 7.1), 0.4, color=colors['intervention'],
                          ec='black', linewidth=2, zorder=10)
    ax.add_patch(intervention2)
    ax.text(7.5, 7.1, '2', ha='center', va='center', fontsize=16, fontweight='bold')
    ax.text(8.5, 7.1, 'KRAS G12C\nInhibitors', ha='left', fontsize=9,
            bbox=dict(boxstyle='round', facecolor=colors['intervention'], alpha=0.7))
    
    # Downstream pathways split
    # PI3K-AKT pathway (left)
    akt_box = FancyBboxPatch((1.5, 4.8), 1.8, 1.0,
                             boxstyle="round,pad=0.1",
                             edgecolor='black', facecolor=colors['downstream'],
                             linewidth=2)
    ax.add_patch(akt_box)
    ax.text(2.4, 5.5, 'PI3K-AKT', ha='center', fontsize=12, fontweight='bold')
    ax.text(2.4, 5.15, '(phosphorylated)', ha='center', fontsize=8)
    
    # RAF-MEK-ERK pathway (right)
    erk_box = FancyBboxPatch((7.0, 4.8), 1.8, 1.0,
                             boxstyle="round,pad=0.1",
                             edgecolor='black', facecolor=colors['downstream'],
                             linewidth=2)
    ax.add_patch(erk_box)
    ax.text(7.9, 5.5, 'RAF-MEK-ERK', ha='center', fontsize=12, fontweight='bold')
    ax.text(7.9, 5.15, '(ERK1/2 phos.)', ha='center', fontsize=8)
    
    # KRAS to pathways
    arrow_akt = FancyArrowPatch((4.5, 6.8), (3.3, 5.8),
                               arrowstyle='->', mutation_scale=25, linewidth=2.5,
                               color='darkred')
    ax.add_patch(arrow_akt)
    
    arrow_erk = FancyArrowPatch((6.3, 6.8), (7.5, 5.8),
                               arrowstyle='->', mutation_scale=25, linewidth=2.5,
                               color='darkred')
    ax.add_patch(arrow_erk)
    
    # Cellular effects (bottom)
    effects = [
        ('Cell Cycle\nProgression', 1.5, 3.0, '??S-phase\n??G1 arrest'),
        ('Proliferation', 3.5, 3.0, '??Cell division'),
        ('Angiogenesis', 5.5, 3.0, '??VEGF\n??Microvessels'),
        ('Metastasis', 7.5, 3.0, '??Migration\n??Invasion'),
    ]
    
    for effect_name, x, y, detail in effects:
        effect_box = FancyBboxPatch((x-0.6, y-0.4), 1.2, 0.8,
                                    boxstyle="round,pad=0.08",
                                    edgecolor='black', facecolor=colors['effects'],
                                    linewidth=1.5, alpha=0.8)
        ax.add_patch(effect_box)
        ax.text(x, y+0.15, effect_name, ha='center', fontsize=10, fontweight='bold')
        ax.text(x, y-0.15, detail, ha='center', fontsize=7)
    
    # Arrows from pathways to effects
    for x_eff in [1.5, 3.5, 5.5, 7.5]:
        if x_eff < 5:
            arrow_eff = FancyArrowPatch((2.4, 4.8), (x_eff, 3.8),
                                       arrowstyle='->', mutation_scale=20, 
                                       linewidth=1.5, color='purple', alpha=0.6)
        else:
            arrow_eff = FancyArrowPatch((7.9, 4.8), (x_eff, 3.8),
                                       arrowstyle='->', mutation_scale=20,
                                       linewidth=1.5, color='purple', alpha=0.6)
        ax.add_patch(arrow_eff)
    
    # Overall outcome
    outcome_box = FancyBboxPatch((3.5, 1.0), 3.0, 1.0,
                                 boxstyle="round,pad=0.1",
                                 edgecolor='darkred', facecolor='#FFE5E5',
                                 linewidth=3)
    ax.add_patch(outcome_box)
    ax.text(5.0, 1.7, 'TUMOR GROWTH', ha='center', fontsize=14, fontweight='bold', color='darkred')
    ax.text(5.0, 1.3, '& PROGRESSION', ha='center', fontsize=12, fontweight='bold', color='darkred')
    
    # Final arrows to outcome
    for x_eff in [1.5, 3.5, 5.5, 7.5]:
        arrow_final = FancyArrowPatch((x_eff, 2.6), (5.0, 2.0),
                                     arrowstyle='->', mutation_scale=15,
                                     linewidth=1.2, color='darkred', alpha=0.4)
        ax.add_patch(arrow_final)
    
    # Legend box
    legend_elements = [
        mpatches.Patch(facecolor=colors['prpc'], edgecolor='black', label='PrPc Protein'),
        mpatches.Patch(facecolor=colors['rpsa'], edgecolor='black', label='RPSA/Receptors'),
        mpatches.Patch(facecolor=colors['kras'], edgecolor='black', label='KRAS Signaling'),
        mpatches.Patch(facecolor=colors['downstream'], edgecolor='black', label='Downstream Kinases'),
        mpatches.Patch(facecolor=colors['effects'], edgecolor='black', label='Cellular Effects'),
        mpatches.Patch(facecolor=colors['intervention'], edgecolor='black', label='Intervention Points'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10, frameon=True)
    
    # Add key insights box
    insights_text = """KEY INSIGHTS:
    
??PrPc-RPSA complex supports KRAS-GTP activation
??Dual pathway activation: PI3K-AKT + RAF-MEK-ERK
??PrPc neutralization reduces RAS-GTP levels by ~50%
??Combination: PrPc block + KRAS inhibitor = synergy
??Validated in KRAS-mutant colorectal cancer models"""
    
    ax.text(0.3, 5.5, insights_text, fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', 
                     edgecolor='black', linewidth=2))
    
    # Add combination therapy box
    combo_text = """COMBINATION STRATEGY:
    
1️⃣ Anti-PrPc mAb
      +
2️⃣ KRAS G12C Inhibitor
      +
       5-FU
       
??Enhanced tumor 
   growth inhibition"""
    
    ax.text(9.2, 4.0, combo_text, fontsize=8, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen',
                     edgecolor='darkgreen', linewidth=2))
    
    # Footer
    ax.text(5, 0.3, 'Evidence: ResearchGate/NIH 2026 | KRAS-mutant CRC xenograft models',
            ha='center', fontsize=9, style='italic')
    ax.text(5, 0.05, 'Generated by ADDS AI-Powered Drug Discovery System',
            ha='center', fontsize=8, color='gray')
    
    plt.tight_layout()
    
    return fig


def create_intervention_points_diagram():
    """
    Create focused diagram showing therapeutic intervention points
    """
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Title
    ax.text(5, 11.5, 'Therapeutic Intervention Points', 
            ha='center', fontsize=18, fontweight='bold')
    ax.text(5, 11.0, 'PrPc-KRAS Pathway Targeting Strategy',
            ha='center', fontsize=14)
    
    intervention_points = [
        {
            'number': '1',
            'name': 'Anti-PrPc Antibody',
            'target': 'Extracellular PrPc',
            'mechanism': '??Block PrPc-RPSA interaction\n??Reduce RAS-GTP levels\n??Decrease AKT/ERK phosphorylation',
            'evidence': 'Preclinical: Dose-dependent tumor inhibition',
            'y': 9.0,
            'color': '#FF6B6B'
        },
        {
            'number': '2',
            'name': 'RPSA Inhibitor',
            'target': 'PrPc-RPSA Complex',
            'mechanism': '??Disrupt PrPc-RPSA complex\n??Prevent KRAS activation\n??Target cancer stem cells',
            'evidence': 'Preclinical: Inhibits CSC proliferation',
            'y': 7.0,
            'color': '#4ECDC4'
        },
        {
            'number': '3',
            'name': 'KRAS G12C Inhibitor',
            'target': 'Mutant KRAS',
            'mechanism': '??Direct KRAS inhibition\n??Block RAF-MEK-ERK cascade\n??Synergy with PrPc blockade',
            'evidence': 'Clinical: FDA approved (sotorasib, adagrasib)',
            'y': 5.0,
            'color': '#95E1D3'
        },
        {
            'number': '4',
            'name': '5-Fluorouracil (5-FU)',
            'target': 'DNA synthesis',
            'mechanism': '??Thymidylate synthase inhibition\n??Enhanced with PrPc neutralization\n??Reduced angiogenesis',
            'evidence': 'Preclinical: Combination shows synergy',
            'y': 3.0,
            'color': '#F3A683'
        },
    ]
    
    for i, point in enumerate(intervention_points):
        # Main box
        box = FancyBboxPatch((1.0, point['y']-0.8), 8.0, 1.5,
                            boxstyle="round,pad=0.1",
                            edgecolor='black', facecolor=point['color'],
                            linewidth=2, alpha=0.3)
        ax.add_patch(box)
        
        # Number circle
        circle = Circle((1.5, point['y']), 0.35, color=point['color'],
                       ec='black', linewidth=2.5, zorder=10)
        ax.add_patch(circle)
        ax.text(1.5, point['y'], point['number'], ha='center', va='center',
                fontsize=20, fontweight='bold')
        
        # Intervention name
        ax.text(2.3, point['y']+0.35, point['name'],
                fontsize=13, fontweight='bold', va='top')
        
        # Target
        ax.text(2.3, point['y']+0.05, f"Target: {point['target']}",
                fontsize=10, style='italic')
        
        # Mechanism
        ax.text(2.3, point['y']-0.3, point['mechanism'],
                fontsize=9, va='top')
        
        # Evidence
        ax.text(7.5, point['y'], point['evidence'],
                fontsize=8, ha='left', va='center',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # Recommended combination box
    combo_box = FancyBboxPatch((1.0, 0.3), 8.0, 1.2,
                               boxstyle="round,pad=0.1",
                               edgecolor='darkgreen', facecolor='lightgreen',
                               linewidth=3, alpha=0.5)
    ax.add_patch(combo_box)
    
    ax.text(5.0, 1.2, '�?RECOMMENDED COMBINATION STRATEGY �?,
            ha='center', fontsize=13, fontweight='bold')
    ax.text(5.0, 0.85, 'Anti-PrPc Antibody (1) + KRAS G12C Inhibitor (3) + 5-FU (4)',
            ha='center', fontsize=11)
    ax.text(5.0, 0.55, 'Target Population: PrPc+ KRAS-mutant colorectal/pancreatic cancer',
            ha='center', fontsize=9, style='italic')
    
    plt.tight_layout()
    
    return fig


if __name__ == "__main__":
    print("Generating PrPc-KRAS MOA pathway diagrams...")
    
    # Create MOA diagram
    print("\n1. Creating detailed MOA pathway diagram...")
    fig1 = create_prpc_moa_diagram()
    output1 = "C:/Users/brook/Desktop/ADDS/data/analysis/prpc_moa_pathway.png"
    fig1.savefig(output1, dpi=300, bbox_inches='tight')
    print(f"   Saved to: {output1}")
    
    # Create intervention points diagram
    print("\n2. Creating intervention points diagram...")
    fig2 = create_intervention_points_diagram()
    output2 = "C:/Users/brook/Desktop/ADDS/data/analysis/prpc_intervention_points.png"
    fig2.savefig(output2, dpi=300, bbox_inches='tight')
    print(f"   Saved to: {output2}")
    
    print("\n??MOA diagrams generation complete!")
    print(f"\nGenerated files:")
    print(f"  - {output1}")
    print(f"  - {output2}")

