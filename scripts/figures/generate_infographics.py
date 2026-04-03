"""
ADDS System Architecture Infographic Generator
Creates professional PPT-ready diagrams with ivory color scheme
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np

# Ivory color palette
COLORS = {
    'bg': '#F5F5DC',           # Ivory background
    'box_light': '#FAEBD7',    # Light beige
    'box_medium': '#E8DCC4',   # Medium beige
    'box_dark': '#D4C4A8',     # Dark beige
    'accent': '#B8B89A',       # Sage green
    'accent_dark': '#8B8B7A',  # Dark sage
    'text': '#4A4A4A',         # Dark gray
    'text_light': '#6A6A6A',   # Medium gray
    'arrow': '#888888',        # Gray arrows
}

def create_figure(width=16, height=10, title=""):
    """Create base figure with ivory background"""
    fig, ax = plt.subplots(figsize=(width, height), facecolor=COLORS['bg'])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')
    ax.set_facecolor(COLORS['bg'])
    
    if title:
        ax.text(50, 95, title, fontsize=24, fontweight='bold', 
                ha='center', va='top', color=COLORS['text'])
    
    return fig, ax

def draw_box(ax, x, y, width, height, text, color, textsize=12, bold=False):
    """Draw rounded rectangle box with text"""
    box = FancyBboxPatch((x, y), width, height, 
                         boxstyle="round,pad=0.1", 
                         facecolor=color, 
                         edgecolor=COLORS['accent_dark'], 
                         linewidth=2)
    ax.add_patch(box)
    
    weight = 'bold' if bold else 'normal'
    ax.text(x + width/2, y + height/2, text, 
            fontsize=textsize, ha='center', va='center', 
            color=COLORS['text'], weight=weight,
            multialignment='center')

def draw_arrow(ax, x1, y1, x2, y2, label="", curved=False):
    """Draw arrow between points"""
    if curved:
        connectionstyle = "arc3,rad=.3"
    else:
        connectionstyle = "arc3,rad=0"
    
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle='->', mutation_scale=20,
                           color=COLORS['arrow'], linewidth=2,
                           connectionstyle=connectionstyle)
    ax.add_patch(arrow)
    
    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x, mid_y, label, fontsize=9, 
                ha='center', va='bottom', 
                color=COLORS['text_light'],
                bbox=dict(boxstyle='round,pad=0.3', 
                         facecolor=COLORS['bg'], 
                         edgecolor='none', alpha=0.8))

def draw_database(ax, x, y, width, height, text):
    """Draw database cylinder"""
    # Top ellipse
    ellipse_top = mpatches.Ellipse((x + width/2, y + height), 
                                   width, height*0.3,
                                   facecolor=COLORS['accent'],
                                   edgecolor=COLORS['accent_dark'],
                                   linewidth=2)
    ax.add_patch(ellipse_top)
    
    # Rectangle body
    rect = mpatches.Rectangle((x, y + height*0.85), width, height*0.15,
                              facecolor=COLORS['accent'],
                              edgecolor=COLORS['accent_dark'],
                              linewidth=2)
    ax.add_patch(rect)
    
    # Bottom ellipse
    ellipse_bottom = mpatches.Ellipse((x + width/2, y + height*0.85), 
                                      width, height*0.3,
                                      facecolor=COLORS['accent'],
                                      edgecolor=COLORS['accent_dark'],
                                      linewidth=2)
    ax.add_patch(ellipse_bottom)
    
    ax.text(x + width/2, y + height*0.92, text, 
            fontsize=10, ha='center', va='center',
            color='white', weight='bold',
            multialignment='center')


# ============================================================================
# DIAGRAM 1: Overall System Architecture
# ============================================================================
def generate_system_architecture():
    fig, ax = create_figure(18, 12, "ADDS CDSS System Architecture")
    
    # Layer 1: UI Layer
    ax.text(5, 88, "UI Layer", fontsize=14, weight='bold', color=COLORS['text'])
    draw_box(ax, 8, 80, 18, 6, "Patient\nManagement", COLORS['box_light'], 11)
    draw_box(ax, 28, 80, 18, 6, "Treatment\nOutcomes", COLORS['box_light'], 11)
    draw_box(ax, 48, 80, 18, 6, "Physician\nNotes", COLORS['box_light'], 11)
    draw_box(ax, 68, 80, 18, 6, "Dashboard", COLORS['box_light'], 11)
    
    # Layer 2: Orchestration
    ax.text(5, 70, "Orchestration", fontsize=14, weight='bold', color=COLORS['text'])
    draw_box(ax, 20, 62, 25, 6, "CDSS Orchestrator", COLORS['box_medium'], 11, True)
    draw_box(ax, 50, 62, 25, 6, "Dynamic Analysis", COLORS['box_medium'], 11, True)
    
    # Arrows from UI to Orchestration
    draw_arrow(ax, 17, 80, 27, 68)
    draw_arrow(ax, 37, 80, 32, 68)
    draw_arrow(ax, 57, 80, 62, 68)
    draw_arrow(ax, 77, 80, 67, 68)
    
    # Layer 3: Services
    ax.text(5, 52, "Service Layer", fontsize=14, weight='bold', color=COLORS['text'])
    draw_box(ax, 5, 43, 16, 7, "CT Metadata\nExtractor", COLORS['box_dark'], 10)
    draw_box(ax, 23, 43, 16, 7, "Cell Feature\nExtractor", COLORS['box_dark'], 10)
    draw_box(ax, 41, 43, 16, 7, "NLP\nParser", COLORS['box_dark'], 10)
    draw_box(ax, 59, 43, 16, 7, "Metadata\nAggregator", COLORS['box_dark'], 10)
    draw_box(ax, 77, 43, 16, 7, "Daily ML\nTrainer", COLORS['box_dark'], 10)
    
    # Arrows from Orchestration to Services
    draw_arrow(ax, 28, 62, 13, 50)
    draw_arrow(ax, 32, 62, 31, 50)
    draw_arrow(ax, 37, 62, 49, 50)
    draw_arrow(ax, 62, 62, 67, 50)
    draw_arrow(ax, 70, 62, 85, 50)
    
    # Layer 4: Data Layer
    ax.text(5, 33, "Data Layer", fontsize=14, weight='bold', color=COLORS['text'])
    draw_database(ax, 8, 18, 14, 12, "Patient\nDB")
    draw_database(ax, 26, 18, 14, 12, "CT\nAnalyses")
    draw_database(ax, 44, 18, 14, 12, "Cell\nImages")
    draw_database(ax, 62, 18, 14, 12, "Treatments")
    draw_database(ax, 80, 18, 14, 12, "ML\nModels")
    
    # Arrows from Services to Data
    draw_arrow(ax, 13, 43, 15, 30)
    draw_arrow(ax, 31, 43, 33, 30)
    draw_arrow(ax, 49, 43, 51, 30)
    draw_arrow(ax, 67, 43, 69, 30)
    draw_arrow(ax, 85, 43, 87, 30)
    
    # Feedback loop
    draw_arrow(ax, 87, 30, 90, 60, curved=True)
    ax.text(92, 45, "Predictions", fontsize=9, rotation=90, 
            ha='center', color=COLORS['text_light'])
    
    # Bottom text
    ax.text(50, 8, "Continuous Learning Pipeline: Data → Training → Deployment → Inference → Outcomes → Data", 
            fontsize=11, ha='center', style='italic', color=COLORS['text_light'])
    
    plt.tight_layout()
    plt.savefig('infographic_1_system_architecture.png', dpi=300, facecolor=COLORS['bg'])
    print("✓ Generated: infographic_1_system_architecture.png")
    plt.close()


# ============================================================================
# DIAGRAM 2: CT Analysis Pipeline
# ============================================================================
def generate_ct_pipeline():
    fig, ax = create_figure(16, 10, "CT Scan Analysis Pipeline")
    
    # Input
    draw_box(ax, 5, 70, 15, 12, "📊\nDICOM\nCT Scan\nFiles", COLORS['box_light'], 12, True)
    
    # Processing Stage 1: Preprocessing
    draw_box(ax, 28, 75, 18, 8, "DICOM Loading\n& Preprocessing", COLORS['box_medium'], 10)
    draw_arrow(ax, 20, 76, 28, 79)
    
    # Processing Stage 2: Segmentation
    draw_box(ax, 28, 63, 18, 8, "Organ Segmentation\n(TotalSegmentator)", COLORS['box_medium'], 10)
    draw_arrow(ax, 37, 75, 37, 71)
    
    # Processing Stage 3: Detection
    draw_box(ax, 28, 51, 18, 8, "3D Tumor\nDetection", COLORS['box_medium'], 10)
    draw_arrow(ax, 37, 63, 37, 59)
    
    # Analysis branches
    draw_box(ax, 54, 75, 18, 7, "Volume\nCalculation", COLORS['box_dark'], 9)
    draw_box(ax, 54, 66, 18, 7, "Diameter\nMeasurement", COLORS['box_dark'], 9)
    draw_box(ax, 54, 57, 18, 7, "HU Value\nAnalysis", COLORS['box_dark'], 9)
    draw_box(ax, 54, 48, 18, 7, "Shape Features\n(Sphericity)", COLORS['box_dark'], 9)
    
    draw_arrow(ax, 46, 55, 54, 78.5)
    draw_arrow(ax, 46, 55, 54, 69.5)
    draw_arrow(ax, 46, 55, 54, 60.5)
    draw_arrow(ax, 46, 55, 54, 51.5)
    
    # Output metadata
    draw_box(ax, 78, 62, 17, 20, "Extracted\nMetadata:\n\n• Tumor Volume\n• Max Diameter\n• Location (X,Y,Z)\n• HU Statistics\n• Shape Metrics\n• Slice Range", 
             COLORS['accent'], 9, True)
    
    draw_arrow(ax, 72, 78.5, 78, 72)
    draw_arrow(ax, 72, 69.5, 78, 72)
    draw_arrow(ax, 72, 60.5, 78, 72)
    draw_arrow(ax, 72, 51.5, 78, 72)
    
    # Database
    draw_database(ax, 78, 25, 17, 15, "CT Analysis\nDatabase")
    draw_arrow(ax, 86.5, 62, 86.5, 40)
    
    # Technical details box
    ax.text(5, 35, "Technical Details:", fontsize=11, weight='bold', color=COLORS['text'])
    details = [
        "• 3D Connected Component Analysis",
        "• Anatomical Saliency Filtering",
        "• HU Threshold: 20-150 (soft tissue)",
        "• Minimum volume: 100 mm³",
        "• Shape metrics: Sphericity, Compactness"
    ]
    for i, detail in enumerate(details):
        ax.text(5, 30 - i*4, detail, fontsize=9, color=COLORS['text_light'])
    
    plt.tight_layout()
    plt.savefig('infographic_2_ct_pipeline.png', dpi=300, facecolor=COLORS['bg'])
    print("✓ Generated: infographic_2_ct_pipeline.png")
    plt.close()


# ============================================================================
# DIAGRAM 3: Cellpose Analysis Pipeline
# ============================================================================
def generate_cellpose_pipeline():
    fig, ax = create_figure(16, 10, "Cellpose Cell Analysis Pipeline")
    
    # Input
    draw_box(ax, 5, 70, 15, 12, "🔬\nMicroscopy\nCell Images\n(H&E, IHC)", COLORS['box_light'], 12, True)
    
    # Cellpose Processing
    draw_box(ax, 28, 75, 20, 8, "Cellpose CNN\nSegmentation", COLORS['box_medium'], 11, True)
    draw_arrow(ax, 20, 76, 28, 79, "RGB Image")
    
    # Segmentation outputs
    draw_box(ax, 28, 63, 20, 8, "Nuclear Detection\n& Boundary Tracing", COLORS['box_medium'], 10)
    draw_arrow(ax, 38, 75, 38, 71)
    
    # Feature extraction branches
    draw_box(ax, 55, 77, 18, 6, "Cell Counting", COLORS['box_dark'], 9)
    draw_box(ax, 55, 69, 18, 6, "Ki-67 Index\nCalculation", COLORS['box_dark'], 9)
    draw_box(ax, 55, 61, 18, 6, "Morphology\nMetrics", COLORS['box_dark'], 9)
    
    draw_arrow(ax, 48, 67, 55, 80)
    draw_arrow(ax, 48, 67, 55, 72)
    draw_arrow(ax, 48, 67, 55, 64)
    
    # Deep learning features
    draw_box(ax, 28, 50, 20, 8, "ResNet50\nCNN Features", COLORS['accent'], 10, True)
    draw_arrow(ax, 38, 63, 38, 58, "Cell Masks")
    
    # Combined output
    draw_box(ax, 78, 62, 17, 22, "Cell Features:\n\n• Total Cells\n• Ki-67 %\n• Cell Area\n• Nuclear Size\n• Texture\n• CNN Embedding\n  (2048-dim)", 
             COLORS['accent'], 9, True)
    
    draw_arrow(ax, 73, 80, 78, 73)
    draw_arrow(ax, 73, 72, 78, 73)
    draw_arrow(ax, 73, 64, 78, 73)
    draw_arrow(ax, 48, 54, 78, 73)
    
    # Database
    draw_database(ax, 78, 25, 17, 15, "Cell Image\nMetadata")
    draw_arrow(ax, 86.5, 62, 86.5, 40)
    
    # Technical details
    ax.text(5, 40, "Cellpose Configuration:", fontsize=11, weight='bold', color=COLORS['text'])
    details = [
        "• Model: Cytoplasm + Nuclei",
        "• Diameter: Auto-detected",
        "• Flow threshold: 0.4",
        "• Probability threshold: 0.0",
        "• ResNet50: Pre-trained on ImageNet",
        "• Output: 2048-dimensional feature vector"
    ]
    for i, detail in enumerate(details):
        ax.text(5, 35 - i*4, detail, fontsize=9, color=COLORS['text_light'])
    
    plt.tight_layout()
    plt.savefig('infographic_3_cellpose_pipeline.png', dpi=300, facecolor=COLORS['bg'])
    print("✓ Generated: infographic_3_cellpose_pipeline.png")
    plt.close()


# ============================================================================
# DIAGRAM 4: Clinical Data Integration
# ============================================================================
def generate_clinical_integration():
    fig, ax = create_figure(16, 10, "Clinical Metadata Integration")
    
    # Three data sources at top
    draw_box(ax, 5, 75, 15, 10, "👤\nPatient\nDemographics\n\n• Age, Gender\n• Med History", 
             COLORS['box_light'], 9)
    draw_box(ax, 25, 75, 15, 10, "💊\nTreatment\nRecords\n\n• Regimens\n• Dosages", 
             COLORS['box_light'], 9)
    draw_box(ax, 45, 75, 15, 10, "📝\nPhysician\nNotes\n\n• Observations\n• Assessments", 
             COLORS['box_light'], 9)
    
    # NLP Processing (only for notes)
    draw_box(ax, 65, 75, 25, 10, "🧠 NLP Engine\n\n• Severity Extraction\n• Symptom Detection\n• Tumor Status\n• Medications", 
             COLORS['box_medium'], 9, True)
    draw_arrow(ax, 60, 80, 65, 80, "Free Text")
    
    # Aggregator
    draw_box(ax, 25, 57, 40, 8, "Metadata Aggregator", COLORS['box_dark'], 11, True)
    
    draw_arrow(ax, 12.5, 75, 30, 65)
    draw_arrow(ax, 32.5, 75, 40, 65)
    draw_arrow(ax, 77.5, 75, 60, 65)
    
    # Unified database
    draw_database(ax, 30, 35, 30, 15, "Unified Clinical\nMetadata")
    draw_arrow(ax, 45, 57, 45, 50)
    
    # Output branches
    draw_box(ax, 5, 18, 18, 8, "ML Training\nDataset", COLORS['accent'], 10)
    draw_box(ax, 27, 18, 18, 8, "Patient\nTimeline", COLORS['accent'], 10)
    draw_box(ax, 49, 18, 18, 8, "Clinical\nInsights", COLORS['accent'], 10)
    draw_box(ax, 71, 18, 18, 8, "Re-analysis\nTriggers", COLORS['accent'], 10)
    
    draw_arrow(ax, 35, 35, 14, 26)
    draw_arrow(ax, 40, 35, 36, 26)
    draw_arrow(ax, 50, 35, 58, 26)
    draw_arrow(ax, 55, 35, 80, 26)
    
    # Intelligence features (right side)
    ax.text(92, 55, "Intelligence:", fontsize=10, weight='bold', 
            color=COLORS['text'], ha='right')
    features = ["✓ Versioning", "✓ Auto Re-analysis", "✓ Quality Metrics"]
    for i, feat in enumerate(features):
        ax.text(92, 50 - i*4, feat, fontsize=9, ha='right', color=COLORS['text_light'])
    
    plt.tight_layout()
    plt.savefig('infographic_4_clinical_integration.png', dpi=300, facecolor=COLORS['bg'])
    print("✓ Generated: infographic_4_clinical_integration.png")
    plt.close()


# ============================================================================
# DIAGRAM 5: Pharmacodynamics Model
# ============================================================================
def generate_pharmacodynamics():
    fig, ax = create_figure(16, 10, "Pharmacodynamics Prediction System")
    
    # Input sources
    ax.text(10, 88, "Multi-Modal Input Features", fontsize=12, weight='bold', color=COLORS['text'])
    
    draw_box(ax, 5, 75, 12, 10, "CT Features\n\n• Volume\n• Location\n• HU values", 
             COLORS['box_light'], 9)
    draw_box(ax, 19, 75, 12, 10, "Cell Features\n\n• Ki-67\n• CNN embed\n• Count", 
             COLORS['box_light'], 9)
    draw_box(ax, 33, 75, 12, 10, "Clinical\n\n• Age, Stage\n• History\n• ECOG", 
             COLORS['box_light'], 9)
    
    # Feature processing
    draw_box(ax, 10, 60, 30, 8, "Feature Normalization & Concatenation", 
             COLORS['box_medium'], 10)
    
    draw_arrow(ax, 11, 75, 15, 68)
    draw_arrow(ax, 25, 75, 25, 68)
    draw_arrow(ax, 39, 75, 35, 68)
    
    # Model architecture
    draw_box(ax, 55, 75, 35, 20, "PharmacodynamicsPredictor\n(PyTorch Neural Network)\n\n" +
             "Input Layer (256 features)\n↓\nHidden (128) + Dropout\n↓\n" +
             "Hidden (64) + Dropout\n↓\nMulti-Task Heads",
             COLORS['accent'], 9, True)
    
    draw_arrow(ax, 40, 64, 55, 78)
    
    # Output predictions
    draw_box(ax, 55, 48, 17, 8, "Treatment\nResponse\n(CR/PR/SD/PD)", 
             COLORS['box_dark'], 9)
    draw_box(ax, 73, 48, 17, 8, "Survival\nPrediction\n(PFS days)", 
             COLORS['box_dark'], 9)
    
    draw_arrow(ax, 64, 75, 64, 56)
    draw_arrow(ax, 81.5, 75, 81.5, 56)
    
    # Outcomes database
    draw_database(ax, 55, 25, 35, 15, "Treatment Outcomes\nDatabase")
    draw_arrow(ax, 64, 48, 64, 40)
    draw_arrow(ax, 81.5, 48, 81.5, 40)
    
    # Feedback loop
    draw_arrow(ax, 55, 35, 20, 60, curved=True)
    ax.text(35, 50, "Continuous\nLearning", fontsize=9, ha='center',
            color=COLORS['text_light'], style='italic')
    
    # Training details
    ax.text(5, 18, "Training Configuration:", fontsize=10, weight='bold', color=COLORS['text'])
    details = [
        "• Optimizer: Adam (lr=0.001)",
        "• Scheduler: ReduceLROnPlateau",
        "• Early Stopping: patience=15",
        "• Batch Size: 32",
        "• Max Epochs: 100",
        "• Loss: Cross-Entropy + MSE"
    ]
    for i, detail in enumerate(details):
        ax.text(5, 13 - i*1.8, detail, fontsize=8, color=COLORS['text_light'])
    
    plt.tight_layout()
    plt.savefig('infographic_5_pharmacodynamics.png', dpi=300, facecolor=COLORS['bg'])
    print("✓ Generated: infographic_5_pharmacodynamics.png")
    plt.close()


# ============================================================================
# DIAGRAM 6: Daily ML Training Cycle
# ============================================================================
def generate_ml_training_cycle():
    fig, ax = create_figure(16, 10, "Daily Deep Learning Training Cycle")
    
    # Timeline
    ax.text(50, 88, "24-Hour Automated Learning Cycle", fontsize=13, 
            weight='bold', ha='center', color=COLORS['text'])
    
    # Time markers
    times = [("00:00", 10), ("02:00", 30), ("04:00", 50), ("06:00", 70), ("12:00", 90)]
    for time, x in times:
        ax.text(x, 78, time, fontsize=9, ha='center', color=COLORS['text_light'])
        ax.plot([x, x], [77, 75], 'k-', linewidth=1, color=COLORS['arrow'])
    
    # Stage 1: Data Collection (all day)
    draw_box(ax, 5, 65, 40, 8, "📥 Continuous Data Collection\n" +
             "New patients • Treatment outcomes • Physician notes",
             COLORS['box_light'], 9)
    
    # Stage 2: Training trigger (2AM)
    draw_box(ax, 28, 52, 18, 8, "⏰ 2AM Trigger\nScheduled Job", 
             COLORS['box_medium'], 10, True)
    draw_arrow(ax, 25, 65, 32, 60)
    
    # Stage 3: Dataset preparation
    draw_box(ax, 52, 65, 18, 8, "Dataset Prep\n" +
             "Aggregation\nNormalization",
             COLORS['box_medium'], 9)
    draw_arrow(ax, 37, 56, 52, 69)
    
    # Stage 4: Training
    draw_box(ax, 52, 52, 18, 8, "🧠 Model Training\n" +
             "100 epochs max\nEarly stopping",
             COLORS['accent'], 9, True)
    draw_arrow(ax, 61, 65, 61, 60)
    
    # Stage 5: Evaluation
    draw_box(ax, 75, 65, 18, 8, "📊 Evaluation\n" +
             "Validation metrics\nPerformance check",
             COLORS['box_dark'], 9)
    draw_arrow(ax, 70, 56, 75, 69)
    
    # Decision point
    draw_box(ax, 75, 52, 18, 8, "Better than\ncurrent?", 
             COLORS['box_medium'], 10, True)
    draw_arrow(ax, 84, 65, 84, 60)
    
    # Stage 6a: Deploy (YES)
    draw_box(ax, 75, 38, 18, 8, "✅ Deploy Model\n" +
             "Update inference\nengine",
             COLORS['accent'], 9, True)
    ax.text(84, 50, "YES", fontsize=9, ha='center', weight='bold', color='green')
    draw_arrow(ax, 84, 52, 84, 46)
    
    # Stage 6b: Keep current (NO)
    draw_box(ax, 52, 38, 18, 8, "Keep Current\nModel", 
             COLORS['box_light'], 9)
    ax.text(75, 50, "NO", fontsize=9, ha='right', weight='bold', color='red')
    draw_arrow(ax, 75, 52, 70, 42)
    
    # Stage 7: Inference
    draw_database(ax, 60, 18, 23, 12, "Production\nInference Engine")
    draw_arrow(ax, 84, 38, 71.5, 30)
    draw_arrow(ax, 61, 38, 61, 30)
    
    # Continuous cycle arrow
    draw_arrow(ax, 93, 69, 93, 25, curved=True)
    draw_arrow(ax, 93, 25, 45, 65, curved=True)
    ax.text(95, 47, "Next Day", fontsize=9, rotation=90, ha='center',
            color=COLORS['text_light'], style='italic')
    
    # Statistics box
    ax.text(5, 30, "Performance Metrics:", fontsize=10, weight='bold', color=COLORS['text'])
    metrics = [
        "• Training: ~2-3 hours (GPU)",
        "• Samples: 500-1000+ patients",
        "• Validation Split: 20%",
        "• Model Size: ~2.5 MB",
        "• Inference: <100ms per patient"
    ]
    for i, metric in enumerate(metrics):
        ax.text(5, 25 - i*3, metric, fontsize=8, color=COLORS['text_light'])
    
    plt.tight_layout()
    plt.savefig('infographic_6_ml_training_cycle.png', dpi=300, facecolor=COLORS['bg'])
    print("✓ Generated: infographic_6_ml_training_cycle.png")
    plt.close()


# ============================================================================
# DIAGRAM 7: Drug Cocktail Formation
# ============================================================================
def generate_drug_cocktail():
    fig, ax = create_figure(16, 10, "AI-Powered Drug Cocktail Formation")
    
    # Patient profile input
    draw_box(ax, 5, 75, 20, 12, "👤 Patient Profile\n\n• Tumor: 15mm, Stage IIIB\n• Ki-67: 68%\n• Age: 62, ECOG: 1\n• History: HTN", 
             COLORS['box_light'], 9)
    
    # AI Inference
    draw_box(ax, 30, 77, 22, 10, "🧠 AI Model Inference\n\n" +
             "PharmacodynamicsPredictor\nPredicted Response: 72%\nPFS: 245 days",
             COLORS['accent'], 9, True)
    draw_arrow(ax, 25, 81, 30, 82)
    
    # Knowledge base
    draw_box(ax, 30, 63, 22, 10, "📚 Drug Database\n\n" +
             "• 50+ anticancer agents\n• Synergy matrix\n• Toxicity profiles",
             COLORS['box_medium'], 9)
    
    # Drug selection engine
    draw_box(ax, 58, 70, 30, 17, "💊 Drug Selection Engine\n\n" +
             "1. Synergy Analysis\n   → Identify compatible combinations\n\n" +
             "2. Dosage Optimization\n   → Balance efficacy vs. toxicity\n\n" +
             "3. Risk Assessment\n   → Contraindications check",
             COLORS['box_dark'], 8.5)
    
    draw_arrow(ax, 52, 82, 58, 78.5)
    draw_arrow(ax, 52, 68, 58, 78.5)
    
    # Recommended cocktail
    draw_box(ax, 58, 48, 30, 15, "🎯 Recommended Cocktail\n\n" +
             "Drug A: 75 mg/m² (Day 1)\n" +
             "Drug B: 1000 mg/m² (Day 1,8)\n" +
             "Drug C: 400 mg (Daily)\n\n" +
             "Cycle: 21 days | Synergy Score: 0.85",
             COLORS['accent'], 8.5, True)
    draw_arrow(ax, 73, 70, 73, 63)
    
    # Clinical review
    draw_box(ax, 58, 30, 30, 12, "👨‍⚕️ Physician Review\n\n" +
             "• Verify compatibility\n• Adjust for patient factors\n• Approve or modify",
             COLORS['box_light'], 9)
    draw_arrow(ax, 73, 48, 73, 42)
    
    # Final treatment plan
    draw_box(ax, 58, 12, 30, 12, "✅ Approved Treatment Plan\n\n" +
             "Documented in EMR\nScheduled for administration\nMonitoring protocol set",
             COLORS['accent'], 9, True)
    draw_arrow(ax, 73, 30, 73, 24)
    
    # Left side: Decision factors
    ax.text(5, 58, "Decision Factors:", fontsize=10, weight='bold', color=COLORS['text'])
    factors = [
        "✓ Tumor Characteristics",
        "  • Size, location, histology",
        "✓ Patient Factors",
        "  • Age, comorbidities, organ function",
        "✓ Treatment History",
        "  • Prior regimens, resistance",
        "✓ Drug Synergies",
        "  • Additive, synergistic effects",
        "✓ Safety Profile",
        "  • Toxicity, contraindications",
        "✓ Cost-Effectiveness",
        "  • Insurance, availability"
    ]
    for i, factor in enumerate(factors):
        size = 9 if factor.startswith('✓') else 8
        ax.text(5, 53 - i*3, factor, fontsize=size, color=COLORS['text_light'])
    
    # Bottom: Success metrics
    ax.text(5, 18, "Success Metrics:", fontsize=9, weight='bold', color=COLORS['text'])
    ax.text(5, 14, "• Prediction Accuracy: 78% (historical validation)", 
            fontsize=8, color=COLORS['text_light'])
    ax.text(5, 11, "• Treatment Adherence: 92%", 
            fontsize=8, color=COLORS['text_light'])
    ax.text(5, 8, "• Physician Agreement Rate: 85%", 
            fontsize=8, color=COLORS['text_light'])
    
    plt.tight_layout()
    plt.savefig('infographic_7_drug_cocktail.png', dpi=300, facecolor=COLORS['bg'])
    print("✓ Generated: infographic_7_drug_cocktail.png")
    plt.close()


# ============================================================================
# Main execution
# ============================================================================
if __name__ == "__main__":
    print("\nGenerating ADDS System Infographics for PPT...")
    print("=" * 60)
    
    generate_system_architecture()
    generate_ct_pipeline()
    generate_cellpose_pipeline()
    generate_clinical_integration()
    generate_pharmacodynamics()
    generate_ml_training_cycle()
    generate_drug_cocktail()
    
    print("=" * 60)
    print("All 7 infographics generated successfully!")
    print("\nFiles created:")
    print("  1. infographic_1_system_architecture.png")
    print("  2. infographic_2_ct_pipeline.png")
    print("  3. infographic_3_cellpose_pipeline.png")
    print("  4. infographic_4_clinical_integration.png")
    print("  5. infographic_5_pharmacodynamics.png")
    print("  6. infographic_6_ml_training_cycle.png")
    print("  7. infographic_7_drug_cocktail.png")
    print("\nAll images are 300 DPI, ready for professional PPT presentations")

