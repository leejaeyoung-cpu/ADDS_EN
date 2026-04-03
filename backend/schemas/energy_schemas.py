"""
Pydantic schemas for Energy Framework API
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional


class EnergyPredictRequest(BaseModel):
    """Request for energy-based drug efficacy prediction"""
    drug_name: str = Field(..., description="Drug name (must be in DrugComb or known)")
    target_gene: Optional[str] = Field(None, description="Primary target gene (e.g., PRNP, EGFR)")
    kd_nm: Optional[float] = Field(None, ge=0.001, description="Dissociation constant (nM)")
    ic50_nm: Optional[float] = Field(None, ge=0.001, description="IC50 (nM)")
    cell_line: Optional[str] = Field("HCT116", description="Cell line")
    mutations: Optional[Dict[str, str]] = Field(None, description="e.g. {'RAS': 'KRAS_G13D'}")
    combination_drug: Optional[str] = Field(None, description="Second drug for combo prediction")
    combination_ic50_nm: Optional[float] = Field(None, description="IC50 of second drug")


class EnergyPredictResponse(BaseModel):
    """Response with energy-based predictions"""
    drug_name: str
    binding_energy_kcal: float = Field(..., description="ΔG binding (kcal/mol)")
    predicted_tumor_suppression_pct: float
    predicted_ic50_nm: float
    predicted_synergy_ci: Optional[float] = None
    synergy_interpretation: Optional[str] = None
    pathway_energies: Dict[str, float] = Field(default_factory=dict)
    model_version: str = "v3_gnn"


class DrugCombValidationRequest(BaseModel):
    """Request to validate model against DrugComb real data"""
    drug_pairs: Optional[List[Dict[str, str]]] = Field(
        None, description="Specific pairs to validate; None = all available")
    cell_line: Optional[str] = Field(None, description="Filter by cell line")


class DrugCombValidationResponse(BaseModel):
    """Cross-validation results against real DrugComb data"""
    n_pairs_validated: int
    n_datapoints: int
    pearson_r_loewe: float
    rmse_loewe: float
    classification_accuracy: float = Field(
        ..., description="% correctly classified as synergistic/antagonistic")
    per_pair_results: List[Dict] = Field(default_factory=list)
    model_version: str = "v3_gnn"


class PathwayEdge(BaseModel):
    source: str
    target: str
    weight: float
    source_type: str = Field("literature", description="literature, biogrid, or learned")


class PathwayGraphResponse(BaseModel):
    """Learned pathway graph structure"""
    n_nodes: int
    n_active_edges: int
    nodes: List[str]
    edges: List[PathwayEdge]
    calibration_source: str = Field("simulation", description="simulation, biogrid, or drugcomb")


class CalibrationStatusResponse(BaseModel):
    """Status of GNN calibration"""
    biogrid_ppi_loaded: bool = False
    biogrid_ppi_count: int = 0
    drugcomb_rows_used: int = 0
    last_calibration: Optional[str] = None
    train_r_tumor_supp: Optional[float] = None
    train_r_ic50: Optional[float] = None
    validation_r_loewe: Optional[float] = None
