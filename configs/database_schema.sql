-- ADDS (AI Anticancer Drug Discovery System) Database Schema
-- PostgreSQL Schema for managing experiments, data, and results

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "postgis";

-- ================================================================
-- Core Tables
-- ================================================================

-- 1. Experiments Table
CREATE TABLE experiments (
    experiment_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    experiment_name VARCHAR(255) NOT NULL,
    experiment_type VARCHAR(100) NOT NULL, -- 'cell_viability', 'western_blot', 'imaging', etc.
    cell_line VARCHAR(100),
    date_performed DATE NOT NULL,
    performed_by VARCHAR(100),
    description TEXT,
    metadata JSONB, -- flexible storage for additional metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(experiment_name, date_performed)
);

-- 2. Drugs/Compounds Table
CREATE TABLE compounds (
    compound_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    compound_name VARCHAR(255) NOT NULL UNIQUE,
    compound_type VARCHAR(100), -- 'anticancer_drug', 'exosome', 'combination'
    cas_number VARCHAR(50),
    molecular_formula VARCHAR(255),
    molecular_weight FLOAT,
    target_pathways TEXT[],
    mechanism_of_action TEXT,
    pubchem_cid INTEGER,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 3. Drug Combinations Table
CREATE TABLE drug_combinations (
    combination_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    combination_name VARCHAR(255),
    compounds UUID[] NOT NULL, -- array of compound_ids
    concentrations FLOAT[] NOT NULL, -- corresponding concentrations
    concentration_units VARCHAR(50) DEFAULT 'μM',
    experiment_id UUID REFERENCES experiments(experiment_id) ON DELETE CASCADE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CHECK (array_length(compounds, 1) = array_length(concentrations, 1))
);

-- 4. Images Table
CREATE TABLE images (
    image_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    experiment_id UUID REFERENCES experiments(experiment_id) ON DELETE CASCADE,
    combination_id UUID REFERENCES drug_combinations(combination_id) ON DELETE SET NULL,
    file_path VARCHAR(500) NOT NULL,
    file_name VARCHAR(255) NOT NULL,
    file_format VARCHAR(20),
    image_type VARCHAR(100), -- 'microscopy', 'western_blot', 'flow_cytometry'
    acquisition_date TIMESTAMP,
    microscope_type VARCHAR(100),
    magnification VARCHAR(50),
    resolution_x INTEGER,
    resolution_y INTEGER,
    channels INTEGER,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 5. Image Analysis Results (Cellpose outputs)
CREATE TABLE image_analysis (
    analysis_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    image_id UUID REFERENCES images(image_id) ON DELETE CASCADE,
    algorithm VARCHAR(100) DEFAULT 'cellpose',
    model_version VARCHAR(50),
    num_cells_detected INTEGER,
    segmentation_masks_path VARCHAR(500),
    cell_features JSONB, -- morphological features per cell
    aggregated_features JSONB, -- mean, std, etc.
    processing_time_seconds FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 6. Experimental Results Table
CREATE TABLE results (
    result_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    experiment_id UUID REFERENCES experiments(experiment_id) ON DELETE CASCADE,
    combination_id UUID REFERENCES drug_combinations(combination_id) ON DELETE SET NULL,
    image_id UUID REFERENCES images(image_id) ON DELETE SET NULL,
    analysis_id UUID REFERENCES image_analysis(analysis_id) ON DELETE SET NULL,
    result_type VARCHAR(100), -- 'viability', 'ic50', 'apoptosis_rate', etc.
    value FLOAT,
    unit VARCHAR(50),
    standard_deviation FLOAT,
    n_replicates INTEGER,
    p_value FLOAT,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 7. Synergy Scores Table
CREATE TABLE synergy_scores (
    synergy_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    experiment_id UUID REFERENCES experiments(experiment_id) ON DELETE CASCADE,
    combination_id UUID REFERENCES drug_combinations(combination_id) ON DELETE CASCADE,
    method VARCHAR(50) NOT NULL, -- 'bliss', 'loewe', 'hsa', 'zip'
    synergy_score FLOAT NOT NULL,
    is_synergistic BOOLEAN,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ================================================================
-- AI/ML Tables
-- ================================================================

-- 8. Feature Vectors Table
CREATE TABLE feature_vectors (
    feature_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_type VARCHAR(100), -- 'image', 'text', 'combination'
    source_id UUID, -- reference to images, documents, or combinations
    feature_type VARCHAR(100), -- 'image_embedding', 'text_embedding', 'morphological'
    model_name VARCHAR(255),
    vector FLOAT[] NOT NULL,
    dimension INTEGER NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 9. Model Predictions Table
CREATE TABLE predictions (
    prediction_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name VARCHAR(255) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    combination_id UUID REFERENCES drug_combinations(combination_id),
    predicted_value FLOAT,
    prediction_type VARCHAR(100), -- 'efficacy', 'toxicity', 'synergy'
    confidence_score FLOAT,
    actual_value FLOAT, -- for validation
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 10. Model Training Runs (MLflow integration)
CREATE TABLE model_runs (
    run_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    mlflow_run_id VARCHAR(255) UNIQUE,
    model_name VARCHAR(255) NOT NULL,
    model_type VARCHAR(100),
    hyperparameters JSONB,
    metrics JSONB,
    artifacts_path VARCHAR(500),
    training_dataset_version VARCHAR(100),
    status VARCHAR(50), -- 'running', 'completed', 'failed'
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP
);

-- ================================================================
-- Document/Literature Tables
-- ================================================================

-- 11. Documents Table (papers, reports)
CREATE TABLE documents (
    document_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_type VARCHAR(100), -- 'paper', 'report', 'protocol'
    title TEXT NOT NULL,
    authors TEXT[],
    publication_date DATE,
    doi VARCHAR(255),
    file_path VARCHAR(500),
    extracted_text TEXT,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 12. Knowledge Base (extracted information)
CREATE TABLE knowledge_base (
    knowledge_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID REFERENCES documents(document_id) ON DELETE CASCADE,
    knowledge_type VARCHAR(100), -- 'drug_interaction', 'mechanism', 'toxicity'
    entity_1 VARCHAR(255), -- e.g., drug name
    entity_2 VARCHAR(255), -- e.g., another drug or target
    relationship VARCHAR(255), -- e.g., 'synergizes_with', 'inhibits'
    confidence_score FLOAT,
    evidence_text TEXT,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ================================================================
-- Indexes for Performance
-- ================================================================

CREATE INDEX idx_experiments_date ON experiments(date_performed);
CREATE INDEX idx_experiments_name ON experiments(experiment_name);
CREATE INDEX idx_images_experiment ON images(experiment_id);
CREATE INDEX idx_analysis_image ON image_analysis(image_id);
CREATE INDEX idx_results_experiment ON results(experiment_id);
CREATE INDEX idx_results_combination ON results(combination_id);
CREATE INDEX idx_synergy_combination ON synergy_scores(combination_id);
CREATE INDEX idx_predictions_model ON predictions(model_name, model_version);
CREATE INDEX idx_features_source ON feature_vectors(source_type, source_id);

-- GiST index for array operations
CREATE INDEX idx_combinations_compounds ON drug_combinations USING GIN(compounds);
CREATE INDEX idx_compounds_pathways ON compounds USING GIN(target_pathways);

-- JSONB indexes
CREATE INDEX idx_experiments_metadata ON experiments USING GIN(metadata);
CREATE INDEX idx_results_metadata ON results USING GIN(metadata);

-- ================================================================
-- Triggers for Updated Timestamp
-- ================================================================

CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_experiments_updated_at
    BEFORE UPDATE ON experiments
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ================================================================
-- Views for Common Queries
-- ================================================================

-- View: Latest experiment results with drug combinations
CREATE VIEW v_latest_results AS
SELECT 
    e.experiment_id,
    e.experiment_name,
    e.date_performed,
    dc.combination_name,
    dc.compounds,
    dc.concentrations,
    r.result_type,
    r.value,
    r.unit,
    s.synergy_score,
    s.method AS synergy_method
FROM experiments e
LEFT JOIN drug_combinations dc ON e.experiment_id = dc.experiment_id
LEFT JOIN results r ON dc.combination_id = r.combination_id
LEFT JOIN synergy_scores s ON dc.combination_id = s.combination_id
ORDER BY e.date_performed DESC;

-- View: Model performance summary
CREATE VIEW v_model_performance AS
SELECT 
    model_name,
    model_version,
    prediction_type,
    COUNT(*) as total_predictions,
    AVG(confidence_score) as avg_confidence,
    AVG(ABS(predicted_value - actual_value)) as mae,
    STDDEV(predicted_value - actual_value) as prediction_std
FROM predictions
WHERE actual_value IS NOT NULL
GROUP BY model_name, model_version, prediction_type;

-- ================================================================
-- Sample Data Insertion (Optional - for testing)
-- ================================================================

-- Insert sample compound
INSERT INTO compounds (compound_name, compound_type, mechanism_of_action) VALUES
('Doxorubicin', 'anticancer_drug', 'DNA intercalation and topoisomerase II inhibition'),
('Cisplatin', 'anticancer_drug', 'DNA cross-linking'),
('Paclitaxel', 'anticancer_drug', 'Microtubule stabilization');

-- ================================================================
-- Comments for Documentation
-- ================================================================

COMMENT ON TABLE experiments IS 'Main experiment metadata and tracking';
COMMENT ON TABLE compounds IS 'Drug and compound library';
COMMENT ON TABLE drug_combinations IS 'Cocktail combinations tested';
COMMENT ON TABLE images IS 'Microscopy and other imaging data';
COMMENT ON TABLE image_analysis IS 'Cellpose and other image analysis results';
COMMENT ON TABLE results IS 'Experimental measurements and outcomes';
COMMENT ON TABLE synergy_scores IS 'Calculated drug synergy scores';
COMMENT ON TABLE feature_vectors IS 'Extracted features for ML models';
COMMENT ON TABLE predictions IS 'AI model predictions and validations';
COMMENT ON TABLE model_runs IS 'ML model training tracking';
COMMENT ON TABLE documents IS 'Scientific literature and reports';
COMMENT ON TABLE knowledge_base IS 'Extracted knowledge from literature';
