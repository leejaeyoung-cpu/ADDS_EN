"""
Sample data generator for precision oncology testing
Creates realistic patient profiles with clinical and genomic data
"""

import json
from datetime import datetime, timedelta
import random
from pathlib import Path


class SampleDataGenerator:
    """샘플 환자 데이터 생성기"""
    
    def __init__(self):
        self.cancer_types = ['Colorectal', 'Lung', 'Breast', 'Pancreatic', 'Gastric']
        self.stages = ['I', 'II', 'III', 'IV']
        self.grades = ['well', 'moderate', 'poor']
        
        self.common_genes = [
            'KRAS', 'NRAS', 'BRAF', 'EGFR', 'ALK', 'ROS1', 
            'TP53', 'PIK3CA', 'PTEN', 'APC', 'HER2', 'MET'
        ]
        
        self.variant_types = ['SNV', 'CNV', 'Fusion', 'Indel']
        self.pathogenicity_levels = ['Pathogenic', 'Likely pathogenic', 'VUS', 'Benign']
    
    def generate_patient(self, patient_id: str, scenario: str = 'random') -> dict:
        """
        환자 데이터 생성
        
        Args:
            patient_id: 환자 ID
            scenario: 시나리오 타입
                - 'random': 랜덤 생성
                - 'high_risk': 고위험 환자
                - 'targeted': 표적치료 후보
                - 'low_risk': 저위험 환자
        """
        if scenario == 'high_risk':
            return self._generate_high_risk_patient(patient_id)
        elif scenario == 'targeted':
            return self._generate_targeted_therapy_patient(patient_id)
        elif scenario == 'low_risk':
            return self._generate_low_risk_patient(patient_id)
        else:
            return self._generate_random_patient(patient_id)
    
    def _generate_random_patient(self, patient_id: str) -> dict:
        """랜덤 환자 생성"""
        age = random.randint(45, 80)
        cancer_type = random.choice(self.cancer_types)
        
        patient = {
            'patient_id': patient_id,
            'age': age,
            'gender': random.choice(['Male', 'Female']),
            'cancer_type': cancer_type,
            'stage': random.choice(self.stages),
            'grade': random.choice(self.grades),
            'primary_site': self._get_primary_site(cancer_type),
            'ecog_score': random.randint(0, 2),
            'diagnosis_date': (datetime.now() - timedelta(days=random.randint(30, 365))).strftime('%Y-%m-%d'),
            
            # Physical measurements
            'weight': round(random.uniform(50, 90), 1),
            'height': round(random.uniform(150, 185), 0),
            
            # Biomarkers
            'ki67_index': random.randint(10, 80),
            'pdl1_tps': random.randint(0, 50),
            'microsatellite_status': random.choice(['MSS', 'MSS', 'MSS', 'MSI-H']),  # MSS more common
            
            # Organ function
            'hepatic_function': random.choice(['Normal', 'Normal', 'Normal', 'Mild']),
            'egfr': round(random.uniform(60, 100), 1),
            
            # Genomic variants
            'genomic_variants': self._generate_variants(cancer_type, num_variants=random.randint(1, 4))
        }
        
        return patient
    
    def _generate_high_risk_patient(self, patient_id: str) -> dict:
        """고위험 환자 (High-Risk Aggressive)"""
        cancer_type = random.choice(['Colorectal', 'Lung', 'Pancreatic'])
        
        return {
            'patient_id': patient_id,
            'age': random.randint(55, 75),
            'gender': random.choice(['Male', 'Female']),
            'cancer_type': cancer_type,
            'stage': random.choice(['III', 'IV']),  # Advanced stage
            'grade': random.choice(['poor', 'poor', 'moderate']),  # Mostly poor
            'primary_site': self._get_primary_site(cancer_type),
            'ecog_score': random.randint(1, 2),
            'diagnosis_date': (datetime.now() - timedelta(days=random.randint(60, 180))).strftime('%Y-%m-%d'),
            
            'weight': round(random.uniform(55, 75), 1),
            'height': round(random.uniform(155, 180), 0),
            
            # High proliferation markers
            'ki67_index': random.randint(50, 90),  # High Ki-67
            'pdl1_tps': random.randint(0, 30),
            'microsatellite_status': 'MSS',
            
            'hepatic_function': 'Normal',
            'egfr': round(random.uniform(70, 95), 1),
            
            # Multiple pathogenic variants
            'genomic_variants': self._generate_high_risk_variants(cancer_type)
        }
    
    def _generate_targeted_therapy_patient(self, patient_id: str) -> dict:
        """표적치료 후보 환자"""
        cancer_type = random.choice(['Colorectal', 'Lung', 'Breast'])
        
        return {
            'patient_id': patient_id,
            'age': random.randint(50, 70),
            'gender': random.choice(['Male', 'Female']),
            'cancer_type': cancer_type,
            'stage': random.choice(['II', 'III']),
            'grade': 'moderate',
            'primary_site': self._get_primary_site(cancer_type),
            'ecog_score': random.randint(0, 1),
            'diagnosis_date': (datetime.now() - timedelta(days=random.randint(30, 120))).strftime('%Y-%m-%d'),
            
            'weight': round(random.uniform(60, 80), 1),
            'height': round(random.uniform(160, 180), 0),
            
            'ki67_index': random.randint(30, 60),
            'pdl1_tps': random.randint(10, 40),
            'microsatellite_status': random.choice(['MSS', 'MSI-H']),
            
            'hepatic_function': 'Normal',
            'egfr': round(random.uniform(80, 100), 1),
            
            # Actionable mutations
            'genomic_variants': self._generate_actionable_variants(cancer_type)
        }
    
    def _generate_low_risk_patient(self, patient_id: str) -> dict:
        """저위험 환자 (Low-Risk Indolent)"""
        cancer_type = random.choice(['Breast', 'Colorectal'])
        
        return {
            'patient_id': patient_id,
            'age': random.randint(45, 65),
            'gender': random.choice(['Male', 'Female']),
            'cancer_type': cancer_type,
            'stage': random.choice(['I', 'II']),  # Early stage
            'grade': random.choice(['well', 'well', 'moderate']),
            'primary_site': self._get_primary_site(cancer_type),
            'ecog_score': 0,
            'diagnosis_date': (datetime.now() - timedelta(days=random.randint(14, 60))).strftime('%Y-%m-%d'),
            
            'weight': round(random.uniform(60, 85), 1),
            'height': round(random.uniform(160, 180), 0),
            
            'ki67_index': random.randint(5, 25),  # Low Ki-67
            'pdl1_tps': random.randint(0, 10),
            'microsatellite_status': 'MSS',
            
            'hepatic_function': 'Normal',
            'egfr': round(random.uniform(85, 105), 1),
            
            # Few or benign variants
            'genomic_variants': self._generate_low_risk_variants()
        }
    
    def _get_primary_site(self, cancer_type: str) -> str:
        """암종별 발생 부위"""
        sites = {
            'Colorectal': ['Sigmoid colon', 'Rectum', 'Ascending colon', 'Cecum'],
            'Lung': ['Right upper lobe', 'Left upper lobe', 'Right lower lobe'],
            'Breast': ['Upper outer quadrant', 'Upper inner quadrant', 'Lower outer quadrant'],
            'Pancreatic': ['Head', 'Body', 'Tail'],
            'Gastric': ['Antrum', 'Body', 'Cardia']
        }
        return random.choice(sites.get(cancer_type, ['Unknown']))
    
    def _generate_variants(self, cancer_type: str, num_variants: int) -> list:
        """일반 유전자 변이 생성"""
        variants = []
        
        # Cancer-specific common mutations
        cancer_genes = {
            'Colorectal': ['KRAS', 'BRAF', 'TP53', 'APC', 'PIK3CA'],
            'Lung': ['EGFR', 'ALK', 'KRAS', 'TP53', 'ROS1'],
            'Breast': ['HER2', 'PIK3CA', 'TP53', 'BRCA1', 'BRCA2'],
            'Pancreatic': ['KRAS', 'TP53', 'CDKN2A', 'SMAD4'],
            'Gastric': ['TP53', 'PIK3CA', 'ARID1A', 'CDH1']
        }
        
        available_genes = cancer_genes.get(cancer_type, self.common_genes)
        selected_genes = random.sample(available_genes, min(num_variants, len(available_genes)))
        
        for gene in selected_genes:
            variants.append({
                'gene_name': gene,
                'variant_type': random.choice(self.variant_types),
                'variant_detail': self._get_variant_detail(gene),
                'allele_frequency': round(random.uniform(0.3, 0.7), 2),
                'pathogenicity': random.choice(self.pathogenicity_levels),
                'test_date': datetime.now().strftime('%Y-%m-%d')
            })
        
        return variants
    
    def _generate_high_risk_variants(self, cancer_type: str) -> list:
        """고위험 변이 생성"""
        variants = []
        
        # TP53 (almost always in high-risk)
        variants.append({
            'gene_name': 'TP53',
            'variant_type': 'SNV',
            'variant_detail': 'p.R273H',
            'allele_frequency': 0.65,
            'pathogenicity': 'Pathogenic',
            'test_date': datetime.now().strftime('%Y-%m-%d')
        })
        
        # Add 2-3 more pathogenic variants
        if cancer_type == 'Colorectal':
            variants.append({
                'gene_name': 'KRAS',
                'variant_type': 'SNV',
                'variant_detail': 'p.G12D',
                'allele_frequency': 0.52,
                'pathogenicity': 'Pathogenic',
                'test_date': datetime.now().strftime('%Y-%m-%d')
            })
        
        return variants
    
    def _generate_actionable_variants(self, cancer_type: str) -> list:
        """표적치료 가능 변이 생성"""
        variants = []
        
        actionable_mutations = {
            'Colorectal': [
                {'gene_name': 'BRAF', 'variant_detail': 'p.V600E', 'drug': 'Encorafenib'},
                {'gene_name': 'HER2', 'variant_detail': 'Amplification', 'drug': 'Trastuzumab'}
            ],
            'Lung': [
                {'gene_name': 'EGFR', 'variant_detail': 'p.L858R', 'drug': 'Osimertinib'},
                {'gene_name': 'ALK', 'variant_detail': 'EML4-ALK fusion', 'drug': 'Alectinib'}
            ],
            'Breast': [
                {'gene_name': 'HER2', 'variant_detail': 'Amplification', 'drug': 'Trastuzumab'},
                {'gene_name': 'PIK3CA', 'variant_detail': 'p.H1047R', 'drug': 'Alpelisib'}
            ]
        }
        
        mutations = actionable_mutations.get(cancer_type, [])
        if mutations:
            mutation = random.choice(mutations)
            variants.append({
                'gene_name': mutation['gene_name'],
                'variant_type': 'SNV' if 'p.' in mutation['variant_detail'] else 'CNV',
                'variant_detail': mutation['variant_detail'],
                'allele_frequency': round(random.uniform(0.4, 0.8), 2),
                'pathogenicity': 'Pathogenic',
                'test_date': datetime.now().strftime('%Y-%m-%d')
            })
        
        return variants
    
    def _generate_low_risk_variants(self) -> list:
        """저위험 변이 생성"""
        # Few variants, mostly VUS or benign
        num = random.randint(0, 2)
        if num == 0:
            return []
        
        variants = []
        for _ in range(num):
            variants.append({
                'gene_name': random.choice(['TP53', 'APC', 'PIK3CA']),
                'variant_type': 'SNV',
                'variant_detail': 'p.A123T',
                'allele_frequency': round(random.uniform(0.35, 0.55), 2),
                'pathogenicity': random.choice(['VUS', 'Benign', 'Likely benign']),
                'test_date': datetime.now().strftime('%Y-%m-%d')
            })
        
        return variants
    
    def _get_variant_detail(self, gene: str) -> str:
        """유전자별 대표 변이"""
        common_variants = {
            'KRAS': random.choice(['p.G12D', 'p.G12V', 'p.G13D']),
            'BRAF': 'p.V600E',
            'EGFR': random.choice(['p.L858R', 'p.T790M', 'Exon 19 del']),
            'TP53': random.choice(['p.R175H', 'p.R273H', 'p.R248W']),
            'PIK3CA': random.choice(['p.H1047R', 'p.E545K']),
            'HER2': 'Amplification'
        }
        
        return common_variants.get(gene, 'p.X123Y')
    
    def generate_quantitative_data(self, scenario: str = 'random') -> dict:
        """정량 분석 데이터 생성"""
        if scenario == 'high_risk':
            return {
                'num_cells': random.randint(600, 1200),
                'mean_area': round(random.uniform(200, 300), 1),
                'cv_area': round(random.uniform(0.4, 0.6), 2),
                'overall_heterogeneity': round(random.uniform(0.7, 0.9), 2),
                'heterogeneity_grade': 'High',
                'clustered_ratio': round(random.uniform(0.6, 0.8), 2),
                'num_clusters': random.randint(10, 20),
                'clark_evans_index': round(random.uniform(0.5, 0.7), 2),
                'density_variance': round(random.uniform(0.5, 0.7), 2),
                'shape_diversity': round(random.uniform(3.0, 4.5), 1)
            }
        elif scenario == 'low_risk':
            return {
                'num_cells': random.randint(300, 600),
                'mean_area': round(random.uniform(180, 220), 1),
                'cv_area': round(random.uniform(0.15, 0.3), 2),
                'overall_heterogeneity': round(random.uniform(0.2, 0.4), 2),
                'heterogeneity_grade': 'Low',
                'clustered_ratio': round(random.uniform(0.2, 0.4), 2),
                'num_clusters': random.randint(3, 8),
                'clark_evans_index': round(random.uniform(0.9, 1.2), 2),
                'density_variance': round(random.uniform(0.2, 0.4), 2),
                'shape_diversity': round(random.uniform(1.5, 2.5), 1)
            }
        else:
            return {
                'num_cells': random.randint(400, 900),
                'mean_area': round(random.uniform(190, 270), 1),
                'cv_area': round(random.uniform(0.25, 0.5), 2),
                'overall_heterogeneity': round(random.uniform(0.4, 0.7), 2),
                'heterogeneity_grade': random.choice(['Moderate', 'High']),
                'clustered_ratio': round(random.uniform(0.4, 0.7), 2),
                'num_clusters': random.randint(6, 15),
                'clark_evans_index': round(random.uniform(0.7, 0.9), 2),
                'density_variance': round(random.uniform(0.3, 0.6), 2),
                'shape_diversity': round(random.uniform(2.0, 3.5), 1)
            }
    
    def save_sample_dataset(self, output_dir: str = 'data/samples', num_patients: int = 10):
        """샘플 데이터셋 저장"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        scenarios = ['high_risk'] * 3 + ['targeted'] * 3 + ['low_risk'] * 2 + ['random'] * 2
        
        samples = []
        for i in range(num_patients):
            scenario = scenarios[i] if i < len(scenarios) else 'random'
            patient_id = f"PT-TEST-{1000+i}"
            
            patient = self.generate_patient(patient_id, scenario)
            quant_data = self.generate_quantitative_data(scenario)
            
            sample = {
                'patient': patient,
                'quantitative_analysis': quant_data,
                'scenario': scenario
            }
            
            samples.append(sample)
            
            # Save individual file
            with open(output_path / f"{patient_id}.json", 'w', encoding='utf-8') as f:
                json.dump(sample, f, indent=2, ensure_ascii=False)
        
        # Save combined file
        with open(output_path / 'all_samples.json', 'w', encoding='utf-8') as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)
        
        print(f"[OK] {num_patients}개 샘플 데이터 생성 완료: {output_path}")
        return samples


if __name__ == "__main__":
    generator = SampleDataGenerator()
    generator.save_sample_dataset(num_patients=10)
