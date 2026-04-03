"""
ESMFold Client - Protein Structure Prediction
Uses ESMFold (Meta AI) for local protein structure prediction

ESMFold provides AlphaFold-level accuracy without requiring external API calls.
Model runs locally on GPU, ensuring HIPAA compliance.

Reference:
- ESMFold paper: https://www.science.org/doi/10.1126/science.ade2574
- Hugging Face: facebook/esmfold_v1
"""

import os
import logging
import pickle
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

# Optional imports with fallback
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. ESMFold client will run in mock mode.")

try:
    from transformers import EsmForProteinFolding, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logger.warning("Transformers not available. ESMFold client will run in mock mode.")


@dataclass
class ProteinStructure:
    """단백질 구조 데이터 클래스
    
    Attributes:
        protein_id: 단백질 식별자 (e.g., "BCR-ABL", "EGFR")
        sequence: 아미노산 서열
        coordinates: 3D 좌표 (N, 3) - N개 원자의 xyz 좌표
        confidence_scores: pLDDT 스코어 (0-100) - 높을수록 신뢰도 높음
        mean_plddt: 평균 pLDDT 스코어
        structure_available: 구조 예측 성공 여부
    """
    protein_id: str
    sequence: str
    coordinates: Optional[np.ndarray] = None
    confidence_scores: Optional[np.ndarray] = None
    mean_plddt: Optional[float] = None
    structure_available: bool = False
    
    def is_high_confidence(self, threshold: float = 70.0) -> bool:
        """고신뢰도 구조인지 확인
        
        의료 안전성: pLDDT > 70은 일반적으로 신뢰할 수 있는 구조로 간주
        """
        if self.mean_plddt is None:
            return False
        return self.mean_plddt > threshold


class ESMFoldClient:
    """ESMFold 기반 단백질 구조 예측 클라이언트
    
    Example:
        >>> client = ESMFoldClient()
        >>> structure = client.predict_structure("EGFR", sequence="MRPSG...")
        >>> if structure.is_high_confidence():
        >>>     print(f"High quality structure: {structure.mean_plddt}")
    """
    
    def __init__(
        self,
        cache_dir: str = "data/protein_structures",
        device: Optional[str] = None,
        use_mock: bool = False
    ):
        """
        Args:
            cache_dir: 구조 캐시 디렉토리
            device: PyTorch device ('cuda' or 'cpu')
            use_mock: True이면 실제 모델 없이 mock 데이터 반환
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_mock = use_mock or not (TORCH_AVAILABLE and TRANSFORMERS_AVAILABLE)
        
        if self.use_mock:
            logger.warning("ESMFold running in MOCK mode (dependencies not available)")
            self.model = None
            self.tokenizer = None
        else:
            # Device 설정
            if device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = device
            
            logger.info(f"Initializing ESMFold on device: {self.device}")
            
            try:
                # ESMFold 모델 로드 (최초 실행 시 ~15GB 다운로드)
                self.tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
                self.model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
                self.model = self.model.to(self.device)
                self.model.eval()
                logger.info("ESMFold model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load ESMFold model: {e}")
                logger.warning("Falling back to MOCK mode")
                self.use_mock = True
                self.model = None
                self.tokenizer = None
    
    def predict_structure(
        self,
        protein_id: str,
        sequence: Optional[str] = None,
        use_cache: bool = True
    ) -> ProteinStructure:
        """단백질 구조 예측
        
        Args:
            protein_id: 단백질 이름 (e.g., "EGFR", "BCR-ABL")
            sequence: 아미노산 서열 (없으면 UniProt에서 자동 검색 시도)
            use_cache: 캐시된 구조가 있으면 사용
            
        Returns:
            ProteinStructure 객체
            
        의료 안전성:
            - 구조 예측 실패 시 structure_available=False 반환
            - pLDDT < 70인 경우 경고 로그 출력
        """
        # 캐시 확인
        if use_cache:
            cached = self._load_from_cache(protein_id)
            if cached is not None:
                logger.info(f"Loaded {protein_id} from cache (pLDDT: {cached.mean_plddt:.1f})")
                return cached
        
        # 서열 확인
        if sequence is None:
            logger.error(f"No sequence provided for {protein_id}")
            return ProteinStructure(
                protein_id=protein_id,
                sequence="",
                structure_available=False
            )
        
        # Mock 모드
        if self.use_mock:
            return self._mock_prediction(protein_id, sequence)
        
        # 실제 예측
        try:
            logger.info(f"Predicting structure for {protein_id} (length: {len(sequence)})")
            
            with torch.no_grad():
                # Tokenize
                inputs = self.tokenizer([sequence], return_tensors="pt", add_special_tokens=False)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Predict
                outputs = self.model(**inputs)
                
                # Extract coordinates and confidence
                coords = outputs.positions.cpu().numpy()[0]  # (L, 37, 3)
                plddt = outputs.plddt.cpu().numpy()[0]  # (L,)
                
                # CA (alpha carbon) coordinates만 사용
                ca_coords = coords[:, 1, :]  # (L, 3)
                
                mean_plddt = float(np.mean(plddt))
                
                structure = ProteinStructure(
                    protein_id=protein_id,
                    sequence=sequence,
                    coordinates=ca_coords,
                    confidence_scores=plddt,
                    mean_plddt=mean_plddt,
                    structure_available=True
                )
                
                # 신뢰도 검증
                if mean_plddt < 70:
                    logger.warning(
                        f"Low confidence structure for {protein_id}: "
                        f"pLDDT={mean_plddt:.1f} (recommend >70)"
                    )
                else:
                    logger.info(f"High confidence structure: pLDDT={mean_plddt:.1f}")
                
                # 캐시 저장
                self._save_to_cache(structure)
                
                return structure
                
        except Exception as e:
            logger.error(f"Structure prediction failed for {protein_id}: {e}")
            return ProteinStructure(
                protein_id=protein_id,
                sequence=sequence,
                structure_available=False
            )
    
    def batch_predict(
        self,
        protein_list: List[Tuple[str, str]],
        max_batch_size: int = 4
    ) -> List[ProteinStructure]:
        """배치 구조 예측
        
        Args:
            protein_list: [(protein_id, sequence), ...] 리스트
            max_batch_size: 최대 배치 크기 (GPU 메모리 고려)
            
        Returns:
            ProteinStructure 리스트
        """
        results = []
        for i in range(0, len(protein_list), max_batch_size):
            batch = protein_list[i:i+max_batch_size]
            for protein_id, sequence in batch:
                structure = self.predict_structure(protein_id, sequence)
                results.append(structure)
        return results
    
    def _load_from_cache(self, protein_id: str) -> Optional[ProteinStructure]:
        """캐시에서 구조 로드"""
        cache_file = self.cache_dir / f"{protein_id}.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache for {protein_id}: {e}")
        return None
    
    def _save_to_cache(self, structure: ProteinStructure):
        """구조를 캐시에 저장"""
        cache_file = self.cache_dir / f"{structure.protein_id}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(structure, f)
            logger.debug(f"Saved {structure.protein_id} to cache")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def _mock_prediction(self, protein_id: str, sequence: str) -> ProteinStructure:
        """Mock 구조 생성 (테스트용) — pLDDT=0 to indicate unreliable structure"""
        np.random.seed(hash(protein_id) % (2**32))
        
        seq_len = len(sequence)
        
        # 랜덤 3D 좌표 생성 (placeholder geometry)
        coords = np.random.randn(seq_len, 3) * 10
        
        # pLDDT = 0.0 for ALL residues (mock → no confidence)
        plddt = np.zeros(seq_len)
        
        return ProteinStructure(
            protein_id=protein_id,
            sequence=sequence,
            coordinates=coords,
            confidence_scores=plddt,
            mean_plddt=0.0,
            structure_available=False  # Mock structure is NOT reliable
        )


# 알려진 암 관련 단백질 서열 (짧은 버전 - 테스트용)
KNOWN_PROTEIN_SEQUENCES = {
    "EGFR": "MRPSGTAGAALLALLAALCPASRALEEKKVCQGTSNKLTQLGTFEDHFLSLQRMFNNCEVVLGNLEITYVQRNYDLSFLKTIQEVAGYVLIALNTVERIPLENLQIIRGNMYYENSYALAVLSNYDANKTGLKELPMRNLQEILHGAVRFSNNPALCNVESIQWRDIVSSDFLSNMSMDFQNHLGSCQKCDPSCPNGSCWGAGEENCQKLTKIICAQQCSGRCRGKSPSDCCHNQCAAGCTGPRESDCLVCRKFRDEATCKDTCPPLMLYNPTTYQMDVNPEGKYSFGATCVKKCPRNYVVTDHGSCVRACGADSYEMEEDGVRKCKKCEGPCRKVCNGIGIGEFKDSLSINATNIKHFKNCTSISGDLHILPVAFRGDSFTHTPPLDPQELDILKTVKEITGFLLIQAWPENRTDLHAFENLEIIRGRTKQHGQFSLAVVSLNITSLGLRSLKEISDGDVIISGNKNLCYANTINWKKLFGTSGQKTKIISNRGENSCKATGQVCHALCSPEGCWGPEPRDCVSCRNVSRGRECVDKCNLLEGEPREFVENSECIQCHPECLPQAMNITCTGRGPDNCIQCAHYIDGPHCVKTCPAGVMGENNTLVWKYADAGHVCHLCHPNCTYGCTGPGLEGCPTNGPKIPSE"
}


if __name__ == "__main__":
    # 테스트 실행
    logging.basicConfig(level=logging.INFO)
    
    client = ESMFoldClient(use_mock=True)
    
    # EGFR 구조 예측
    structure = client.predict_structure(
        protein_id="EGFR",
        sequence=KNOWN_PROTEIN_SEQUENCES["EGFR"][:100]  # 짧은 버전으로 테스트
    )
    
    print(f"Protein: {structure.protein_id}")
    print(f"Sequence length: {len(structure.sequence)}")
    print(f"Structure available: {structure.structure_available}")
    print(f"Mean pLDDT: {structure.mean_plddt:.1f}")
    print(f"High confidence: {structure.is_high_confidence()}")
