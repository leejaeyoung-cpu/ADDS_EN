"""
Preprocessing Options UI for CT Image Analysis
Provides interactive interface for selecting preprocessing techniques
"""

import streamlit as st
import numpy as np
from typing import Dict, Any, List, Optional
from pathlib import Path
import json


class PreprocessingOptionsUI:
    """
    Interactive UI for selecting CT image preprocessing options.
    
    Allows users to choose from multiple preprocessing techniques:
    - Intensity Windowing (HU value ranges)
    - Normalization methods
    - Resampling strategies
    - Contrast enhancement
    - Artifact removal
    """
    
    PREPROCESSING_OPTIONS = {
        "intensity_windowing": {
            "name": "강도 윈도우잉 (Intensity Windowing)",
            "description": "CT 영상의 HU 값 범위를 조정하여 특정 조직을 강조합니다.",
            "options": {
                "soft_tissue": {
                    "name": "연조직 (Soft Tissue)",
                    "window_center": 40,
                    "window_width": 400,
                    "hu_range": (-175, 250),
                    "use_case": "종양, 내장기관 분석"
                },
                "bone": {
                    "name": "뼈 (Bone)",
                    "window_center": 300,
                    "window_width": 1500,
                    "hu_range": (-450, 1050),
                    "use_case": "골격 구조 분석"
                },
                "lung": {
                    "name": "폐 (Lung)",
                    "window_center": -500,
                    "window_width": 1500,
                    "hu_range": (-1000, -300),
                    "use_case": "폐 질환 분석"
                },
                "liver": {
                    "name": "간 (Liver)",
                    "window_center": 50,
                    "window_width": 400,
                    "hu_range": (-150, 250),
                    "use_case": "간 전이 검출"
                },
                "custom": {
                    "name": "사용자 정의 (Custom)",
                    "window_center": 0,
                    "window_width": 400,
                    "hu_range": (-200, 200),
                    "use_case": "특정 조직 맞춤 설정"
                }
            }
        },
        "normalization": {
            "name": "정규화 (Normalization)",
            "description": "픽셀 강도 값을 표준화하여 학습 안정성을 높입니다.",
            "options": {
                "z_score": {
                    "name": "Z-Score 정규화",
                    "method": "zscore",
                    "description": "평균 0, 표준편차 1로 정규화",
                    "formula": "(x - μ) / σ"
                },
                "min_max": {
                    "name": "Min-Max (0-1) 정규화",
                    "method": "minmax",
                    "description": "값을 0~1 범위로 스케일링",
                    "formula": "(x - min) / (max - min)"
                },
                "percentile": {
                    "name": "백분위수 기반 정규화",
                    "method": "percentile",
                    "description": "1-99 백분위수 기준 정규화",
                    "formula": "clip(x, p1, p99) then normalize"
                },
                "none": {
                    "name": "정규화 없음",
                    "method": "none",
                    "description": "원본 값 유지",
                    "formula": "x"
                }
            }
        },
        "resampling": {
            "name": "리샘플링 (Resampling)",
            "description": "복셀 간격을 변경하여 해상도를 조정합니다.",
            "options": {
                "isotropic_1mm": {
                    "name": "등방성 1mm³ 리샘플링",
                    "spacing": (1.0, 1.0, 1.0),
                    "method": "linear",
                    "description": "모든 축을 1mm 간격으로 통일",
                    "use_case": "표준 품질"
                },
                "downsample_2mm": {
                    "name": "2mm 다운샘플링 (빠른 처리)",
                    "spacing": (2.0, 2.0, 2.0),
                    "method": "linear",
                    "description": "처리 속도 향상",
                    "use_case": "프로토타입, 빠른 분석"
                },
                "upsample_0.5mm": {
                    "name": "0.5mm 업샘플링 (고해상도)",
                    "spacing": (0.5, 0.5, 0.5),
                    "method": "cubic",
                    "description": "미세 구조 분석",
                    "use_case": "정밀 검출"
                },
                "none": {
                    "name": "리샘플링 없음",
                    "spacing": None,
                    "method": None,
                    "description": "원본 해상도 유지",
                    "use_case": "DICOM 원본"
                }
            }
        },
        "enhancement": {
            "name": "대비 향상 (Contrast Enhancement)",
            "description": "영상의 대비를 향상시켜 특징을 강조합니다.",
            "options": {
                "clahe": {
                    "name": "CLAHE (Adaptive Histogram Equalization)",
                    "method": "clahe",
                    "clip_limit": 2.0,
                    "tile_size": (8, 8, 8),
                    "description": "적응형 히스토그램 평활화"
                },
                "unsharp_mask": {
                    "name": "언샤프 마스크 (Unsharp Mask)",
                    "method": "unsharp",
                    "radius": 1.0,
                    "amount": 1.0,
                    "description": "엣지 강조"
                },
                "gaussian_filter": {
                    "name": "가우시안 필터 (Noise Reduction)",
                    "method": "gaussian",
                    "sigma": 0.5,
                    "description": "노이즈 감소"
                },
                "none": {
                    "name": "향상 없음",
                    "method": "none",
                    "description": "원본 대비 유지"
                }
            }
        },
        "artifact_removal": {
            "name": "아티팩트 제거 (Artifact Removal)",
            "description": "금속 아티팩트, 모션 블러 등을 제거합니다.",
            "options": {
                "metal_artifact_reduction": {
                    "name": "금속 아티팩트 감소 (MAR)",
                    "method": "mar",
                    "threshold": 3000,
                    "description": "금속 임플란트로 인한 아티팩트 감소"
                },
                "motion_correction": {
                    "name": "모션 보정",
                    "method": "motion",
                    "description": "호흡/움직임으로 인한 블러 감소"
                },
                "noise_reduction": {
                    "name": "노이즈 감소 (NLM)",
                    "method": "nlm",
                    "h": 10,
                    "description": "Non-Local Means 노이즈 제거"
                },
                "none": {
                    "name": "제거 없음",
                    "method": "none",
                    "description": "아티팩트 제거 건너뛰기"
                }
            }
        }
    }
    
    def __init__(self):
        """Initialize Preprocessing Options UI"""
        self.selected_options = {}
    
    def render_ui(self) -> Dict[str, Any]:
        """
        Render Streamlit UI for preprocessing options selection.
        
        Returns:
            Dictionary with selected preprocessing configurations
        """
        st.title("🔧 CT 영상 전처리 옵션 선택")
        st.markdown("""
        **Medical Decathlon 및 CTdata 처리를 위한 전처리 기법을 선택하세요.**
        
        각 옵션은 SOTA 논문에서 검증된 방법을 기반으로 합니다.
        """)
        
        selected_config = {}
        
        # Create tabs for each preprocessing category
        tabs = st.tabs([
            "🎚️ 강도 윈도우잉",
            "📊 정규화",
            "🔍 리샘플링",
            "✨ 대비 향상",
            "🛠️ 아티팩트 제거"
        ])
        
        # Tab 1: Intensity Windowing
        with tabs[0]:
            st.header("강도 윈도우잉 (Intensity Windowing)")
            st.info(self.PREPROCESSING_OPTIONS["intensity_windowing"]["description"])
            
            options = self.PREPROCESSING_OPTIONS["intensity_windowing"]["options"]
            option_names = {k: v["name"] for k, v in options.items()}
            
            selected = st.selectbox(
                "윈도우 타입 선택:",
                list(option_names.keys()),
                format_func=lambda x: option_names[x],
                key="windowing"
            )
            
            config = options[selected]
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Window Center", config["window_center"])
                st.metric("Window Width", config["window_width"])
            with col2:
                st.metric("HU Range", f"{config['hu_range'][0]} ~ {config['hu_range'][1]}")
                st.info(f"**적용 대상**: {config['use_case']}")
            
            if selected == "custom":
                st.markdown("#### 사용자 정의 설정")
                custom_center = st.slider("Window Center", -1000, 1000, 0)
                custom_width = st.slider("Window Width", 100, 2000, 400)
                config["window_center"] = custom_center
                config["window_width"] = custom_width
                config["hu_range"] = (
                    custom_center - custom_width // 2,
                    custom_center + custom_width // 2
                )
            
            selected_config["intensity_windowing"] = {
                "type": selected,
                "config": config
            }
        
        # Tab 2: Normalization
        with tabs[1]:
            st.header("정규화 (Normalization)")
            st.info(self.PREPROCESSING_OPTIONS["normalization"]["description"])
            
            options = self.PREPROCESSING_OPTIONS["normalization"]["options"]
            option_names = {k: v["name"] for k, v in options.items()}
            
            selected = st.selectbox(
                "정규화 방법 선택:",
                list(option_names.keys()),
                format_func=lambda x: option_names[x],
                key="normalization"
            )
            
            config = options[selected]
            st.code(config["formula"], language="python")
            st.write(config["description"])
            
            selected_config["normalization"] = {
                "type": selected,
                "config": config
            }
        
        # Tab 3: Resampling
        with tabs[2]:
            st.header("리샘플링 (Resampling)")
            st.info(self.PREPROCESSING_OPTIONS["resampling"]["description"])
            
            options = self.PREPROCESSING_OPTIONS["resampling"]["options"]
            option_names = {k: v["name"] for k, v in options.items()}
            
            selected = st.selectbox(
                "리샘플링 전략 선택:",
                list(option_names.keys()),
                format_func=lambda x: option_names[x],
                key="resampling"
            )
            
            config = options[selected]
            if config["spacing"]:
                st.metric("Voxel Spacing (mm)", f"{config['spacing'][0]} x {config['spacing'][1]} x {config['spacing'][2]}")
                st.metric("Interpolation Method", config["method"])
            st.info(f"**적용 대상**: {config['use_case']}")
            
            selected_config["resampling"] = {
                "type": selected,
                "config": config
            }
        
        # Tab 4: Contrast Enhancement
        with tabs[3]:
            st.header("대비 향상 (Contrast Enhancement)")
            st.info(self.PREPROCESSING_OPTIONS["enhancement"]["description"])
            
            options = self.PREPROCESSING_OPTIONS["enhancement"]["options"]
            option_names = {k: v["name"] for k, v in options.items()}
            
            selected = st.selectbox(
                "향상 기법 선택:",
                list(option_names.keys()),
                format_func=lambda x: option_names[x],
                key="enhancement"
            )
            
            config = options[selected]
            st.write(config["description"])
            
            selected_config["enhancement"] = {
                "type": selected,
                "config": config
            }
        
        # Tab 5: Artifact Removal
        with tabs[4]:
            st.header("아티팩트 제거 (Artifact Removal)")
            st.info(self.PREPROCESSING_OPTIONS["artifact_removal"]["description"])
            
            options = self.PREPROCESSING_OPTIONS["artifact_removal"]["options"]
            option_names = {k: v["name"] for k, v in options.items()}
            
            selected = st.selectbox(
                "제거 기법 선택:",
                list(option_names.keys()),
                format_func=lambda x: option_names[x],
                key="artifact"
            )
            
            config = options[selected]
            st.write(config["description"])
            
            selected_config["artifact_removal"] = {
                "type": selected,
                "config": config
            }
        
        # Summary and Export
        st.divider()
        st.header("📋 선택된 전처리 설정 요약")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.json(selected_config)
        with col2:
            if st.button("💾 설정 저장", use_container_width=True):
                self.save_config(selected_config)
                st.success("설정이 저장되었습니다!")
            
            if st.button("🔄 기본값 복원", use_container_width=True):
                st.rerun()
        
        self.selected_options = selected_config
        return selected_config
    
    def save_config(self, config: Dict[str, Any], filename: str = "preprocessing_config.json"):
        """
        Save preprocessing configuration to JSON file.
        
        Args:
            config: Preprocessing configuration dictionary
            filename: Output filename
        """
        config_dir = Path("configs/preprocessing")
        config_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = config_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        return filepath
    
    def load_config(self, filename: str = "preprocessing_config.json") -> Dict[str, Any]:
        """
        Load preprocessing configuration from JSON file.
        
        Args:
            filename: Configuration filename
            
        Returns:
            Preprocessing configuration dictionary
        """
        filepath = Path("configs/preprocessing") / filename
        if not filepath.exists():
            return {}
        
        with open(filepath, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        self.selected_options = config
        return config
    
    @staticmethod
    def get_recommended_config(analysis_type: str) -> Dict[str, Any]:
        """
        Get recommended preprocessing configuration for specific analysis type.
        
        Args:
            analysis_type: Type of analysis ('tumor_detection', 'organ_segmentation', 
                          'staging', 'prognosis')
        
        Returns:
            Recommended configuration dictionary
        """
        recommendations = {
            "tumor_detection": {
                "intensity_windowing": {"type": "soft_tissue"},
                "normalization": {"type": "z_score"},
                "resampling": {"type": "isotropic_1mm"},
                "enhancement": {"type": "clahe"},
                "artifact_removal": {"type": "noise_reduction"}
            },
            "organ_segmentation": {
                "intensity_windowing": {"type": "soft_tissue"},
                "normalization": {"type": "percentile"},
                "resampling": {"type": "isotropic_1mm"},
                "enhancement": {"type": "none"},
                "artifact_removal": {"type": "none"}
            },
            "staging": {
                "intensity_windowing": {"type": "soft_tissue"},
                "normalization": {"type": "z_score"},
                "resampling": {"type": "isotropic_1mm"},
                "enhancement": {"type": "clahe"},
                "artifact_removal": {"type": "noise_reduction"}
            },
            "prognosis": {
                "intensity_windowing": {"type": "soft_tissue"},
                "normalization": {"type": "percentile"},
                "resampling": {"type": "isotropic_1mm"},
                "enhancement": {"type": "unsharp_mask"},
                "artifact_removal": {"type": "noise_reduction"}
            }
        }
        
        return recommendations.get(analysis_type, recommendations["tumor_detection"])


def main():
    """Main function for standalone UI testing"""
    st.set_page_config(
        page_title="CT 전처리 옵션",
        page_icon="🔧",
        layout="wide"
    )
    
    ui = PreprocessingOptionsUI()
    selected_config = ui.render_ui()
    
    # Display recommended configs
    st.divider()
    st.header("💡 권장 설정")
    
    analysis_types = {
        "tumor_detection": "🔍 종양 검출",
        "organ_segmentation": "🫀 장기 분할",
        "staging": "📊 TNM 병기 분류",
        "prognosis": "📈 예후 예측"
    }
    
    selected_analysis = st.selectbox(
        "분석 유형별 권장 설정 보기:",
        list(analysis_types.keys()),
        format_func=lambda x: analysis_types[x]
    )
    
    recommended = PreprocessingOptionsUI.get_recommended_config(selected_analysis)
    st.json(recommended)


if __name__ == "__main__":
    main()
