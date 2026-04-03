"""
Metadata Tracking Utility
Automatically adds author and timestamp information to all data operations
"""

import streamlit as st
from datetime import datetime
from typing import Dict, Any, Optional
import json


def get_current_author_info() -> Dict[str, str]:
    """
    Get current user information from session state
    
    Returns:
        Dictionary with author information
    """
    current_user = st.session_state.get('current_user', '이재영')
    user_info = st.session_state.get('user_info', {
        'role': '연구원',
        'department': '바이오메디컬사이언스'
    })
    
    return {
        'author_name': current_user,
        'author_role': user_info.get('role', '연구원'),
        'author_department': user_info.get('department', '바이오메디컬사이언스'),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'date': datetime.now().strftime('%Y-%m-%d'),
        'time': datetime.now().strftime('%H:%M:%S')
    }


def add_author_metadata(data: Dict[str, Any], operation_type: str = "general") -> Dict[str, Any]:
    """
    Add author metadata to existing data dictionary
    
    Args:
        data: Original data dictionary
        operation_type: Type of operation (e.g., 'image_analysis', 'training', 'patient_data')
        
    Returns:
        Updated dictionary with author metadata
    """
    author_info = get_current_author_info()
    
    # Add metadata section if not exists
    if 'metadata' not in data:
        data['metadata'] = {}
    
    # Add author information
    data['metadata']['author'] = author_info
    data['metadata']['operation_type'] = operation_type
    
    return data


def create_analysis_metadata(
    analysis_type: str,
    input_files: Optional[list] = None,
    parameters: Optional[Dict[str, Any]] = None,
    additional_info: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create comprehensive metadata for analysis operations
    
    Args:
        analysis_type: Type of analysis (e.g., 'cellpose', 'ct_detection', 'drug_cocktail')
        input_files: List of input file names
        parameters: Analysis parameters
        additional_info: Any additional information
        
    Returns:
        Complete metadata dictionary
    """
    author_info = get_current_author_info()
    
    metadata = {
        'analysis_type': analysis_type,
        'author': author_info,
        'input_files': input_files or [],
        'parameters': parameters or {},
        'created_at': datetime.now().isoformat(),
        'version': '1.0.0'
    }
    
    if additional_info:
        metadata['additional_info'] = additional_info
    
    return metadata


def create_training_metadata(
    model_name: str,
    dataset_info: Dict[str, Any],
    hyperparameters: Dict[str, Any],
    training_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Create metadata for deep learning training
    
    Args:
        model_name: Name of the model being trained
        dataset_info: Information about training dataset
        hyperparameters: Training hyperparameters
        training_config: Additional training configuration
        
    Returns:
        Training metadata dictionary
    """
    author_info = get_current_author_info()
    
    metadata = {
        'model_name': model_name,
        'author': author_info,
        'dataset': dataset_info,
        'hyperparameters': hyperparameters,
        'training_started': datetime.now().isoformat(),
        'status': 'initialized'
    }
    
    if training_config:
        metadata['config'] = training_config
    
    return metadata


def create_patient_metadata(
    patient_id: str,
    data_type: str = "registration"
) -> Dict[str, Any]:
    """
    Create metadata for patient data operations
    
    Args:
        patient_id: Patient identifier
        data_type: Type of patient data operation
        
    Returns:
        Patient metadata dictionary
    """
    author_info = get_current_author_info()
    
    metadata = {
        'patient_id': patient_id,
        'data_type': data_type,
        'recorded_by': author_info,
        'record_created': datetime.now().isoformat()
    }
    
    return metadata


def format_author_display(metadata: Dict[str, Any]) -> str:
    """
    Format author information for display
    
    Args:
        metadata: Metadata dictionary containing author info
        
    Returns:
        Formatted string for display
    """
    if 'author' not in metadata:
        return "작성자 정보 없음"
    
    author = metadata['author']
    return (
        f"👤 {author.get('author_name', 'Unknown')} "
        f"({author.get('author_role', 'Unknown')})\n"
        f"📚 {author.get('author_department', 'Unknown')}\n"
        f"📅 {author.get('timestamp', 'Unknown')}"
    )


def save_metadata_to_json(metadata: Dict[str, Any], filepath: str):
    """
    Save metadata to JSON file
    
    Args:
        metadata: Metadata dictionary
        filepath: Path to save JSON file
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def load_metadata_from_json(filepath: str) -> Dict[str, Any]:
    """
    Load metadata from JSON file
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Metadata dictionary
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def show_author_info_box():
    """
    Display current author information in Streamlit
    """
    author_info = get_current_author_info()
    
    st.info(
        f"**현재 작성자**\n\n"
        f"👤 {author_info['author_name']} ({author_info['author_role']})\n\n"
        f"📚 {author_info['author_department']}\n\n"
        f"📅 {author_info['timestamp']}"
    )


# Example usage functions
def example_image_analysis_metadata():
    """Example: Creating metadata for image analysis"""
    metadata = create_analysis_metadata(
        analysis_type='cellpose_segmentation',
        input_files=['patient_001_cells.tiff'],
        parameters={
            'model_type': 'cyto2',
            'diameter': 30,
            'flow_threshold': 0.4
        }
    )
    return metadata


def example_training_metadata():
    """Example: Creating metadata for training"""
    metadata = create_training_metadata(
        model_name='tumor_detection_model',
        dataset_info={
            'name': 'Medical Decathlon Task06',
            'num_samples': 420,
            'train_split': 0.8
        },
        hyperparameters={
            'learning_rate': 0.001,
            'batch_size': 4,
            'epochs': 100
        }
    )
    return metadata
