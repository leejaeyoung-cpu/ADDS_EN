"""
Utility functions for parsing experimental metadata from filenames
"""

import re
from typing import Dict, Optional


def parse_filename_metadata(filename: str) -> Dict[str, Optional[str]]:
    """
    Parse experimental metadata from filename
    
    Supported patterns:
    - CellLine_Condition_Rep#.ext (e.g., HUVEC_Control_1.tif)
    - CellLine_Treatment_Condition_Rep#.ext (e.g., HUVEC_TNFa_6hr_2.jpg)
    - Treatment_Condition_Rep#.ext (e.g., TNFa_24hr_3.tif)
    
    Args:
        filename: Image filename
        
    Returns:
        Dictionary with parsed metadata
    """
    # Remove file extension
    name_without_ext = filename.rsplit('.', 1)[0]
    
    # Split by underscore or hyphen
    parts = re.split(r'[_-]', name_without_ext)
    
    metadata = {
        'cell_line': None,
        'treatment': None,
        'condition': None,
        'replicate_number': None
    }
    
    if not parts:
        return metadata
    
    # Filter out date patterns (YYMMDD: 260107, YYYYMMDD: 20260107)
    # Remove parts that look like dates before parsing other metadata
    filtered_parts = []
    for part in parts:
        # Check if it's a 6-digit or 8-digit number (likely a date)
        if re.match(r'^\d{6}$', part) or re.match(r'^\d{8}$', part):
            # Skip date-like patterns
            continue
        filtered_parts.append(part)
    
    parts = filtered_parts
    
    # Try to extract replicate number
    for i, part in enumerate(parts):
        # Look for patterns like "1", "2", "Rep1", "Rep2", etc.
        # But ignore large numbers (likely timestamps or IDs)
        rep_match = re.search(r'(?:Rep|rep|R)?(\d+)$', part)
        if rep_match:
            rep_num = int(rep_match.group(1))
            # Only accept replicate numbers in reasonable range (1-10)
            if 1 <= rep_num <= 10:
                metadata['replicate_number'] = rep_num
                parts.pop(i)
                break
    
    # Look for time conditions (6hr, 12hr, 24hr, 48hr, etc.)
    for i, part in enumerate(parts):
        time_match = re.search(r'(\d+)\s*h(?:r|ours?)?', part, re.IGNORECASE)
        if time_match:
            metadata['condition'] = f"{time_match.group(1)}hr"
            parts.pop(i)
            break
        elif part.lower() in ['control', 'ctrl', 'untreated', 'baseline']:
            metadata['condition'] = 'Control'
            parts.pop(i)
            break
    
    # Remaining parts: try to identify cell line and treatment
    if len(parts) >= 2:
        # Common cell line patterns
        cell_lines = ['HUVEC', 'HEK293', 'HeLa', 'MCF7', 'A549', 'Jurkat', 'U2OS']
        
        # Check if first part is a known cell line
        if parts[0].upper() in [cl.upper() for cl in cell_lines]:
            metadata['cell_line'] = parts[0]
            # Rest might be treatment
            if len(parts) > 1:
                metadata['treatment'] = parts[1]
        else:
            # First might be treatment, second might be cell line
            metadata['treatment'] = parts[0]
            if parts[1].upper() in [cl.upper() for cl in cell_lines]:
                metadata['cell_line'] = parts[1]
    elif len(parts) == 1:
        # Single part - assume it's cell line
        metadata['cell_line'] = parts[0]
    
    return metadata


def format_metadata_preview(metadata: Dict[str, Optional[str]]) -> str:
    """
    Format metadata as preview string
    
    Args:
        metadata: Parsed metadata dictionary
        
    Returns:
        Formatted preview string
    """
    parts = []
    
    if metadata.get('cell_line'):
        parts.append(f"세포주: {metadata['cell_line']}")
    if metadata.get('treatment'):
        parts.append(f"처리: {metadata['treatment']}")
    if metadata.get('condition'):
        parts.append(f"조건: {metadata['condition']}")
    if metadata.get('replicate_number'):
        parts.append(f"반복: #{metadata['replicate_number']}")
    
    return " | ".join(parts) if parts else "정보 없음"
