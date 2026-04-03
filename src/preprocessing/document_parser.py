"""
Document parsing utilities for ADDS
Handles PDFs, reports, and text extraction
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import PyPDF2
import pdfplumber
import pandas as pd
from datetime import datetime

from utils import get_logger

logger = get_logger(__name__)


class DocumentParser:
    """
    Parse scientific papers, reports, and analysis documents
    """
    
    def __init__(self):
        """Initialize document parser"""
        self.supported_formats = ['.pdf', '.txt']
        logger.info("✓ Document parser initialized")
    
    def parse_pdf(
        self,
        pdf_path: Union[str, Path],
        extract_tables: bool = True,
        extract_images: bool = False
    ) -> Dict[str, Any]:
        """
        Parse PDF document
        
        Args:
            pdf_path: Path to PDF file
            extract_tables: Whether to extract tables
            extract_images: Whether to extract images
        
        Returns:
            Dictionary with extracted content
        """
        pdf_path = Path(pdf_path)
        logger.info(f"Parsing PDF: {pdf_path.name}")
        
        result = {
            'file_path': str(pdf_path),
            'file_name': pdf_path.name,
            'text': '',
            'tables': [],
            'metadata': {},
            'num_pages': 0
        }
        
        try:
            # Extract text and metadata using PyPDF2
            with open(pdf_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                result['num_pages'] = len(pdf_reader.pages)
                
                # Get metadata
                if pdf_reader.metadata:
                    result['metadata'] = {
                        'title': pdf_reader.metadata.get('/Title', ''),
                        'author': pdf_reader.metadata.get('/Author', ''),
                        'subject': pdf_reader.metadata.get('/Subject', ''),
                        'creator': pdf_reader.metadata.get('/Creator', ''),
                    }
                
                # Extract text from all pages
                text_parts = []
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        text_parts.append(page.extract_text())
                    except Exception as e:
                        logger.warning(f"Error extracting text from page {page_num}: {e}")
                
                result['text'] = '\n\n'.join(text_parts)
            
            # Extract tables using pdfplumber
            if extract_tables:
                with pdfplumber.open(pdf_path) as pdf:
                    for page_num, page in enumerate(pdf.pages):
                        tables = page.extract_tables()
                        for table_idx, table in enumerate(tables):
                            if table:
                                df = pd.DataFrame(table[1:], columns=table[0])
                                result['tables'].append({
                                    'page': page_num + 1,
                                    'table_index': table_idx,
                                    'data': df
                                })
            
            logger.info(f"✓ Parsed {result['num_pages']} pages, {len(result['tables'])} tables")
            
        except Exception as e:
            logger.error(f"Error parsing PDF: {e}")
            result['error'] = str(e)
        
        return result
    
    def extract_drug_concentrations(
        self,
        text: str
    ) -> List[Dict[str, Any]]:
        """
        Extract drug concentrations from text
        
        Args:
            text: Input text
        
        Returns:
            List of extracted concentration information
        """
        results = []
        
        # Pattern for concentration (e.g., "10 μM", "5 nM", "0.1 mg/ml")
        concentration_pattern = r'(\d+\.?\d*)\s*(μM|uM|nM|mM|pM|mg/ml|μg/ml|ng/ml)'
        
        # Pattern for IC50 values
        ic50_pattern = r'IC50[:\s]*(\d+\.?\d*)\s*(μM|uM|nM|mM|pM)'
        
        # Find concentrations
        for match in re.finditer(concentration_pattern, text, re.IGNORECASE):
            value = float(match.group(1))
            unit = match.group(2)
            
            # Get surrounding context
            start = max(0, match.start() - 50)
            end = min(len(text), match.end() + 50)
            context = text[start:end].replace('\n', ' ')
            
            results.append({
                'type': 'concentration',
                'value': value,
                'unit': unit,
                'context': context
            })
        
        # Find IC50 values
        for match in re.finditer(ic50_pattern, text, re.IGNORECASE):
            value = float(match.group(1))
            unit = match.group(2)
            
            start = max(0, match.start() - 50)
            end = min(len(text), match.end() + 50)
            context = text[start:end].replace('\n', ' ')
            
            results.append({
                'type': 'IC50',
                'value': value,
                'unit': unit,
                'context': context
            })
        
        return results
    
    def extract_compound_names(
        self,
        text: str,
        known_compounds: Optional[List[str]] = None
    ) -> List[str]:
        """
        Extract compound/drug names from text
        
        Args:
            text: Input text
            known_compounds: List of known compound names to search for
        
        Returns:
            List of found compound names
        """
        found_compounds = []
        
        if known_compounds:
            # Search for known compounds (case-insensitive)
            for compound in known_compounds:
                pattern = re.compile(re.escape(compound), re.IGNORECASE)
                if pattern.search(text):
                    found_compounds.append(compound)
        
        # Common anticancer drugs pattern (you can expand this)
        common_drugs = [
            'doxorubicin', 'cisplatin', 'paclitaxel', 'carboplatin',
            'gemcitabine', 'fluorouracil', '5-FU', 'oxaliplatin',
            'irinotecan', 'docetaxel', 'etoposide', 'vincristine',
            'methotrexate', 'tamoxifen', 'trastuzumab', 'bevacizumab'
        ]
        
        for drug in common_drugs:
            pattern = re.compile(r'\b' + re.escape(drug) + r'\b', re.IGNORECASE)
            if pattern.search(text):
                if drug not in found_compounds:
                    found_compounds.append(drug)
        
        return found_compounds
    
    def parse_western_blot_report(
        self,
        pdf_path: Union[str, Path]
    ) -> Dict[str, Any]:
        """
        Parse Western blot analysis report (외주 분석 보고서)
        
        Args:
            pdf_path: Path to Western blot report PDF
        
        Returns:
            Structured data from report
        """
        result = self.parse_pdf(pdf_path, extract_tables=True)
        
        # Try to extract protein expression data
        protein_data = []
        
        for table_info in result.get('tables', []):
            df = table_info['data']
            
            # Look for columns that might contain protein names and expression levels
            # This is heuristic and may need customization based on actual report format
            for col in df.columns:
                if any(keyword in str(col).lower() for keyword in ['protein', 'target', 'marker']):
                    # Found potential protein column
                    for idx, row in df.iterrows():
                        protein_name = row[col]
                        
                        # Try to find expression level in other columns
                        expression_data = {
                            'protein_name': protein_name,
                            'page': table_info['page'],
                            'table_index': table_info['table_index']
                        }
                        
                        # Look for numerical values
                        for other_col in df.columns:
                            if other_col != col:
                                try:
                                    value = pd.to_numeric(row[other_col], errors='ignore')
                                    if isinstance(value, (int, float)):
                                        expression_data[other_col] = value
                                except:
                                    pass
                        
                        protein_data.append(expression_data)
        
        result['protein_expression'] = protein_data
        logger.info(f"✓ Extracted {len(protein_data)} protein expression entries")
        
        return result
    
    def parse_experiment_report(
        self,
        pdf_path: Union[str, Path]
    ) -> Dict[str, Any]:
        """
        Parse general experiment report
        
        Args:
            pdf_path: Path to experiment report
        
        Returns:
            Structured experiment data
        """
        result = self.parse_pdf(pdf_path, extract_tables=True)
        
        text = result['text']
        
        # Extract key information
        experiment_info = {
            'file_path': str(pdf_path),
            'file_name': Path(pdf_path).name,
            'compounds': self.extract_compound_names(text),
            'concentrations': self.extract_drug_concentrations(text),
            'tables': result['tables'],
            'full_text': text
        }
        
        # Try to extract date
        date_pattern = r'(\d{4}[-/]\d{2}[-/]\d{2}|\d{2}[-/]\d{2}[-/]\d{4})'
        date_matches = re.findall(date_pattern, text)
        if date_matches:
            experiment_info['dates_found'] = date_matches
        
        # Try to extract cell line information
        cell_line_pattern = r'\b([A-Z]{2,}\d+|MCF-7|HeLa|A549|HCT116|U251|SW480)\b'
        cell_lines = re.findall(cell_line_pattern, text)
        if cell_lines:
            experiment_info['cell_lines'] = list(set(cell_lines))
        
        return experiment_info
    
    def save_structured_data(
        self,
        data: Dict[str, Any],
        output_path: Union[str, Path],
        format: str = 'json'
    ):
        """
        Save extracted data to file
        
        Args:
            data: Structured data dictionary
            output_path: Output file path
            format: 'json' or 'csv'
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'json':
            import json
            
            # Convert DataFrames to dict for JSON serialization
            data_copy = data.copy()
            if 'tables' in data_copy:
                for table in data_copy['tables']:
                    if 'data' in table and isinstance(table['data'], pd.DataFrame):
                        table['data'] = table['data'].to_dict('records')
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data_copy, f, indent=2, ensure_ascii=False)
        
        elif format == 'csv':
            # For CSV, we'll save flattened data
            if 'tables' in data and data['tables']:
                # Save first table as CSV
                df = data['tables'][0]['data']
                if isinstance(df, pd.DataFrame):
                    df.to_csv(output_path, index=False)
        
        logger.info(f"✓ Saved structured data to {output_path}")


class ExperimentDataExtractor:
    """
    Extract and structure experimental data from various sources
    """
    
    def __init__(self):
        """Initialize extractor"""
        self.parser = DocumentParser()
    
    def extract_from_directory(
        self,
        directory: Union[str, Path],
        file_pattern: str = "*.pdf",
        output_dir: Optional[Union[str, Path]] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract data from all documents in a directory
        
        Args:
            directory: Source directory
            file_pattern: File pattern to match
            output_dir: Directory to save extracted data
        
        Returns:
            List of extracted data dictionaries
        """
        directory = Path(directory)
        results = []
        
        pdf_files = list(directory.glob(file_pattern))
        logger.info(f"Found {len(pdf_files)} files matching '{file_pattern}'")
        
        for pdf_file in pdf_files:
            logger.info(f"Processing: {pdf_file.name}")
            
            # Determine document type based on filename keywords
            filename_lower = pdf_file.name.lower()
            
            if 'western' in filename_lower or 'blot' in filename_lower:
                data = self.parser.parse_western_blot_report(pdf_file)
            else:
                data = self.parser.parse_experiment_report(pdf_file)
            
            results.append(data)
            
            # Save individual results if output directory specified
            if output_dir:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                output_file = output_dir / f"{pdf_file.stem}_extracted.json"
                self.parser.save_structured_data(data, output_file)
        
        logger.info(f"✓ Processed {len(results)} documents")
        return results
