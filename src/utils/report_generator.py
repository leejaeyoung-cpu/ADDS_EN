"""
Automatic Report Generator for Document Analysis
Generates structured analysis reports from AI analysis results
"""

from datetime import datetime
from typing import Dict, Any
import json


class ReportGenerator:
    """Generate analysis reports in various formats"""
    
    def __init__(self):
        pass
    
    def generate_markdown_report(
        self,
        parsed_doc: Dict[str, Any],
        analysis: Dict[str, Any]
    ) -> str:
        """
        Generate comprehensive markdown report
        
        Args:
            parsed_doc: Parsed document data
            analysis: AI analysis results
            
        Returns:
            Markdown formatted report
        """
        metadata = parsed_doc.get('metadata', {})
        analysis_data = analysis.get('analysis', {})
        
        # Handle both JSON and markdown format
        if analysis_data.get('format') == 'markdown':
            # If already markdown, use raw analysis
            return self._generate_from_markdown(parsed_doc, analysis)
        else:
            # If JSON, structure it
            return self._generate_from_json(parsed_doc, analysis)
    
    def _generate_from_json(
        self,
        parsed_doc: Dict,
        analysis: Dict
    ) -> str:
        """Generate report from JSON structured data"""
        
        analysis_data = analysis.get('analysis', {})
        metadata = parsed_doc.get('metadata', {})
        
        report_parts = []
        
        # Header
        report_parts.append(f"# 📊 문서 분석 보고서\n")
        report_parts.append(f"**생성일**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        report_parts.append("---\n\n")
        
        # Document Information
        report_parts.append("## 📄 문서 정보\n\n")
        report_parts.append(f"- **파일명**: {parsed_doc.get('filename', 'Unknown')}\n")
        report_parts.append(f"- **유형**: {parsed_doc.get('file_type', 'Unknown')}\n")
        if metadata.get('title'):
            report_parts.append(f"- **제목**: {metadata['title']}\n")
        if metadata.get('author'):
            report_parts.append(f"- **저자**: {metadata['author']}\n")
        report_parts.append(f"- **단어 수**: {len(parsed_doc.get('full_text', '').split()):,}\n")
        report_parts.append("\n")
        
        # Executive Summary
        if analysis_data.get('executive_summary'):
            report_parts.append("## 📋 Executive Summary\n\n")
            report_parts.append(f"{analysis_data['executive_summary']}\n\n")
        
        # Key Findings
        if analysis_data.get('key_findings'):
            report_parts.append("## 🔍 주요 발견사항\n\n")
            findings = analysis_data['key_findings']
            if isinstance(findings, list):
                for idx, finding in enumerate(findings, 1):
                    report_parts.append(f"{idx}. {finding}\n")
            else:
                report_parts.append(f"{findings}\n")
            report_parts.append("\n")
        
        # Drugs Analysis
        if analysis_data.get('drugs_mentioned'):
            report_parts.append("## 💊 약물 분석\n\n")
            drugs = analysis_data['drugs_mentioned']
            if isinstance(drugs, list):
                for drug in drugs:
                    if isinstance(drug, dict):
                        report_parts.append(f"### {drug.get('name', 'Unknown')}\n")
                        report_parts.append(f"**용도**: {drug.get('purpose', 'N/A')}\n\n")
                    else:
                        report_parts.append(f"- {drug}\n")
            else:
                report_parts.append(f"{drugs}\n")
            report_parts.append("\n")
        
        # Mechanisms
        if analysis_data.get('mechanisms'):
            report_parts.append("## 🧬 생물학적 메커니즘\n\n")
            mechanisms = analysis_data['mechanisms']
            if isinstance(mechanisms, list):
                for mech in mechanisms:
                    report_parts.append(f"- {mech}\n")
            else:
                report_parts.append(f"{mechanisms}\n")
            report_parts.append("\n")
        
        # Clinical Data
        if analysis_data.get('clinical_data'):
            report_parts.append("## 🏥 임상 데이터\n\n")
            clinical = analysis_data['clinical_data']
            if isinstance(clinical, dict):
                report_parts.append("```json\n")
                report_parts.append(json.dumps(clinical, indent=2, ensure_ascii=False))
                report_parts.append("\n```\n\n")
            else:
                report_parts.append(f"{clinical}\n\n")
        
        # Implications
        if analysis_data.get('implications'):
            report_parts.append("## 💡 임상적 의의\n\n")
            implications = analysis_data['implications']
            if isinstance(implications, list):
                for impl in implications:
                    report_parts.append(f"- {impl}\n")
            else:
                report_parts.append(f"{implications}\n")
            report_parts.append("\n")
        
        # Footer
        report_parts.append("---\n\n")
        report_parts.append("*본 보고서는 AI 분석을 기반으로 자동 생성되었습니다.*\n")
        report_parts.append(f"*분석 모델: {analysis.get('model', 'Unknown')}*\n")
        
        return "".join(report_parts)
    
    def _generate_from_markdown(
        self,
        parsed_doc: Dict,
        analysis: Dict
    ) -> str:
        """Generate report from markdown formatted analysis"""
        
        analysis_data = analysis.get('analysis', {})
        metadata = parsed_doc.get('metadata', {})
        
        report_parts = []
        
        # Header
        report_parts.append(f"# 📊 문서 분석 보고서\n")
        report_parts.append(f"**생성일**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        report_parts.append("---\n\n")
        
        # Document Information
        report_parts.append("## 📄 문서 정보\n\n")
        report_parts.append(f"- **파일명**: {parsed_doc.get('filename', 'Unknown')}\n")
        report_parts.append(f"- **유형**: {parsed_doc.get('file_type', 'Unknown')}\n")
        if metadata.get('title'):
            report_parts.append(f"- **제목**: {metadata['title']}\n")
        if metadata.get('author'):
            report_parts.append(f"- **저자**: {metadata['author']}\n")
        report_parts.append("\n")
        
        # Analysis Content
        report_parts.append("## 🤖 AI 분석 결과\n\n")
        report_parts.append(analysis_data.get('raw_analysis', ''))
        report_parts.append("\n\n")
        
        # Footer
        report_parts.append("---\n\n")
        report_parts.append("*본 보고서는 AI 분석을 기반으로 자동 생성되었습니다.*\n")
        
        return "".join(report_parts)
    
    def generate_html_report(
        self,
        parsed_doc: Dict[str, Any],
        analysis: Dict[str, Any]
    ) -> str:
        """Generate HTML report (future enhancement)"""
        # Convert markdown to HTML
        markdown_report = self.generate_markdown_report(parsed_doc, analysis)
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>분석 보고서</title>
            <style>
                body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
                h1 {{ color: #2c3e50; }}
                h2 {{ color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 5px; }}
                code {{ background: #f4f4f4; padding: 2px 5px; }}
            </style>
        </head>
        <body>
            <pre>{markdown_report}</pre>
        </body>
        </html>
        """
        return html
