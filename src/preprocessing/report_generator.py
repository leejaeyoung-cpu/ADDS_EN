"""
Analysis Report Generator for ADDS
상세한 분석 리포트를 PDF 형식으로 자동 생성
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # GUI 없이 사용
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import io
import logging

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT

logger = logging.getLogger(__name__)

try:
    from utils import get_logger
except ImportError:
    from src.utils import get_logger

logger = get_logger(__name__)


class AnalysisReportGenerator:
    """
    이미지 분석 결과를 상세한 PDF 리포트로 생성
    """
    
    def __init__(self):
        """Initialize report generator"""
        self.styles = getSampleStyleSheet()
        self._create_custom_styles()
        logger.info("AnalysisReportGenerator initialized")
    
    def _create_custom_styles(self):
        """커스텀 스타일 생성"""
        # 제목 스타일
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#667eea'),
            spaceAfter=30,
            alignment=TA_CENTER
        ))
        
        # 섹션 헤더 스타일
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#764ba2'),
            spaceBefore=20,
            spaceAfter=10
        ))
    
    def _create_chart(
        self,
        data: Dict[str, Any],
        chart_type: str = 'histogram'
    ) -> io.BytesIO:
        """
        차트 생성
        
        Args:
            data: 차트 데이터
            chart_type: 차트 유형
            
        Returns:
            이미지 버퍼
        """
        fig, ax = plt.subplots(figsize=(6, 4))
        
        if chart_type == 'bar':
            keys = list(data.keys())
            values = list(data.values())
            ax.bar(keys, values, color='#667eea')
            ax.set_ylabel('Value')
        
        plt.tight_layout()
        
        # 버퍼에 저장
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        return buf
    
    def generate_report(
        self,
        output_path: Path,
        image_path: Path,
        metadata: Dict[str, Any],
        quality_assessment: Dict[str, Any],
        analysis_results: Dict[str, Any],
        visualization_paths: Optional[Dict[str, str]] = None,
        **kwargs  # For comparison_results, hyperparameter_recommendations, etc.
    ):
        """
        완전한 분석 리포트 생성
        
        Args:
            output_path: 리포트 저장 경로
            image_path: 원본 이미지 경로
            metadata: 메타데이터
            quality_assessment: 품질 평가 결과
            analysis_results: 분석 결과
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        doc = SimpleDocTemplate(
            str(output_path),
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18
        )
        
        story = []
        
        # 제목
        title = Paragraph(
            "Image Analysis Report<br/><font size=12>ADDS - AI Anticancer Drug Discovery System</font>",
            self.styles['CustomTitle']
        )
        story.append(title)
        story.append(Spacer(1, 12))
        
        # 기본 정보
        story.append(Paragraph("1. Overview", self.styles['SectionHeader']))
        
        basic_info = metadata.get('basic_info', {})
        overview_data = [
            ['File Name', basic_info.get('filename', 'N/A')],
            ['File Size', f"{basic_info.get('file_size_mb', 0)} MB"],
            ['Image Size', f"{basic_info.get('image_shape', {}).get('width', 0)} x {basic_info.get('image_shape', {}).get('height', 0)}"],
            ['Analyzed', datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
        ]
        
        overview_table = Table(overview_data, colWidths=[2*inch, 4*inch])
        overview_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f0f0f0')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
        ]))
        story.append(overview_table)
        story.append(Spacer(1, 20))
        
        # 품질 평가
        story.append(Paragraph("2. Quality Assessment", self.styles['SectionHeader']))
        
        overall_quality = quality_assessment.get('overall_quality', 'N/A')
        overall_score = quality_assessment.get('overall_score', 0)
        
       # 종합 품질
        quality_summary = Paragraph(
            f"<b>Overall Quality:</b> {overall_quality} (Score: {overall_score:.2f}/1.0)",
            self.styles['Normal']
        )
        story.append(quality_summary)
        story.append(Spacer(1, 12))
        
        # 상세 평가
        detailed = quality_assessment.get('detailed_assessment', {})
        quality_data = [['Metric', 'Value', 'Quality']]
        
        for metric, values in detailed.items():
            metric_name = metric.replace('_', ' ').title()
            if 'score' in values:
                value_str = f"{values['score']:.2f}"
            elif 'value' in values:
                value_str = f"{values['value']:.2f}"
            elif 'snr' in values:
                value_str = f"{values['snr']:.2f}"
            else:
                value_str = "N/A"
            
            quality_data.append([metric_name, value_str, values.get('quality', 'N/A')])
        
        quality_table = Table(quality_data, colWidths=[2*inch, 1.5*inch, 2*inch])
        quality_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        story.append(quality_table)
        story.append(Spacer(1, 12))
        
        # 권장사항
        recommendations = quality_assessment.get('recommendations', [])
        if recommendations:
            story.append(Paragraph("<b>Recommendations:</b>", self.styles['Normal']))
            for rec in recommendations:
                story.append(Paragraph(f"• {rec}", self.styles['Normal']))
            story.append(Spacer(1, 20))
        
        # 분석 결과
        story.append(Paragraph("3. Quantitative Analysis", self.styles['SectionHeader']))
        
        if 'num_cells' in analysis_results:
            cell_data = [
                ['Total Cells', str(analysis_results.get('num_cells', 'N/A'))],
                ['Cell Density', f"{analysis_results.get('cell_density', 'N/A'):.1f} cells/mm²" if isinstance(analysis_results.get('cell_density'), (int, float)) else 'N/A'],
                ['Mean Cell Area', f"{analysis_results.get('mean_cell_area', 'N/A'):.2f} μm²" if isinstance(analysis_results.get('mean_cell_area'), (int, float)) else 'N/A'],
                ['Viability Score', f"{analysis_results.get('viability_score', 'N/A'):.2f}" if isinstance(analysis_results.get('viability_score'), (int, float)) else 'N/A']
            ]
            
            cell_table = Table(cell_data, colWidths=[2.5*inch, 3*inch])
            cell_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f0f0f0')),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
            ]))
            story.append(cell_table)
        
        story.append(Spacer(1, 20))
        
        # 통계 정보
        story.append(Paragraph("4. Statistical Information", self.styles['SectionHeader']))
        
        intensity_stats = metadata.get('intensity_statistics', {})
        stats_data = [
            ['Mean Intensity', f"{intensity_stats.get('mean_intensity', 0):.2f}"],
            ['Std Intensity', f"{intensity_stats.get('std_intensity', 0):.2f}"],
            ['Min/Max', f"{intensity_stats.get('min_intensity', 0)} / {intensity_stats.get('max_intensity', 0)}"],
            ['Dynamic Range', str(intensity_stats.get('dynamic_range', 0))]
        ]
        
        stats_table = Table(stats_data, colWidths=[2.5*inch, 3*inch])
        stats_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f0f0f0')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
        ]))
        story.append(stats_table)
        
        # Visualization Results Section
        if visualization_paths:
            story.append(PageBreak())  # Start new page
            story.append(Paragraph("5. Visualization Results", self.styles['SectionHeader']))
            story.append(Spacer(1, 12))
            
            # Image configurations (order and captions)
            images_config = [
                ('original_path', '① Original Image'),
                ('preprocessed_path', '② Preprocessed (CLAHE)'),
                ('colored_mask_path', '③ Segmentation Mask'),
                ('overlay_path', '④ Overlay'),
                ('contour_path', '⑤ Contours'),
                ('heatmap_path', '⑥ Size Heatmap')
            ]
            
            # Image dimensions
            img_width = 2.3 * inch
            img_height = 2.3 * inch
            
            # Create 2x3 grid
            for row in range(2):
                row_data = []
                for col in range(3):
                    idx = row * 3 + col
                    if idx < len(images_config):
                        key, caption = images_config[idx]
                        img_path = visualization_paths.get(key)
                        
                        if img_path and Path(img_path).exists():
                            try:
                                # Use original image directly, let reportlab handle sizing
                                # reportlab will automatically resize to fit the specified dimensions
                                img = RLImage(str(img_path), width=img_width, height=img_height)
                                caption_para = Paragraph(
                                    f"<font size=9>{caption}</font>",
                                    ParagraphStyle(name='ImageCaption', alignment=TA_CENTER, fontSize=9)
                                )
                                row_data.append([img, caption_para])
                                
                            except Exception as e:
                                logger.error(f"Failed to load image {img_path}: {e}")
                                # Fallback placeholder
                                placeholder = Paragraph(
                                    f"<font size=9>{caption}<br/>Load Error</font>",
                                    ParagraphStyle(name='ErrorCaption', alignment=TA_CENTER, fontSize=9)
                                )
                                row_data.append([placeholder])
                        else:
                            # Placeholder if image not available
                            placeholder = Paragraph(
                                f"<font size=9>{caption}<br/>Not Available</font>",
                                ParagraphStyle(name='PlaceholderCaption', alignment=TA_CENTER, fontSize=9)
                            )
                            row_data.append([placeholder])
                    else:
                        row_data.append([''])
                
                # Create table for this row (3 columns, each containing [image, caption])
                if row_data:
                    img_table = Table(row_data, colWidths=[2.5*inch, 2.5*inch, 2.5*inch])
                    img_table.setStyle(TableStyle([
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                        ('LEFTPADDING', (0, 0), (-1, -1), 5),
                        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
                    ]))
                    story.append(img_table)
                    story.append(Spacer(1, 10))
        
        # ========== AI Platform Comparison Section ==========
        comparison_results = kwargs.get('comparison_results')
        hyperparameter_recs = kwargs.get('hyperparameter_recommendations')
        
        if comparison_results and comparison_results.get('comparison_available'):
            story.append(PageBreak())  # Start new page
            story.append(Paragraph("6. AI Platform Comparison", self.styles['SectionHeader']))
            story.append(Spacer(1, 12))
            
            # Comparison summary
            agreement_score = comparison_results.get('agreement_score', 0)
            agreement_level = comparison_results.get('agreement_level', 'N/A')
            
            summary_text = f"""
            <b>Agreement Level:</b> {agreement_level}<br/>
            <b>Agreement Score:</b> {agreement_score*100:.1f}%<br/>
            <b>GPT-4 Vision Confidence:</b> {comparison_results.get('gpt4v_confidence', 0)*100:.0f}%
            """
            story.append(Paragraph(summary_text, self.styles['Normal']))
            story.append(Spacer(1, 12))
            
            # Cell count comparison table
            cell_comp = comparison_results.get('cell_count_comparison', {})
            comp_data = [
                ['Platform', 'Cell Count', 'Difference'],
                ['Cellpose', str(cell_comp.get('cellpose', 0)), '-'],
                ['GPT-4 Vision', str(cell_comp.get('gpt4v', 0)), 
                 f"{cell_comp.get('difference', 0)} cells ({cell_comp.get('difference_percent', 0):.1f}%)"],
            ]
            
            comp_table = Table(comp_data, colWidths=[2*inch, 2*inch, 2.5*inch])
            comp_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#667eea')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 11),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(comp_table)
            story.append(Spacer(1, 20))
            
            # ========== Side-by-Side Analysis Comparison ==========
            story.append(Paragraph("Detailed Analysis Comparison", self.styles['SectionHeader']))
            story.append(Spacer(1, 12))
            
            # Create side-by-side comparison table
            analysis_comp_data = [
                ['ADDS Cellpose Analysis', 'OpenAI GPT-4 Vision Analysis']
            ]
            
            # ADDS/Cellpose analysis summary
            adds_analysis = f"""
Cell Count: {analysis_results.get('num_cells', 0)} cells

Mean Cell Area: {analysis_results.get('mean_cell_area', 0):.2f} μm²

Cell Density: {analysis_results.get('cell_density', 0):.1f} cells/mm²

Viability Score: {analysis_results.get('viability_score', 0):.2f}

Quality Score: {quality_assessment.get('overall_score', 0):.2f}

Quality Level: {quality_assessment.get('overall_quality', 'N/A')}
"""
            
            # OpenAI analysis summary
            gpt4v_analysis = f"""
Cell Count Estimate: {cell_comp.get('gpt4v', 0)} cells

{comparison_results.get('gpt4v_characteristics', 'N/A')}

Quality Assessment: {comparison_results.get('gpt4v_quality', 'N/A')}

Confidence: {comparison_results.get('gpt4v_confidence', 0)*100:.0f}%
"""
            
            if comparison_results.get('gpt4v_notes'):
                gpt4v_analysis += f"\n\nAdditional Notes:\n{comparison_results.get('gpt4v_notes', '')}"
            
            analysis_comp_data.append([
                Paragraph(adds_analysis.replace('\n', '<br/>'), self.styles['Normal']),
                Paragraph(gpt4v_analysis.replace('\n', '<br/>'), self.styles['Normal'])
            ])
            
            analysis_comp_table = Table(analysis_comp_data, colWidths=[3.5*inch, 3.5*inch])
            analysis_comp_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#764ba2')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 11),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('LEFTPADDING', (0, 1), (-1, -1), 10),
                ('RIGHTPADDING', (0, 1), (-1, -1), 10),
                ('TOPPADDING', (0, 1), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 1), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('BACKGROUND', (0, 1), (0, -1), colors.HexColor('#f0f0ff')),
                ('BACKGROUND', (1, 1), (1, -1), colors.HexColor('#fff0f0'))
            ]))
            story.append(analysis_comp_table)
            story.append(Spacer(1, 20))
            
            # Full GPT-4V Raw Response (if available)
            if comparison_results.get('gpt4v_raw'):
                story.append(Paragraph("<b>Complete GPT-4 Vision Report:</b>", self.styles['Normal']))
                story.append(Spacer(1, 8))
                
                # Create a box for the full report
                raw_report = comparison_results.get('gpt4v_raw', '').replace('\n', '<br/>')
                report_para = Paragraph(
                    f"<font size=9><i>{raw_report}</i></font>",
                    self.styles['Normal']
                )
                
                report_table = Table([[report_para]], colWidths=[6.5*inch])
                report_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#fffef0')),
                    ('LEFTPADDING', (0, 0), (-1, -1), 12),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 12),
                    ('TOPPADDING', (0, 0), (-1, -1), 12),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                    ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#cccccc'))
                ]))
                story.append(report_table)
                story.append(Spacer(1, 20))
            
            # Discrepancies
            discrepancies = comparison_results.get('discrepancies', [])
            if discrepancies:
                story.append(Paragraph("<b>Discrepancies Detected:</b>", self.styles['Normal']))

                for disc in discrepancies:
                    story.append(Paragraph(f"• {disc}", self.styles['Normal']))
                story.append(Spacer(1, 12))
            
            # Hyperparameter recommendations
            if hyperparameter_recs and hyperparameter_recs.get('recommendations'):
                story.append(Spacer(1, 10))
                story.append(Paragraph("Hyperparameter Optimization Recommendations", self.styles['SectionHeader']))
                story.append(Spacer(1, 10))
                
                # Overall assessment
                assessment = hyperparameter_recs.get('overall_assessment', '')
                if assessment:
                    story.append(Paragraph(f"<b>Assessment:</b> {assessment}", self.styles['Normal']))
                    story.append(Spacer(1, 12))
                
                # Recommendations table
                rec_data = [['Parameter', 'Current', 'Suggested', 'Priority']]
                
                for rec in hyperparameter_recs.get('recommendations', []):
                    rec_data.append([
                        rec.get('parameter', ''),
                        str(rec.get('current_value', '')),
                        str(rec.get('suggested_value', '')),
                        rec.get('priority', 'medium')
                    ])
                
                rec_table = Table(rec_data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch, 1.5*inch])
                rec_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#764ba2')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                story.append(rec_table)
                story.append(Spacer(1, 12))
                
                # Reasons for each recommendation
                for rec in hyperparameter_recs.get('recommendations', []):
                    reason_text = f"<b>{rec.get('parameter')}:</b> {rec.get('reason', '')}"
                    story.append(Paragraph(reason_text, self.styles['Normal']))
                    story.append(Spacer(1, 8))
        
        # Footer
        story.append(Spacer(1, 40))
        footer = Paragraph(
            "<font size=8>Generated by ADDS - AI Anticancer Drug Discovery System<br/>"
            "Inha University Hospital, Department of Biomedical Engineering</font>",
            ParagraphStyle(name='Footer', alignment=TA_CENTER, fontSize=8, textColor=colors.grey)
        )
        story.append(footer)
        
        # PDF 생성
        doc.build(story)
        logger.info(f"Analysis report generated: {output_path}")
