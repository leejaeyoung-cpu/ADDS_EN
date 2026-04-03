"""
Show Document Processing Page
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime

# Add src to path  
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from preprocessing.document_parser import DocumentParser
from ui.app_core import get_document_parser
from utils.ai_analyzer import generate_comprehensive_insights


def show_document_processing():
    """Enhanced Document processing with AI analysis"""
    st.header("📄 문서 처리 및 AI 분석")
    st.markdown("PDF/DOCX 문서 파싱, AI 분석, 데이터 저장")
    
    # Import modules
    try:
        from preprocessing.enhanced_document_parser import EnhancedDocumentParser
        from utils.document_analyzer import DocumentAnalyzer
        from utils.document_db import DocumentDatabase
    except ImportError as e:
        st.error(f"모듈 import 오류: {e}")
        st.info("enhanced_document_parser, document_analyzer, document_db 모듈이 필요합니다.")
        return
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📄 문서 파싱",
        "🤖 AI 분석",
        "📊 분석 보고서",
        "🎯 인터랙티브 뷰",
        "💾 데이터 저장",
        "📚 문서 라이브러리"
    ])
    
    # ===== TAB 1: Document Parsing =====
    with tab1:
        st.markdown("### 📄 문서 업로드 및 파싱")
        
        uploaded_file = st.file_uploader(
            "PDF 또는 DOCX 파일 업로드",
            type=['pdf', 'docx', 'txt'],
            help="논문, 임상 보고서, 연구 문서 등"
        )
        
        if uploaded_file:
            st.success(f"✅ {uploaded_file.name} 업로드됨 ({uploaded_file.size:,} bytes)")
            
            if st.button("🔍 문서 파싱 시작", type="primary"):
                with st.spinner("문서를 분석하고 있습니다..."):
                    try:
                        # Save temporary file
                        temp_path = Path(f"temp_{uploaded_file.name}")
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getvalue())
                        
                        # Parse document
                        parser = EnhancedDocumentParser()
                        parsed_doc = parser.parse_document(str(temp_path))
                        
                        # Store in session state
                        st.session_state['parsed_doc'] = parsed_doc
                        st.session_state['doc_filename'] = uploaded_file.name
                        
                        st.success("✅ 파싱 완료!")
                        
                        # Display results
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("문서 유형", parsed_doc.get('file_type', 'Unknown'))
                        with col2:
                            word_count = len(parsed_doc.get('full_text', '').split())
                            st.metric("단어 수", f"{word_count:,}")
                        with col3:
                            st.metric("섹션 수", len(parsed_doc.get('sections', [])))
                        
                        # Metadata
                        if parsed_doc.get('metadata'):
                            with st.expander("📋 메타데이터"):
                                st.json(parsed_doc['metadata'])
                        
                        # Sections
                        if parsed_doc.get('sections'):
                            with st.expander(f"📑 섹션 ({len(parsed_doc['sections'])}개)"):
                                for section in parsed_doc['sections'][:5]:  # Show first 5
                                    st.markdown(f"**{section.get('title', 'Untitled')}**")
                                    st.caption(section.get('content', '')[:200] + "...")
                        
                        # Tables
                        if parsed_doc.get('tables'):
                            with st.expander(f"📊 표 ({len(parsed_doc['tables'])}개)"):
                                for idx, table in enumerate(parsed_doc['tables'][:3]):  # Show first 3
                                    st.markdown(f"**표 {idx+1}**")
                                    if table.get('data'):
                                        try:
                                            df = pd.DataFrame(table['data'][1:], columns=table['data'][0])
                                            st.dataframe(df)
                                        except:
                                            pass
                        
                        # Entities
                        if parsed_doc.get('entities'):
                            with st.expander("🧬 추출된 엔티티"):
                                entities = parsed_doc['entities']
                                if entities.get('drugs'):
                                    st.markdown(f"**약물**: {', '.join(entities['drugs'][:10])}")
                                if entities.get('genes'):
                                    st.markdown(f"**유전자**: {', '.join(entities['genes'][:10])}")
                                if entities.get('dosages'):
                                    st.markdown(f"**용량**: {', '.join(entities['dosages'][:5])}")
                        
                        # Cleanup
                        temp_path.unlink()
                        
                    except Exception as e:
                        st.error(f"파싱 오류: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
    
    # ===== TAB 2: AI Analysis =====
    with tab2:
        st.markdown("### 🤖 OpenAI 기반 문서 분석")
        
        if 'parsed_doc' not in st.session_state:
            st.warning("⚠️ 먼저 '문서 파싱' 탭에서 문서를 업로드하고 파싱하세요.")
        else:
            parsed_doc = st.session_state['parsed_doc']
            
            st.info(f"분석 대상: **{st.session_state.get('doc_filename', 'Unknown')}**")
            
            analysis_type = st.selectbox(
                "분석 유형 선택",
                ["comprehensive", "summary", "entities", "relationships"],
                format_func=lambda x: {
                    "comprehensive": "종합 분석 (요약 + 엔티티 + 관계)",
                    "summary": "요약만",
                    "entities": "엔티티 추출",
                    "relationships": "관계 추출"
                }[x]
            )
            
            if st.button("🧠 AI 분석 시작", type="primary"):
                with st.spinner("OpenAI GPT-4로 분석 중... (30-60초 소요)"):
                    try:
                        analyzer = DocumentAnalyzer()
                        
                        analysis = analyzer.analyze_document(
                            parsed_doc,
                            analysis_type=analysis_type
                        )
                        
                        # Store in session state
                        st.session_state['analysis'] = analysis
                        
                        if analysis.get('success'):
                            st.success("✅ AI 분석 완료!")
                            
                            # Display based on type
                            if analysis_type == "comprehensive":
                                st.markdown("### 📊 종합 분석 보고서")
                                
                                analysis_data = analysis.get('analysis', {})
                                
                                # Check if it's structured data (has expected fields)
                                is_structured = isinstance(analysis_data, dict) and (
                                    'executive_summary' in analysis_data or 
                                    'key_findings' in analysis_data or
                                    'drugs_mentioned' in analysis_data
                                )
                                
                                # Check if markdown format
                                if analysis_data.get('format') == 'markdown' or 'raw_analysis' in analysis_data:
                                    st.markdown("#### 📋 분석 결과")
                                    st.markdown(analysis_data.get('raw_analysis', str(analysis_data)))
                                
                                # Handle structured JSON data
                                elif is_structured:
                                    # Executive Summary
                                    if 'executive_summary' in analysis_data:
                                        st.markdown("#### 📋 Executive Summary")
                                        st.info(analysis_data['executive_summary'])
                                    
                                    # Key Findings
                                    if 'key_findings' in analysis_data:
                                        st.markdown("#### 🔍 주요 발견사항")
                                        findings = analysis_data['key_findings']
                                        if isinstance(findings, list):
                                            for idx, finding in enumerate(findings, 1):
                                                st.markdown(f"{idx}. {finding}")
                                        else:
                                            st.write(findings)
                                    
                                    # Drugs
                                    if 'drugs_mentioned' in analysis_data:
                                        st.markdown("#### 💊 언급된 약물")
                                        drugs = analysis_data['drugs_mentioned']
                                        if isinstance(drugs, list):
                                            for drug in drugs:
                                                if isinstance(drug, dict):
                                                    st.markdown(f"**{drug.get('name', 'Unknown')}**")
                                                    st.markdown(f"- 용도: {drug.get('purpose', '')}")
                                                    st.markdown("")
                                                else:
                                                    st.markdown(f"- {drug}")
                                        else:
                                            st.write(drugs)
                                    
                                    # Clinical Data
                                    if 'clinical_data' in analysis_data:
                                        st.markdown("#### 🏥 임상 데이터")
                                        clinical = analysis_data['clinical_data']
                                        if isinstance(clinical, dict):
                                            # Display in a more readable format
                                            for key, value in clinical.items():
                                                st.markdown(f"**{key.replace('_', ' ').title()}**:")
                                                if isinstance(value, dict):
                                                    for k, v in value.items():
                                                        st.markdown(f"  - {k.replace('_', ' ').title()}: {v}")
                                                else:
                                                    st.markdown(f"  {value}")
                                        else:
                                            st.write(clinical)
                                    
                                    # Mechanisms
                                    if 'mechanisms' in analysis_data:
                                        st.markdown("#### 🧬 생물학적 메커니즘")
                                        mechanisms = analysis_data['mechanisms']
                                        if isinstance(mechanisms, list):
                                            for mech in mechanisms:
                                                st.markdown(f"- {mech}")
                                        else:
                                            st.write(mechanisms)
                                    
                                    # Implications
                                    if 'implications' in analysis_data:
                                        st.markdown("#### 💡 임상적 의의")
                                        implications = analysis_data['implications']
                                        if isinstance(implications, list):
                                            for impl in implications:
                                                st.markdown(f"- {impl}")
                                        else:
                                            st.write(implications)
                                
                                else:
                                    # Fallback: show as markdown
                                    st.markdown(str(analysis_data))
                                
                            elif analysis_type == "summary":
                                st.markdown("### 📄 문서 요약")
                                st.markdown(analysis.get('summary', ''))
                            
                            elif analysis_type == "entities":
                                st.markdown("### 🧬 추출된 엔티티")
                                entities = analysis.get('entities', {})
                                st.json(entities)
                            
                            elif analysis_type == "relationships":
                                st.markdown("### 🔗 관계 추출")
                                relationships = analysis.get('relationships', {})
                                st.json(relationships)
                            
                            # Show token usage
                            if 'tokens_used' in analysis:
                                st.caption(f"🔢 사용된 토큰: {analysis['tokens_used']}")
                        
                        else:
                            st.error(f"분석 실패: {analysis.get('error', 'Unknown error')}")
                            if 'fallback' in analysis:
                                st.markdown("### 기본 분석 (API 없음)")
                                st.json(analysis['fallback'])
                    
                    except Exception as e:
                        st.error(f"분석 오류: {str(e)}")
    
    # ===== TAB 3: Analysis Report =====
    with tab3:
        st.markdown("### 📊 자동 분석 보고서 생성")
        
        if 'parsed_doc' not in st.session_state:
            st.warning("⚠️ 먼저 문서를 파싱하세요.")
        elif 'analysis' not in st.session_state:
            st.warning("⚠️ AI 분석을 먼저 실행하세요.")
        else:
            parsed_doc = st.session_state['parsed_doc']
            analysis = st.session_state['analysis']
            
            st.info(f"보고서 생성: **{st.session_state.get('doc_filename', 'Unknown')}**")
            
            # Report generation options
            col1, col2 = st.columns(2)
            with col1:
                report_format = st.selectbox(
                    "보고서 형식",
                    ["Markdown", "HTML"],
                    index=0
                )
            
            if st.button("📊 보고서 생성", type="primary"):
                with st.spinner("보고서를 생성하고 있습니다..."):
                    try:
                        from utils.report_generator import ReportGenerator
                        
                        generator = ReportGenerator()
                        
                        if report_format == "Markdown":
                            report_content = generator.generate_markdown_report(
                                parsed_doc,
                                analysis
                            )
                            
                            # Store in session state
                            st.session_state['report'] = report_content
                            st.session_state['report_format'] = 'markdown'
                            
                            st.success("✅ 마크다운 보고서 생성 완료!")
                            
                            # Display report
                            st.markdown("---")
                            st.markdown(report_content)
                            st.markdown("---")
                            
                            # Download button
                            filename = f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
                            st.download_button(
                                label="📥 마크다운 다운로드",
                                data=report_content,
                                file_name=filename,
                                mime="text/markdown"
                            )
                        
                        elif report_format == "HTML":
                            report_content = generator.generate_html_report(
                                parsed_doc,
                                analysis
                            )
                            
                            st.session_state['report'] = report_content
                            st.session_state['report_format'] = 'html'
                            
                            st.success("✅ HTML 보고서 생성 완료!")
                            
                            # Download button
                            filename = f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                            st.download_button(
                                label="📥 HTML 다운로드",
                                data=report_content,
                                file_name=filename,
                                mime="text/html"
                            )
                            
                            # Preview
                            with st.expander("미리보기"):
                                st.components.v1.html(report_content, height=600, scrolling=True)
                    
                    except Exception as e:
                        st.error(f"보고서 생성 오류: {str(e)}")
                        import traceback
                        st.code(traceback.format_exc())
            
            # Show existing report if available
            if 'report' in st.session_state:
                st.markdown("---")
                st.markdown("### 💾 마지막 생성된 보고서")
                
                if st.session_state.get('report_format') == 'markdown':
                    with st.expander("보고서 보기", expanded=False):
                        st.markdown(st.session_state['report'])
                    
                    filename = f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
                    st.download_button(
                        label="📥 다시 다운로드",
                        data=st.session_state['report'],
                        file_name=filename,
                        mime="text/markdown"
                    )
    
    # ===== TAB 4: Data Storage =====
    with tab4:
        st.markdown("### 💾 데이터베이스 저장")
        
        if 'parsed_doc' not in st.session_state:
            st.warning("⚠️ 먼저 문서를 파싱하세요.")
        else:
            parsed_doc = st.session_state['parsed_doc']
            analysis = st.session_state.get('analysis', None)
            
            st.info(f"저장할 문서: **{st.session_state.get('doc_filename', 'Unknown')}**")
            
            if analysis:
                st.success("✅ AI 분석 결과 포함")
            else:
                st.warning("⚠️ AI 분석 없음 (파싱 결과만 저장)")
            
            if st.button("💾 데이터베이스에 저장", type="primary"):
                with st.spinner("저장 중..."):
                    try:
                        db = DocumentDatabase()
                        
                        doc_id = db.save_document(
                            parsed_doc,
                            file_path=st.session_state.get('doc_filename'),
                            analysis=analysis
                        )
                        
                        st.success(f"✅ 저장 완료! (문서 ID: {doc_id})")
                        
                        # Generate fine-tuning data if analysis exists
                        if analysis:
                            st.markdown("#### 🤖 파인튜닝 데이터 생성")
                            
                            analyzer = DocumentAnalyzer()
                            training_data = analyzer.generate_fine_tuning_data(
                                parsed_doc,
                                analysis
                            )
                            
                            db.save_fine_tuning_data(doc_id, training_data)
                            
                            st.success("✅ 파인튜닝 데이터 생성 완료")
                            
                            with st.expander("파인튜닝 데이터 미리보기"):
                                st.json(training_data)
                    
                    except Exception as e:
                        st.error(f"저장 오류: {str(e)}")
    
    # ===== TAB 5: Document Library =====`r`n    with tab5:
        st.markdown("### 📚 문서 라이브러리")
        
        try:
            db = DocumentDatabase()
            stats = db.get_statistics()
            
            # Statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("총 문서", stats['total_documents'])
            with col2:
                st.metric("분석됨", stats['analyzed_documents'])
            with col3:
                st.metric("총 단어", f"{stats['total_words']:,}")
            with col4:
                st.metric("섹션", stats['total_sections'])
            
            # Search
            search_query = st.text_input("🔍 문서 검색", placeholder="제목, 내용 검색...")
            
            if search_query:
                documents = db.search_documents(search_query)
                st.info(f"검색 결과: {len(documents)}개 문서")
            else:
                documents = db.get_all_documents()
            
            # Display documents
            if documents:
                for doc in documents[:10]:  # Show first 10
                    with st.expander(f"📄 {doc['filename']} ({doc['upload_date'][:10]})"):
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.markdown(f"**제목**: {doc.get('title', 'N/A')}")
                            st.markdown(f"**저자**: {doc.get('author', 'N/A')}")
                            st.markdown(f"**단어 수**: {doc.get('word_count', 0):,}")
                        with col_b:
                            st.markdown(f"**유형**: {doc.get('file_type', 'N/A')}")
                            st.markdown(f"**페이지**: {doc.get('page_count', 0)}")
                            st.markdown(f"**분석됨**: {'✅' if doc.get('analyzed') else '❌'}")
                        
                        if doc.get('summary'):
                            st.markdown("**요약**:")
                            st.caption(doc['summary'][:300] + "...")
                        
                        if st.button(f"🗑️ 삭제", key=f"del_{doc['id']}"):
                            db.delete_document(doc['id'])
                            st.rerun()
            else:
                st.info("저장된 문서가 없습니다.")
        
        except Exception as e:
            st.error(f"데이터베이스 오류: {str(e)}")
