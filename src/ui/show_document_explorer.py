"""
Document Explorer UI
Browse and manage generated reports, exports, and logs
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime
import zipfile
import io


def show_document_explorer():
    """Document and report browser"""
    
    st.title("📚 파일 탐색기")
    st.markdown("""
    생성된 리포트, 내보낸 데이터, 분석 문서를 탐색하고 관리합니다.  
    **기능:** PDF 리포트 확인, 데이터 다운로드, 로그 조회
    """)
    
    st.markdown("---")
    
    # Document type selector
    doc_type = st.radio(
        "📂 문서 유형",
        [
            "📊 분석 리포트",
            "📁 내보낸 데이터",
            "📄 생성된 PDF",
            "📝 로그 파일"
        ],
        horizontal=True
    )
    
    st.markdown("---")
    
    # 1. Analysis Reports
    if doc_type == "📊 분석 리포트":
        st.subheader("📊 분석 리포트")
        
        reports_dir = Path("data/reports")
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Get PDF files
        pdf_files = list(reports_dir.glob("*.pdf"))
        
        if pdf_files:
            st.success(f"✅ {len(pdf_files)}개 리포트 발견")
            
            # Sort by modification time (newest first)
            pdf_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            # Display each report
            for pdf_file in pdf_files:
                with st.expander(f"📄 {pdf_file.name}"):
                    col1, col2, col3 = st.columns([3, 1, 1])
                    
                    with col1:
                        stat = pdf_file.stat()
                        mod_time = datetime.fromtimestamp(stat.st_mtime)
                        
                        st.text(f"📅 생성일: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
                        st.text(f"📦 크기: {stat.st_size / 1024:.1f} KB")
                        st.text(f"📍 경로: {pdf_file}")
                    
                    with col2:
                        # View button (placeholder)
                        if st.button("👁️ 보기", key=f"view_{pdf_file.name}"):
                            st.info("PDF 뷰어 기능은 실제 구현에서 가능합니다")
                    
                    with col3:
                        # Download button
                        try:
                            with open(pdf_file, 'rb') as f:
                                st.download_button(
                                    "📥 다운로드",
                                    f.read(),
                                    pdf_file.name,
                                    "application/pdf",
                                    key=f"dl_{pdf_file.name}"
                                )
                        except Exception as e:
                            st.error(f"다운로드 오류: {e}")
        else:
            st.info("📭 생성된 리포트가 없습니다")
            
            with st.expander("💡 리포트 생성 방법"):
                st.markdown("""
                리포트는 다음 페이지에서 생성할 수 있습니다:
                
                1. **성과지표 분석** → "📑 PDF 리포트 생성" 버튼
                2. **이미지 분석** → 분석 완료 후 "리포트 생성"
                3. **분석 히스토리** → 개별 분석 리포트
                
                리포트는 `data/reports/` 디렉토리에 저장됩니다.
                """)
    
    # 2. Exported Data
    elif doc_type == "📁 내보낸 데이터":
        st.subheader("📁 내보낸 데이터")
        
        exports_dir = Path("data/exports")
        exports_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all export files
        export_files = list(exports_dir.glob("*"))
        export_files = [f for f in export_files if f.is_file()]
        
        if export_files:
            st.success(f"✅ {len(export_files)}개 파일 발견")
            
            # Create file info table
            file_data = []
            for f in sorted(export_files, key=lambda x: x.stat().st_mtime, reverse=True):
                stat = f.stat()
                file_data.append({
                    '파일명': f.name,
                    '유형': f.suffix.upper() or 'N/A',
                    '크기': f"{stat.st_size / 1024:.1f} KB",
                    '수정일': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M'),
                    '경로': str(f)
                })
            
            df = pd.DataFrame(file_data)
            
            # Display table with download buttons
            st.dataframe(df.drop('경로', axis=1), use_container_width=True)
            
            # File selection for individual download
            st.markdown("---")
            st.subheader("개별 파일 다운로드")
            
            selected_file = st.selectbox(
                "파일 선택",
                [f.name for f in export_files]
            )
            
            if selected_file:
                file_path = exports_dir / selected_file
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.info(f"📄 {selected_file}")
                    st.text(f"크기: {file_path.stat().st_size / 1024:.1f} KB")
                
                with col2:
                    try:
                        with open(file_path, 'rb') as f:
                            st.download_button(
                                "📥 다운로드",
                                f.read(),
                                selected_file,
                                key=f"dl_export_{selected_file}",
                                use_container_width=True
                            )
                    except Exception as e:
                        st.error(f"오류: {e}")
            
            # Bulk download
            st.markdown("---")
            st.subheader("일괄 다운로드")
            
            if st.button("📦 전체 파일 ZIP으로 다운로드", type="primary"):
                zip_buffer = io.BytesIO()
                
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    for f in export_files:
                        zip_file.write(f, f.name)
                
                st.download_button(
                    "📥 ZIP 다운로드",
                    zip_buffer.getvalue(),
                    f"exports_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                    "application/zip"
                )
        else:
            st.info("📭 내보낸 데이터가 없습니다")
            
            with st.expander("💡 데이터 내보내기 방법"):
                st.markdown("""
                데이터는 다양한 페이지에서 내보낼 수 있습니다:
                
                1. **데이터 처리** → CSV/Excel 다운로드
                2. **성과지표 분석** → CSV/Excel 다운로드
                3. **분석 히스토리** → 내보내기 버튼
                
                내보낸 파일은 `data/exports/` 디렉토리에 저장됩니다.
                """)
    
    # 3. Generated PDFs
    elif doc_type == "📄 생성된 PDF":
        st.subheader("📄 생성된 PDF")
        st.info("이 섹션은 '📊 분석 리포트'와 동일한 기능을 제공합니다")
        # Redirect to reports
    
    # 4. Log Files
    elif doc_type == "📝 로그 파일":
        st.subheader("📝 로그 파일")
        
        logs_dir = Path("logs")
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        log_files = list(logs_dir.glob("*.log"))
        
        if log_files:
            st.success(f"✅ {len(log_files)}개 로그 파일")
            
            # Log file selector
            selected_log = st.selectbox(
                "로그 파일 선택",
                [f.name for f in sorted(log_files, reverse=True)]
            )
            
            if selected_log:
                log_path = logs_dir / selected_log
                
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    # Lines to show
                    n_lines = st.slider(
                        "표시할 줄 수",
                        10, 1000, 100,
                        step=10
                    )
                
                with col2:
                    # Refresh button
                    if st.button("🔄 새로고침"):
                        st.rerun()
                
                # Read and display log
                try:
                    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = f.readlines()
                        
                    # Show last N lines
                    last_lines = lines[-n_lines:] if len(lines) > n_lines else lines
                    
                    st.code(''.join(last_lines), language='log', line_numbers=True)
                    
                    # Download button
                    st.download_button(
                        "📥 전체 로그 다운로드",
                        ''.join(lines),
                        selected_log,
                        "text/plain"
                    )
                    
                except Exception as e:
                    st.error(f"로그 파일 읽기 오류: {e}")
        else:
            st.info("📭 로그 파일이 없습니다")
            
            with st.expander("💡 로그 파일 정보"):
                st.markdown("""
                로그 파일은 시스템 실행 중 자동으로 생성됩니다:
                
                - `api.log`: API 서버 로그
                - `ui.log`: UI 애플리케이션 로그
                - `error.log`: 오류 로그
                - `access.log`: 접근 로그
                
                로그는 문제 해결 및 모니터링에 유용합니다.
                """)
    
    # Directory management
    st.markdown("---")
    st.subheader("🗂️ 디렉토리 관리")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📁 디렉토리 생성", help="필요한 디렉토리 생성"):
            Path("data/reports").mkdir(parents=True, exist_ok=True)
            Path("data/exports").mkdir(parents=True, exist_ok=True)
            Path("logs").mkdir(parents=True, exist_ok=True)
            st.success("✅ 디렉토리 생성 완료")
    
    with col2:
        if st.button("🗑️ 임시 파일 정리", help="오래된 임시 파일 삭제"):
            st.info("실제 구현에서는 30일 이상 된 파일 삭제")


if __name__ == "__main__":
    # For standalone testing
    st.set_page_config(page_title="문서 탐색기", page_icon="📚", layout="wide")
    show_document_explorer()
