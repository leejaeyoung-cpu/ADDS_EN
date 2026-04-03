def show_analysis_history():
    """Analysis History Page - Comprehensive filtering, statistics, and export"""
    st.header("📊 분석 히스토리 - 과거 분석 결과 조회")
    
    try:
        history_mgr = AnalysisHistoryManager()
        
        # Global Statistics Dashboard
        st.markdown("### 📈 전체 통계")
        stats = history_mgr.get_statistics()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "총 분석 수",
                f"{stats['total_analyses']:,}",
                help="모든 분석 기록 수"
            )
        
        with col2:
            st.metric(
                "총 세포 수",
                f"{stats['total_cells']:,}",
                help="모든 분석에서 검출된 총 세포 수"
            )
        
        with col3:
            avg_cells = stats.get('avg_cells_per_analysis', 0)
            st.metric(
                "평균 세포 수",
                f"{avg_cells:.0f}",
                help="분석당 평균 세포 수"
            )
        
        with col4:
            avg_quality = stats.get('avg_quality_score', 0)
            st.metric(
                "평균 품질 점수",
                f"{avg_quality:.2f}",
                help="전체 분석의 평균 이미지 품질"
            )
        
        st.markdown("---")
        
        # Filters Section
        st.markdown("### 🔍 필터")
        
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        
        with filter_col1:
            filename_filter = st.text_input(
                "파일명 검색",
                placeholder="예: HUVEC",
                help="파일명에 포함된 텍스트로 검색"
            )
        
        with filter_col2:
            experiment_filter = st.text_input(
                "실험명 검색",
                placeholder="예: TimeCourse",
                help="실험명으로 검색"
            )
        
        with filter_col3:
            limit = st.number_input(
                "표시 개수",
                min_value=10,
                max_value=1000,
                value=50,
                step=10,
                help="한 번에 표시할 최대 기록 수"
            )
        
        # Date range filter
        date_col1, date_col2 = st.columns(2)
        with date_col1:
            date_from = st.date_input("시작일", value=None, help="이 날짜 이후의 기록만 표시")
        
        with date_col2:
            date_to = st.date_input("종료일", value=None, help="이 날짜 이전의 기록만 표시")
        
        st.markdown("---")
        
        # Get filtered history
        records = history_mgr.get_history(
            filename_filter=filename_filter if filename_filter else None,
            experiment_filter=experiment_filter if experiment_filter else None,
            date_from=date_from,
            date_to=date_to,
            limit=limit
        )
        
        # Export Section
        if records:
            st.markdown(f"### 📋 분석 기록 ({len(records)}개)")
            
            # Export buttons
            export_col1, export_col2, export_col3, _ = st.columns([1, 1, 1, 3])
            
            with export_col1:
                if st.button("📥 CSV 내보내기", use_container_width=True):
                    export_path = Path("data/exports") / f"analysis_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                    export_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    if history_mgr.export_to_csv(str(export_path)):
                        st.success(f"✓ CSV 저장 완료: {export_path.name}")
                        with open(export_path, 'rb') as f:
                            st.download_button(
                                "⬇️ 다운로드",
                                data=f,
                                file_name=export_path.name,
                                mime="text/csv",
                                key="csv_download"
                            )
                    else:
                        st.error("CSV 내보내기 실패")
            
            with export_col2:
                if st.button("📥 JSON 내보내기", use_container_width=True):
                    export_path = Path("data/exports") / f"analysis_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    export_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    if history_mgr.export_to_json(str(export_path)):
                        st.success(f"✓ JSON 저장 완료: {export_path.name}")
                        with open(export_path, 'rb') as f:
                            st.download_button(
                                "⬇️ 다운로드",
                                data=f,
                                file_name=export_path.name,
                                mime="application/json",
                                key="json_download"
                            )
                    else:
                        st.error("JSON 내보내기 실패")
            
            with export_col3:
                if st.button("📥 Excel 내보내기", use_container_width=True):
                    try:
                        export_path = Path("data/exports") / f"analysis_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                        export_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        if history_mgr.export_to_excel(str(export_path)):
                            st.success(f"✓ Excel 저장 완료: {export_path.name}")
                            with open(export_path, 'rb') as f:
                                st.download_button(
                                    "⬇️ 다운로드",
                                    data=f,
                                    file_name=export_path.name,
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    key="excel_download"
                                )
                        else:
                            st.error("Excel 내보내기 실패")
                    except Exception as e:
                        st.error(f"Excel 내보내기 실패: openpyxl이 설치되지 않았을 수 있습니다.")
            
            st.markdown("---")
            
            # Display records as expandable cards
            for idx, record in enumerate(records):
                # Create card title
                timestamp_str = record.get('timestamp', 'N/A')
                try:
                    dt = datetime.fromisoformat(timestamp_str)
                    timestamp_display = dt.strftime('%Y-%m-%d %H:%M')
                except:
                    timestamp_display = timestamp_str
                
                image_name = record.get('image_name', 'Unknown')
                num_cells = record.get('num_cells', 0)
                quality_score = record.get('quality_score', 0)
                
                # Expander title with key info
                title = f"#{record['id']} | {image_name} | {num_cells} 세포 | 품질: {quality_score:.2f} | {timestamp_display}"
                
                with st.expander(title):
                    # Two-column layout for record details
                    detail_col1, detail_col2 = st.columns(2)
                    
                    with detail_col1:
                        st.markdown("**기본 정보**")
                        st.write(f"- **ID**: {record['id']}")
                        st.write(f"- **파일명**: {image_name}")
                        st.write(f"- **분석 시간**: {timestamp_display}")
                        
                        if record.get('experiment_name'):
                            st.write(f"- **실험명**: {record['experiment_name']}")
                        if record.get('cell_line'):
                            st.write(f"- **세포주**: {record['cell_line']}")
                        if record.get('treatment'):
                            st.write(f"- **처리**: {record['treatment']}")
                        if record.get('condition'):
                            st.write(f"- **조건**: {record['condition']}")
                        if record.get('replicate_number'):
                            st.write(f"- **반복**: #{record['replicate_number']}")
                    
                    with detail_col2:
                        st.markdown("**분석 결과**")
                        st.write(f"- **세포 수**: {num_cells:,}")
                        st.write(f"- **평균 면적**: {record.get('mean_area', 0):.1f} px²")
                        st.write(f"- **면적 표준편차**: {record.get('std_area', 0):.1f}")
                        st.write(f"- **평균 원형도**: {record.get('mean_circularity', 0):.3f}")
                        st.write(f"- **품질 점수**: {quality_score:.2f}")
                        st.write(f"- **품질 등급**: {record.get('quality_grade', 'N/A')}")
                        
                        # Performance metrics if available
                        if record.get('processing_time_seconds'):
                            st.write(f"- **처리 시간**: {format_duration(record['processing_time_seconds'])}")
                        if record.get('gpu_memory_used_mb'):
                            st.write(f"- **GPU 메모리**: {format_file_size(int(record['gpu_memory_used_mb'] * 1024 * 1024))}")
                        if record.get('model_version'):
                            st.write(f"- **모델 버전**: {record['model_version']}")
                    
                    # Notes section
                    if record.get('notes'):
                        st.markdown("**메모**")
                        st.info(record['notes'])
                    
                    # Action buttons
                    btn_col1, btn_col2, btn_col3 = st.columns(3)
                    
                    with btn_col1:
                        if st.button("📝 메모 추가/수정", key=f"note_btn_{record['id']}_{idx}"):
                            st.session_state[f'editing_notes_{record["id"]}'] = True
                    
                    with btn_col2:
                        if st.button("🔄 파라미터 재사용", key=f"reuse_btn_{record['id']}_{idx}"):
                            # Load parameters into session state for reuse
                            if record.get('experiment_name'):
                                st.session_state['experiment_name'] = record['experiment_name']
                            if record.get('cell_line'):
                                st.session_state['cell_line'] = record['cell_line']
                            if record.get('treatment'):
                                st.session_state['treatment'] = record['treatment']
                            if record.get('condition'):
                                st.session_state['parsed_condition'] = record['condition']
                            
                            st.success("✓ 파라미터가 로드되었습니다! 이미지 분석 페이지에서 사용하세요.")
                    
                    with btn_col3:
                        if st.button("🗑️ 삭제", key=f"del_btn_{record['id']}_{idx}"):
                            db = AnalysisDatabase()
                            db.delete_analysis(record['id'])
                            st.success("✓ 기록이 삭제되었습니다")
                            st.rerun()
                    
                    # Notes editing
                    if st.session_state.get(f'editing_notes_{record["id"]}', False):
                        new_note = st.text_area(
                            "메모 내용",
                            value=record.get('notes', ''),
                            key=f"note_edit_{record['id']}_{idx}"
                        )
                        
                        save_col, cancel_col = st.columns(2)
                        with save_col:
                            if st.button("💾 저장", key=f"save_note_{record['id']}_{idx}"):
                                db = AnalysisDatabase()
                                db.update_notes(record['id'], new_note)
                                st.session_state[f'editing_notes_{record["id"]}'] = False
                                st.success("✓ 메모가 저장되었습니다")
                                st.rerun()
                        
                        with cancel_col:
                            if st.button("취소", key=f"cancel_note_{record['id']}_{idx}"):
                                st.session_state[f'editing_notes_{record["id"]}'] = False
                                st.rerun()
        
        else:
            st.info("📌 필터 조건에 맞는 기록이 없습니다. 필터를 조정하거나 새로운 분석을 실행하세요.")
    
    except Exception as e:
        st.error(f"오류 발생: {str(e)}")
        import traceback
        with st.expander("오류 상세"):
            st.code(traceback.format_exc())


