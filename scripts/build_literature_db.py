"""
Batch Literature Collection
대량 논문 수집 실행 스크립트
"""

import subprocess
import sys
from pathlib import Path
import time

def run_step(script_name: str, description: str):
    """단계별 실행"""
    print("\n" + "="*80)
    print(f"STEP: {description}")
    print("="*80 + "\n")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
        
        elapsed = time.time() - start_time
        print(f"\n✓ Completed in {elapsed:.1f}s")
        return True
    
    except subprocess.CalledProcessError as e:
        print(f"✗ Error: {e}")
        print(e.stderr)
        return False


def main():
    """전체 파이프라인 실행"""
    
    scripts_dir = Path("F:/ADDS/scripts")
    
    steps = [
        ("collect_literature.py", "논문 수집 (PubMed + ArXiv)"),
        ("extract_formulas.py", "수식 및 파라미터 추출"),
        ("rag_system.py", "RAG 시스템 구축")
    ]
    
    print("="*80)
    print("LITERATURE DATABASE CONSTRUCTION PIPELINE")
    print("="*80)
    print(f"Target: 400+ papers on physics-based drug modeling")
    print(f"Output: F:/ADDS/literature_database/")
    print()
    
    input("Press Enter to start...")
    
    success_count = 0
    
    for script_name, description in steps:
        script_path = scripts_dir / script_name
        
        if not script_path.exists():
            print(f"✗ Script not found: {script_path}")
            continue
        
        if run_step(str(script_path), description):
            success_count += 1
        else:
            print(f"\n⚠️ Step failed: {description}")
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                break
    
    # 결과 요약
    print("\n" + "="*80)
    print("PIPELINE SUMMARY")
    print("="*80)
    print(f"Completed: {success_count}/{len(steps)} steps")
    
    # 데이터베이스 확인
    db_dir = Path("F:/ADDS/literature_database")
    if db_dir.exists():
        json_files = list(db_dir.glob("*.json"))
        txt_files = list(db_dir.glob("*.txt"))
        
        print(f"\nDatabase files:")
        print(f"  JSON: {len(json_files)} files")
        print(f"  Text: {len(txt_files)} files")
        
        # 논문 개수
        total_papers = 0
        for json_file in json_files:
            if 'all_papers' in json_file.name:
                import json
                with open(json_file, 'r') as f:
                    papers = json.load(f)
                    total_papers = len(papers)
                break
        
        print(f"  Total papers: {total_papers}")
    
    print("\n✓ Pipeline complete!")


if __name__ == '__main__':
    main()
