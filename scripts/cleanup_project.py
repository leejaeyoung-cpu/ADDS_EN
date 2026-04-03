#!/usr/bin/env python3
"""
Project Cleanup Script
Removes temporary fix scripts and organizes project structure
"""

import os
import shutil
from pathlib import Path
from typing import List
import re

# Patterns to match temporary files
TEMP_FILE_PATTERNS = [
    r'^fix_.*\.py$',
    r'^add_.*\.py$',
    r'^check_.*\.py$',
    r'^debug_.*\.py$',
    r'^test_.*\.py$',
    r'^temp_.*\.py$',
    r'^restore_.*\.py$',
    r'^review_.*\.py$',
    r'^verify_.*\.py$',
    r'^apply_.*\.py$',
    r'^move_.*\.py$',
    r'^simple_.*\.py$',
    r'^quick_.*\.py$',
    r'^update_.*\.py$',
    r'^compare_.*\.py$',
    r'^extract_.*\.py$',
    r'^final_.*\.py$',
    r'^complete_.*\.py$',
    r'^hybrid_.*\.py$',
    r'^manual_.*\.py$',
    r'^prepare_.*\.py$',
    r'^scan_.*\.py$',
    r'^surgical_.*\.py$',
    r'^convert_.*\.py$'
]

# Files to keep (exceptions)
KEEP_FILES = [
    'test_system.py',  # Main system test
]

def should_remove(filename: str) -> bool:
    """Check if file should be removed"""
    if filename in KEEP_FILES:
        return False
    
    for pattern in TEMP_FILE_PATTERNS:
        if re.match(pattern, filename):
            return True
    
    return False

def cleanup_root_directory(dry_run: bool = True):
    """
    Clean up temporary files from root directory
    
    Args:
        dry_run: If True, only print what would be deleted
    """
    project_root = Path('.')
    removed_files: List[Path] = []
    
    print("🧹 ADDS Project Cleanup")
    print("=" * 50)
    print(f"Mode: {'DRY RUN' if dry_run else 'ACTUAL DELETION'}")
    print()
    
    # Find all Python files in root
    for file_path in project_root.glob('*.py'):
        if should_remove(file_path.name):
            removed_files.append(file_path)
            
            if dry_run:
                print(f"Would remove: {file_path.name} ({file_path.stat().st_size} bytes)")
            else:
                try:
                    file_path.unlink()
                    print(f"✓ Removed: {file_path.name}")
                except Exception as e:
                    print(f"✗ Error removing {file_path.name}: {e}")
    
    # Find text files to remove
    text_patterns = ['*.txt']
    exclude_text = ['requirements.txt', 'LICENSE.txt']
    
    for pattern in text_patterns:
        for file_path in project_root.glob(pattern):
            if file_path.name not in exclude_text and 'corrupted' in file_path.name.lower():
                removed_files.append(file_path)
                
                if dry_run:
                    print(f"Would remove: {file_path.name} ({file_path.stat().st_size} bytes)")
                else:
                    try:
                        file_path.unlink()
                        print(f"✓ Removed: {file_path.name}")
                    except Exception as e:
                        print(f"✗ Error removing {file_path.name}: {e}")
    
    print()
    print("=" * 50)
    print(f"Total files to remove: {len(removed_files)}")
    
    if removed_files:
        total_size = sum(f.stat().st_size for f in removed_files if f.exists())
        print(f"Total size: {total_size:,} bytes ({total_size / 1024:.2f} KB)")
    
    if dry_run:
        print()
        print("⚠️  This was a DRY RUN. No files were actually deleted.")
        print("Run with --execute to actually delete files.")
    else:
        print()
        print("✅ Cleanup complete!")

def organize_scripts():
    """Move useful scripts to scripts/ directory"""
    project_root = Path('.')
    scripts_dir = project_root / 'scripts'
    scripts_dir.mkdir(exist_ok=True)
    
    # Scripts to keep and organize
    keep_scripts = [
        'test_system.py',
        'test_gpu_performance.py'
    ]
    
    print("\n📂 Organizing scripts...")
    for script in keep_scripts:
        src = project_root / script
        if src.exists():
            dst = scripts_dir / script
            if not dst.exists():
                shutil.copy2(src, dst)
                print(f"✓ Copied {script} to scripts/")

def create_gitkeep_files():
    """Create .gitkeep files in important empty directories"""
    directories = [
        'data/raw',
        'data/outputs',
        'data/processed',
        'data/cache',
        'data/exports',
        'models',
        'logs'
    ]
    
    print("\n📌 Creating .gitkeep files...")
    for dir_path in directories:
        full_path = Path(dir_path)
        full_path.mkdir(parents=True, exist_ok=True)
        
        gitkeep = full_path / '.gitkeep'
        if not gitkeep.exists():
            gitkeep.touch()
            print(f"✓ Created {gitkeep}")

if __name__ == '__main__':
    import sys
    
    # Check for --execute flag
    execute = '--execute' in sys.argv or '-e' in sys.argv
    
    # Run cleanup
    cleanup_root_directory(dry_run=not execute)
    
    if execute:
        organize_scripts()
        create_gitkeep_files()
    
    print()
