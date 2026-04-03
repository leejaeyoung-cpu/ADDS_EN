"""
Real-time Extraction Monitor
=============================
Live monitoring of abstract extraction progress with visual updates.

Features:
- Real-time progress display
- Live quality metrics
- Error tracking
- ETA calculation
- Beautiful terminal UI with colors

Author: ADDS Team
Date: 2026-01-31
"""

import json
import time
import os
from pathlib import Path
from datetime import datetime, timedelta
import sys
from typing import Dict, List, Optional

# Paths
EXTRACTED_DIR = Path("data/extracted/abstracts")
PROGRESS_FILE = EXTRACTED_DIR / "progress.json"
LOG_FILE = EXTRACTED_DIR / "extraction.log"


class ExtractionMonitor:
    """Real-time extraction progress monitor"""
    
    def __init__(self, refresh_interval: float = 2.0):
        self.refresh_interval = refresh_interval
        self.start_time = None
        self.last_progress = None
        
    def clear_screen(self):
        """Clear terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def get_terminal_width(self) -> int:
        """Get terminal width for formatting"""
        try:
            return os.get_terminal_size().columns
        except:
            return 80
    
    def load_progress(self) -> Optional[Dict]:
        """Load current progress"""
        try:
            if PROGRESS_FILE.exists():
                with open(PROGRESS_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            return None
        return None
    
    def get_latest_extractions(self, count: int = 5) -> List[str]:
        """Get latest extracted files"""
        try:
            files = sorted(
                EXTRACTED_DIR.glob("*_extracted.json"),
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )
            return [f.stem.replace('_extracted', '') for f in files[:count]]
        except:
            return []
    
    def get_recent_log_lines(self, count: int = 10) -> List[str]:
        """Get recent log lines"""
        try:
            if LOG_FILE.exists():
                with open(LOG_FILE, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                    return [line.strip() for line in lines[-count:] if line.strip()]
        except:
            return []
        return []
    
    def calculate_eta(self, completed: int, total: int, elapsed: timedelta) -> str:
        """Calculate estimated time remaining"""
        if completed == 0:
            return "calculating..."
        
        rate = completed / elapsed.total_seconds()  # papers per second
        remaining = total - completed
        
        if rate > 0:
            seconds_remaining = remaining / rate
            eta = timedelta(seconds=int(seconds_remaining))
            return str(eta)
        return "unknown"
    
    def format_duration(self, seconds: float) -> str:
        """Format duration nicely"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"
    
    def draw_progress_bar(self, current: int, total: int, width: int = 50) -> str:
        """Draw a progress bar"""
        if total == 0:
            return "[" + " " * width + "] 0.0%"
        
        percent = current / total
        filled = int(width * percent)
        bar = "█" * filled + "░" * (width - filled)
        return f"[{bar}] {percent*100:.1f}%"
    
    def display_status(self, progress: Dict, elapsed: Optional[timedelta] = None):
        """Display current status"""
        width = self.get_terminal_width()
        
        # Header
        print("=" * width)
        print("📊 LITERATURE EXTRACTION - REAL-TIME MONITOR".center(width))
        print("=" * width)
        print()
        
        # Current time and elapsed
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"⏰ Current Time: {current_time}")
        
        if elapsed:
            print(f"⌛ Elapsed: {self.format_duration(elapsed.total_seconds())}")
        print()
        
        # Progress metrics
        completed = len(set(progress.get('completed', [])))
        failed = len(set(progress.get('failed', [])))
        pending = len(progress.get('pending', []))
        
        # Estimate total from metadata
        try:
            metadata_file = Path("data/literature/comprehensive_metadata.json")
            if metadata_file.exists():
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    total = len([
                        p for p in metadata.get('papers', [])
                        if p.get('abstract') and len(p.get('abstract', '')) > 100
                    ])
            else:
                total = completed + failed + pending
        except:
            total = completed + failed + pending
        
        if total == 0:
            total = completed + failed
        
        # Progress bar
        print("📈 OVERALL PROGRESS")
        print("-" * width)
        bar_width = min(60, width - 20)
        progress_bar = self.draw_progress_bar(completed, total, bar_width)
        print(f"   {progress_bar}")
        print()
        
        # Statistics
        print("📊 STATISTICS")
        print("-" * width)
        print(f"   ✅ Completed:  {completed:>4} / {total}")
        print(f"   ❌ Failed:     {failed:>4}")
        print(f"   ⏳ Remaining:  {total - completed:>4}")
        
        if completed > 0:
            success_rate = (completed / (completed + failed)) * 100 if (completed + failed) > 0 else 0
            print(f"   📈 Success Rate: {success_rate:.1f}%")
        
        if elapsed and completed > 0:
            rate = completed / elapsed.total_seconds() * 60  # papers per minute
            print(f"   ⚡ Rate: {rate:.2f} papers/min")
            
            eta = self.calculate_eta(completed, total, elapsed)
            print(f"   ⏱️  ETA: {eta}")
        
        print()
        
        # Latest extractions
        latest = self.get_latest_extractions(5)
        if latest:
            print("🔄 LATEST EXTRACTIONS")
            print("-" * width)
            for pmid in latest:
                print(f"   ✓ PMID {pmid}")
            print()
        
        # Recent log activity
        recent_logs = self.get_recent_log_lines(5)
        if recent_logs:
            print("📝 RECENT ACTIVITY")
            print("-" * width)
            for log_line in recent_logs:
                # Truncate long lines
                if len(log_line) > width - 5:
                    log_line = log_line[:width-8] + "..."
                print(f"   {log_line}")
            print()
        
        # Footer
        print("=" * width)
        print("Press Ctrl+C to exit monitor".center(width))
        print("=" * width)
    
    def monitor(self):
        """Start monitoring"""
        print("Starting extraction monitor...")
        print("Checking for extraction process...")
        time.sleep(1)
        
        self.start_time = datetime.now()
        
        try:
            while True:
                self.clear_screen()
                
                # Load current progress
                progress = self.load_progress()
                
                if not progress:
                    print("⚠️  No progress file found. Waiting for extraction to start...")
                    print(f"   Looking for: {PROGRESS_FILE}")
                    time.sleep(self.refresh_interval)
                    continue
                
                # Calculate elapsed time
                elapsed = datetime.now() - self.start_time
                
                # Display status
                self.display_status(progress, elapsed)
                
                # Check if extraction is complete
                completed = len(set(progress.get('completed', [])))
                failed = len(set(progress.get('failed', [])))
                
                # Detect if process stopped
                if self.last_progress:
                    last_completed = len(set(self.last_progress.get('completed', [])))
                    if completed == last_completed and completed > 0:
                        # No progress in this interval
                        print("\n⚠️  No new extractions detected. Process may have completed or stopped.")
                
                self.last_progress = progress
                
                # Wait before next refresh
                time.sleep(self.refresh_interval)
                
        except KeyboardInterrupt:
            print("\n\n👋 Monitor stopped by user.")
            print("\nFinal Status:")
            if progress:
                completed = len(set(progress.get('completed', [])))
                failed = len(set(progress.get('failed', [])))
                print(f"   Completed: {completed}")
                print(f"   Failed: {failed}")
            print()


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Real-time extraction monitor")
    parser.add_argument(
        '--interval',
        type=float,
        default=2.0,
        help='Refresh interval in seconds (default: 2.0)'
    )
    
    args = parser.parse_args()
    
    monitor = ExtractionMonitor(refresh_interval=args.interval)
    monitor.monitor()


if __name__ == "__main__":
    main()
