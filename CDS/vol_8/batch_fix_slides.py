import os
import glob
import re
import subprocess

def fix_and_run_all():
    target_dir = r"f:\ADDS\CDS"
    files = glob.glob(os.path.join(target_dir, "create_ppt_slide_*.py"))
    
    for file_path in files:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 1. Widen the pane
        content = re.sub(r'pane_w\s*=\s*\d+', 'pane_w = 1100', content)
        content = re.sub(r'px\s*=\s*W\s*-\s*pane_w\s*-\s*\d+', 'px = W - pane_w - 40', content)
        
        # 2. Reduce font sizes to prevent cutoff
        content = re.sub(r'title_font\s*=\s*get_font\(\s*\d+', 'title_font = get_font(38', content)
        content = re.sub(r'sub_font\s*=\s*get_font\(\s*\d+', 'sub_font = get_font(24', content)
        content = re.sub(r'point_font\s*=\s*get_font\(\s*\d+', 'point_font = get_font(26', content)
        content = re.sub(r'desc_font\s*=\s*get_font\(\s*\d+', 'desc_font = get_font(20', content)
        
        # Fix the footer font if it exists
        content = re.sub(r'font\s*=\s*get_font\(\s*2[24]\s*,\s*bold=True\)', 'font=get_font(20, bold=True)', content)

        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
            
        print(f"Fixed {file_path}")
        subprocess.run(['python', file_path], cwd=target_dir)

if __name__ == "__main__":
    fix_and_run_all()
