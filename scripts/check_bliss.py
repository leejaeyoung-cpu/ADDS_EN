import re
src = open(r'f:\ADDS\figures\toxicity_verification_dashboard_v2.py', encoding='utf-8').read()
lines = src.split('\n')
bliss_lines = [(i+1, l) for i, l in enumerate(lines) if 'Bliss' in l]
print('Lines with Bliss (%d total):' % len(bliss_lines))
for no, l in bliss_lines:
    stripped = l.strip()
    is_comment_or_doc = stripped.startswith('#') or stripped.startswith('"""') or stripped.startswith("'''")
    print('  L%d [doc/comment=%s]: %s' % (no, is_comment_or_doc, stripped[:100]))
