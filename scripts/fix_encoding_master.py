import re
src = open(r'f:\ADDS\scripts\adds_dataset_enrichment_master.py', encoding='utf-8').read()
# Replace all non-ASCII chars that cause cp949 issues
replacements = [
    ('\u2014', '--'), ('\u2013', '-'),
    ('\u2019', "'"), ('\u2018', "'"),
    ('\u201c', '"'), ('\u201d', '"'),
    ('\u00b2', '2'), ('\u00b3', '3'),
    ('\u2265', '>='), ('\u2264', '<='),
    ('\u00b1', '+/-'), ('\u00d7', 'x'),
    ('\u03c1', 'rho'), ('\u03b1', 'alpha'),
    ('\u2192', '->'), ('\u2190', '<-'),
]
for old, new in replacements:
    src = src.replace(old, new)

# Also encode the file as ascii-safe
# Find any remaining non-ASCII
bad = [(i, c) for i, c in enumerate(src) if ord(c) > 127]
if bad:
    print(f"Remaining non-ASCII chars: {len(bad)}")
    for pos, c in bad[:10]:
        print(f"  pos={pos} char={repr(c)} (U+{ord(c):04X})")
    # Replace all remaining with '?'
    src = ''.join(c if ord(c) <= 127 else '?' for c in src)
    print("Replaced with ?")

open(r'f:\ADDS\scripts\adds_dataset_enrichment_master.py', 'w', encoding='utf-8').write(src)
print("Done - file saved ASCII-safe")
