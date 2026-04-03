import os
targets = ['collect_training_data.py']
for fname in targets:
    path = os.path.join(r'f:\ADDS\scripts', fname)
    txt = open(path, encoding='utf-8').read()
    fixed = (txt
        .replace('\u2014', '--')
        .replace('\u2013', '-')
        .replace('\u2192', '->')
        .replace('\u2190', '<-')
        .replace('\u2022', '*')
        .replace('\u2018', "'")
        .replace('\u2019', "'")
        .replace('\u201c', '"')
        .replace('\u201d', '"')
    )
    open(path, 'w', encoding='utf-8').write(fixed)
    remaining = sum(1 for c in fixed if ord(c) > 127)
    print(f"Fixed {fname}: {remaining} non-ASCII chars remaining (Korean text OK)")
