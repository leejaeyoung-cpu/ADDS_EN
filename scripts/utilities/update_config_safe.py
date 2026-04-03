# config.yaml의 Cellpose 섹션 안전하게 업데이트

with open('configs/config.yaml', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Cellpose 섹션 찾아서 업데이트 (line 30-38)
updates = {
    33: '  gpu: true  # Auto-detect GPU, overridable by env var\n',
    34: '  batch_size: 8  # Optimized for RTX 5070\n',
    35: '  diameter: null  # Auto-detect (research-recommended)\n',
    36: '  flow_threshold: 0.6  # Increased for complex morphologies\n',
    37: '  cellprob_threshold: -1.0  # Lowered for better sensitivity\n',
}

for line_num, new_content in updates.items():
    if line_num - 1 < len(lines):
        lines[line_num - 1] = new_content
        print(f"Updated line {line_num}")

with open('configs/config.yaml', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("\nCellpose parameters updated successfully!")
print("\nNew settings:")
print("  diameter: null (auto-detect)")
print("  flow_threshold: 0.6")
print("  cellprob_threshold: -1.0")
print("  batch_size: 8")
print("  gpu: true")
