import re

# Read the original file
with open('특허출원서_ADDS_최종통합본.md', 'r', encoding='utf-8') as f:
    content = f.read()

# Remove markdown formatting
clean = content

# Remove headers (# ## ### etc.)
clean = re.sub(r'^#{1,6}\s+', '', clean, flags=re.MULTILINE)

# Remove bold (**text**)
clean = re.sub(r'\*\*(.+?)\*\*', r'\1', clean)

# Remove italic (*text*)
clean = re.sub(r'\*(.+?)\*', r'\1', clean)

# Remove code blocks and inline code
clean = re.sub(r'```[^`]*```', '', clean, flags=re.DOTALL)
clean = re.sub(r'`([^`]+)`', r'\1', clean)

# Remove horizontal rules (---, ===)
clean = re.sub(r'^[-=]{3,}$', '', clean, flags=re.MULTILINE)

# Remove list markers (-, *, +)  
clean = re.sub(r'^[ \t]*[-*+]\s+', '', clean, flags=re.MULTILINE)

# Remove links [text](url)
clean = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', clean)

# Remove multiple consecutive newlines (but keep double newlines for paragraph separation)
clean = re.sub(r'\n{3,}', '\n\n', clean)

# Remove leading/trailing whitespace from lines
lines = [line.rstrip() for line in clean.split('\n')]
clean = '\n'.join(lines)

# Remove any remaining backticks
clean = clean.replace('`', '')

# Save back to the original file
with open('특허출원서_ADDS_최종통합본.md', 'w', encoding='utf-8') as f:
    f.write(clean)

print('특허출원서 파일이 업데이트되었습니다.')
print(f'원본 크기: {len(content)} bytes')
print(f'정리된 크기: {len(clean)} bytes')
print(f'제거된 크기: {len(content) - len(clean)} bytes')
