import re

# Read the original file
with open('특허출원서_ADDS_최종통합본.md', 'r', encoding='utf-8') as f:
    content = f.read()

# Store original content length
original_length = len(content)

# Apply cleaning transformations
clean = content

# 1. Remove headers (# ## ### etc.) - must be at start of line
clean = re.sub(r'^#{1,6}\s+(.*)$', r'\1', clean, flags=re.MULTILINE)

# 2. Remove bold (**text** or __text__)
clean = re.sub(r'\*\*(.+?)\*\*', r'\1', clean)
clean = re.sub(r'__(.+?)__', r'\1', clean)

# 3. Remove italic - minimal approach to avoid issues
clean = re.sub(r'\*([^\*\n]+)\*', r'\1', clean)

# 4. Remove code blocks (```...```)
clean = re.sub(r'```[\s\S]*?```', '', clean)

# 5. Remove inline code (`text`)
clean = re.sub(r'`([^`\n]+)`', r'\1', clean)

# 6. Remove horizontal rules (---, ===, ***)
clean = re.sub(r'^[\s]*[-=*]{3,}[\s]*$', '', clean, flags=re.MULTILINE)

# 7. Remove list markers (-, *, +, numbered lists)
clean = re.sub(r'^[\s]*[-*+]\s+', '', clean, flags=re.MULTILINE)
clean = re.sub(r'^[\s]*\d+\.\s+', '', clean, flags=re.MULTILINE)

# 8. Remove markdown links [text](url) - keep only text
clean = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', clean)

# 9. Remove any remaining single backticks
clean = clean.replace('`', '')

# 10. Remove blockquote markers (>)
clean = re.sub(r'^[\s]*>\s+', '', clean, flags=re.MULTILINE)

# 11. Clean up excessive newlines (more than 2 consecutive)
clean = re.sub(r'\n{3,}', '\n\n', clean)

# 12. Remove trailing whitespace from each line
lines = [line.rstrip() for line in clean.split('\n')]
clean = '\n'.join(lines)

# 13. Remove leading/trailing whitespace from the entire document
clean = clean.strip()

# Save back to file
with open('특허출원서_ADDS_최종통합본.md', 'w', encoding='utf-8') as f:
    f.write(clean)

print('Patent document cleaned successfully')
print(f'Original size: {original_length:,} bytes')
print(f'Cleaned size: {len(clean):,} bytes')
print(f'Removed: {original_length - len(clean):,} bytes')
