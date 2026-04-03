import fitz

doc = fitz.open("2. [양식] 연구개발계획서(PART2_연구자 작성) 26.01.28 이상훈 최종 (1).pdf")
text = ""
for page in doc:
    text += page.get_text()

with open("plan_text.txt", "w", encoding="utf-8") as f:
    f.write(text)
