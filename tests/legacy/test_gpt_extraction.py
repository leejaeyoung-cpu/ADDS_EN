"""
Quick test to see what GPT-4 actually returns
"""
import json
import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

test_abstract = "KRAS is the most frequently mutated oncogene in human cancer."

prompt = """Extract structured cancer knowledge from this abstract. RESPOND ONLY WITH A VALID JSON OBJECT.

Return JSON with this structure:
{
  "mechanisms": ["list of pathways"],
  "drugs": ["list of drugs"]
}

ABSTRACT:
""" + test_abstract

print("=" * 70)
print("SENDING REQUEST TO GPT-4...")
print("=" * 70)

response = client.chat.completions.create(
    model="gpt-4-turbo-preview",
    messages=[
        {"role": "system", "content": "You are an expert oncology knowledge extractor. Return only valid JSON."},
        {"role": "user", "content": prompt}
    ],
    response_format={"type": "json_object"},
    temperature=0.1,
    max_tokens=500
)

content = response.choices[0].message.content

print("\nRESPONSE RECEIVED:")
print("=" * 70)
print(content)
print("=" * 70)

try:
    parsed = json.loads(content)
    print("\n✅ JSON PARSING SUCCESSFUL!")
    print(json.dumps(parsed, indent=2))
except json.JSONDecodeError as e:
    print(f"\n❌ JSON PARSING FAILED: {e}")
    print(f"First 200 chars: {content[:200]}")
