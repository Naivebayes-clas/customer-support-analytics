"""Generate Q&A pairs from knowledge base files using local LLM."""
import json
import subprocess
from pathlib import Path

kb_dir = Path("./data/knowledge_base")

def ask_ollama(prompt: str) -> str:
    result = subprocess.run(
        ["ollama", "run", "llama3.2:3b", prompt],
        capture_output=True, text=True, timeout=60
    )
    return result.stdout.strip()

pairs = []
for f in sorted(kb_dir.glob("*.txt")):
    text = f.read_text()[:1500]  # first 1500 chars is enough
    prompt = f"""Read this documentation text and generate ONE question-answer pair.
The question should be something a user would actually ask.
The answer should be 1-2 sentences extracted from the text.
Return ONLY valid JSON: {{"query": "...", "gold_answer": "..."}}

Text:
{text}"""
    try:
        raw = ask_ollama(prompt)
        # Extract JSON from response
        start = raw.index("{")
        end = raw.rindex("}") + 1
        pair = json.loads(raw[start:end])
        pair["expected_doc"] = f.stem
        pairs.append(pair)
        print(f"✓ {f.name}")
    except Exception as e:
        print(f"✗ {f.name}: {e}")

with open("./data/qa_pairs.json", "w") as f:
    json.dump(pairs, f, indent=2)
print(f"\nDone: {len(pairs)} pairs → data/qa_pairs.json")   
