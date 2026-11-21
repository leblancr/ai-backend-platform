import sys

import httpx

URL = "http://localhost:8000/documents/assistant"
QUESTION = "What does the python backend engineering knowledge base say about FastAPI?"

response = httpx.get(URL, params={"q": QUESTION}, timeout=180.0)
response.raise_for_status()
data = response.json()

tools_used = [call["tool"] for call in data["tool_calls"]]
failures = []

if "search_document" not in tools_used:
    failures.append("never called search_document")

if "don't have enough information" in data["answer"]:
    failures.append("refused to answer")

print("Tools used:", tools_used)
print("Answer:", data["answer"])

if failures:
    print("FAIL:", "; ".join(failures))
    sys.exit(1)

print("PASS")
