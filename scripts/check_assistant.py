import sys

import httpx

URL = "http://localhost:8000/documents/assistant"

response = httpx.get(
    URL,
    params={"q": "What documents do I have?"},
    timeout=180.0,
)
response.raise_for_status()
data = response.json()

answer = data["answer"]
tools_used = [call["tool"] for call in data["tool_calls"]]

failures = []

if "list_documents" not in tools_used:
    failures.append("did not call list_documents")

if "[Chunk ID" in answer:
    failures.append("cited a chunk when listing documents")

print("Answer:", answer)
print("Tools used:", tools_used)

if failures:
    print("FAIL:", "; ".join(failures))
    sys.exit(1)

print("PASS")
