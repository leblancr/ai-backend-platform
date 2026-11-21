import argparse
import time

import httpx


EVALUATION_CASES = [
    {
        "name": "Authentication and password security",
        "question": "How should an API authenticate users and protect passwords?",
        "expected_terms": ["bearer", "argon2id", "bcrypt"],
        "should_abstain": False,
    },
    {
        "name": "Password hashing",
        "question": "Why should passwords not be stored using SHA-256?",
        "expected_terms": ["sha-256", "password"],
        "should_abstain": False,
    },
    {
        "name": "Document authorization",
        "question": "How can a server prevent one user from accessing another user's documents?",
        "expected_terms": ["user_id", "document_id"],
        "should_abstain": False,
    },
    {
        "name": "Unknown server information",
        "question": "What is the public IP address of the server hosting this application?",
        "expected_terms": [
            "i don't have enough information in the provided documents."
        ],
        "should_abstain": True,
    },
]


def evaluate_case(
    client: httpx.Client,
    base_url: str,
    document_id: int,
    case: dict,
) -> dict:
    url = f"{base_url}/documents/{document_id}/ask"

    start = time.perf_counter()

    response = client.get(
        url,
        params={"q": case["question"]},
    )

    elapsed = time.perf_counter() - start
    response.raise_for_status()

    result = response.json()
    answer = result["answer"].lower()
    sources = result["sources"]

    missing_terms = [
        term
        for term in case["expected_terms"]
        if term.lower() not in answer
    ]

    sources_valid = (
        isinstance(sources, list)
        and len(sources) > 0
        and all(
            "chunk_id" in source
            and "similarity" in source
            for source in sources
        )
    )

    passed = not missing_terms and sources_valid

    return {
        "name": case["name"],
        "passed": passed,
        "elapsed": elapsed,
        "source_count": len(sources),
        "missing_terms": missing_terms,
        "should_abstain": case["should_abstain"],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the RAG question-answering system"
    )
    parser.add_argument("--document-id", type=int, default=2)
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
    )
    args = parser.parse_args()

    results = []

    with httpx.Client(timeout=240.0) as client:
        for case in EVALUATION_CASES:
            print(f"\nTesting: {case['name']}")

            result = evaluate_case(
                client,
                args.base_url.rstrip("/"),
                args.document_id,
                case,
            )

            results.append(result)

            status = "PASS" if result["passed"] else "FAIL"

            print(f"  Result: {status}")
            print(f"  Response time: {result['elapsed']:.2f}s")
            print(f"  Sources returned: {result['source_count']}")

            if result["missing_terms"]:
                print(
                    "  Missing expected terms:",
                    ", ".join(result["missing_terms"]),
                )

    passed = sum(result["passed"] for result in results)
    total = len(results)

    print("\n" + "=" * 45)
    print(f"Evaluation results: {passed}/{total} passed")
    print(f"Pass rate: {passed / total * 100:.1f}%")
    print(
        "Average response time: "
        f"{sum(r['elapsed'] for r in results) / total:.2f}s"
    )
    print("=" * 45)

    if passed != total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
