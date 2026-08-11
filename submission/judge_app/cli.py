import json
import os
import time

from judge_app.judge import LLMJudge


INPUT_FILE = "data/test_suite.json"
OUTPUT_FILE = "reports/judge_results.json"


def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        suite = json.load(f)

    judge = LLMJudge()

    results = []

    print("=" * 60)
    print(f"Running suite: {suite['suite_name']}")
    print(f"Cases: {len(suite['cases'])}")
    print("=" * 60)

    for index, case in enumerate(suite["cases"], start=1):

        print(
            f"[{index}/{len(suite['cases'])}] "
            f"Judging {case['id']}..."
        )

        try:
            result = judge.judge_case(case)

            results.append({
                "case": case,
                "result": result
            })

            print(
                f"  Score: {result.get('overall_score')} | "
                f"Pass: {result.get('pass')}"
            )

        except Exception as e:

            print(
                f"  ERROR: {type(e).__name__}: {e}"
            )

            results.append({
                "case": case,
                "result": None,
                "error": str(e)
            })

    successful = [
        r for r in results
        if r["result"] is not None
    ]

    passed = [
        r for r in successful
        if r["result"].get("pass") is True
    ]

    scores = [
        r["result"]["overall_score"]
        for r in successful
        if isinstance(
            r["result"].get("overall_score"),
            (int, float)
        )
    ]

    total_latency = sum(
        r["result"]["_meta"]["latency_ms"]
        for r in successful
    )

    total_tokens = sum(
        r["result"]["_meta"]["total_tokens"]
        for r in successful
    )

    summary = {
        "suite_name": suite["suite_name"],
        "total_cases": len(results),
        "successful_cases": len(successful),
        "failed_cases": len(results) - len(successful),
        "passed_cases": len(passed),
        "pass_rate": (
            len(passed) / len(successful)
            if successful
            else 0
        ),
        "mean_score": (
            sum(scores) / len(scores)
            if scores
            else 0
        ),
        "total_latency_ms": total_latency,
        "mean_latency_ms": (
            total_latency / len(successful)
            if successful
            else 0
        ),
        "total_tokens": total_tokens
    }

    report = {
        "summary": summary,
        "results": results
    }

    os.makedirs("reports", exist_ok=True)

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            report,
            f,
            indent=2,
            ensure_ascii=False
        )

    print()
    print("=" * 60)
    print("SUITE COMPLETE")
    print("=" * 60)

    print(f"Cases:        {summary['total_cases']}")
    print(f"Successful:   {summary['successful_cases']}")
    print(f"Passed:       {summary['passed_cases']}")
    print(
        f"Pass rate:    "
        f"{summary['pass_rate']:.1%}"
    )
    print(
        f"Mean score:   "
        f"{summary['mean_score']:.2f}/5"
    )
    print(
        f"Mean latency: "
        f"{summary['mean_latency_ms']:.2f} ms"
    )
    print(
        f"Total tokens: "
        f"{summary['total_tokens']}"
    )

    print()
    print(f"Saved report: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()