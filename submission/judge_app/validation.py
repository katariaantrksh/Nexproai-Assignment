import json
import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types


load_dotenv()


class JudgeValidator:

    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        self.model = os.getenv(
            "JUDGE_MODEL",
            "gemini-3.1-flash-lite"
        )

        if not self.api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY is missing from .env"
            )

        self.client = genai.Client(
            api_key=self.api_key
        )

    def call(self, case):

        prompt = f"""
You are validating an LLM judge.

Evaluate the following model answer.

Question:
{case["question"]}

Model answer:
{case["answer"]}

Expected answer:
{case["expected"]}

Evaluate factual correctness and whether the answer
satisfies the question.

Important:
- Do not reward confidence.
- Do not reward unnecessary verbosity.
- Do not punish concise correct answers.
- Focus on the actual content.

Return ONLY JSON:

{{
  "score": 1,
  "pass": true,
  "reason": "short explanation"
}}

Scoring:
1 = completely incorrect
2 = mostly incorrect
3 = partially correct
4 = correct
5 = excellent

Pass should be true only when the answer is acceptable.
"""

        start = time.perf_counter()

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )

        latency = round(
            (time.perf_counter() - start) * 1000,
            2
        )

        raw = response.text or ""

        try:
            result = json.loads(raw)

        except json.JSONDecodeError:

            start_json = raw.find("{")
            end_json = raw.rfind("}")

            if (
                start_json != -1
                and end_json > start_json
            ):
                result = json.loads(
                    raw[start_json:end_json + 1]
                )
            else:
                raise ValueError(
                    "Could not parse judge response:\n"
                    + raw
                )

        result["_meta"] = {
            "model": self.model,
            "latency_ms": latency
        }

        return result

    def run(self):

        cases = [
            {
                "id": "gold_01",
                "question": "What is the capital of France?",
                "answer": "Paris.",
                "expected": "Paris.",
                "gold_pass": True
            },
            {
                "id": "gold_02",
                "question": "What is 10 multiplied by 5?",
                "answer": "50.",
                "expected": "50.",
                "gold_pass": True
            },
            {
                "id": "gold_03",
                "question": "What is the largest planet?",
                "answer": "Mars.",
                "expected": "Jupiter.",
                "gold_pass": False
            },
            {
                "id": "gold_04",
                "question": "What is 2 + 2?",
                "answer": "2 + 2 = 5.",
                "expected": "4.",
                "gold_pass": False
            },
            {
                "id": "gold_05",
                "question": "Name the capital of Japan.",
                "answer": "Tokyo.",
                "expected": "Tokyo.",
                "gold_pass": True
            },
            {
                "id": "gold_06",
                "question": "What does CPU stand for?",
                "answer": "Central Processing Unit.",
                "expected": "Central Processing Unit.",
                "gold_pass": True
            }
        ]

        results = []

        print("=" * 60)
        print("JUDGE VALIDATION")
        print("=" * 60)

        for case in cases:

            print()
            print(
                f"Testing {case['id']}..."
            )

            run_1 = self.call(case)
            run_2 = self.call(case)

            pass_1 = bool(
                run_1.get("pass")
            )

            pass_2 = bool(
                run_2.get("pass")
            )

            test_retest_same = (
                pass_1 == pass_2
            )

            gold_agreement = (
                pass_1 == case["gold_pass"]
            )

            results.append({
                "id": case["id"],
                "gold_pass": case["gold_pass"],
                "run_1": run_1,
                "run_2": run_2,
                "gold_agreement": gold_agreement,
                "test_retest_same": test_retest_same
            })

            print(
                f"  Gold: {case['gold_pass']}"
            )

            print(
                f"  Run 1: {pass_1}"
            )

            print(
                f"  Run 2: {pass_2}"
            )

            print(
                f"  Gold agreement: "
                f"{gold_agreement}"
            )

            print(
                f"  Test-retest same: "
                f"{test_retest_same}"
            )

        total = len(results)

        gold_agreements = sum(
            r["gold_agreement"]
            for r in results
        )

        retest_agreements = sum(
            r["test_retest_same"]
            for r in results
        )

        gold_agreement_rate = (
            gold_agreements / total
        )

        test_retest_rate = (
            retest_agreements / total
        )

        flip_rate = (
            1 - test_retest_rate
        )

        report = {
            "validation_method": [
                "gold_label_agreement",
                "test_retest_consistency"
            ],
            "model": self.model,
            "sample_size": total,
            "gold_agreement_rate":
                gold_agreement_rate,
            "test_retest_consistency":
                test_retest_rate,
            "test_retest_flip_rate":
                flip_rate,
            "results": results
        }

        os.makedirs(
            "reports",
            exist_ok=True
        )

        with open(
            "reports/judge_validation.json",
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
        print("VALIDATION COMPLETE")
        print("=" * 60)

        print(
            f"Sample size: "
            f"{total}"
        )

        print(
            f"Gold agreement: "
            f"{gold_agreement_rate:.1%}"
        )

        print(
            f"Test-retest consistency: "
            f"{test_retest_rate:.1%}"
        )

        print(
            f"Test-retest flip rate: "
            f"{flip_rate:.1%}"
        )

        print()
        print(
            "Saved: reports/judge_validation.json"
        )


if __name__ == "__main__":
    validator = JudgeValidator()
    validator.run()