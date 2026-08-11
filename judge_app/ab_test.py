import json
import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types


load_dotenv()


class ABTester:

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

    def call_judge(self, prompt):
        time.sleep(5)
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

    # --------------------------------------------------
    # CONFIGURATION A
    # --------------------------------------------------

    def prompt_a(self, case):

        return f"""
Evaluate the following LLM response.

Question:
{case["input"]}

Model response:
{case["model_output"]}

Expected answer:
{case.get("expected_output", "Not provided")}

Give an overall quality score from 1 to 5.

Return ONLY JSON:

{{
  "score": 1,
  "pass": true,
  "reason": "short explanation"
}}

A score of 4 or 5 should normally pass.
"""

    # --------------------------------------------------
    # CONFIGURATION B
    # --------------------------------------------------

    def prompt_b(self, case):

        return f"""
You are a careful and impartial LLM evaluator.

Evaluate the model response using the following order:

1. Factual correctness
2. Completeness
3. Instruction following
4. Relevance

Question:
{case["input"]}

Model response:
{case["model_output"]}

Expected answer:
{case.get("expected_output", "Not provided")}

Important evaluation rules:

- Do NOT reward verbosity by itself.
- Do NOT reward confident or authoritative wording.
- Do NOT penalize a concise answer if it fully answers the question.
- Do NOT use answer position as evidence of quality.
- Do NOT assume the model is correct.
- Base the score on the actual content.
- A factual error should substantially reduce the score.

Scoring:

1 = completely incorrect
2 = mostly incorrect
3 = partially correct / mixed
4 = correct and good
5 = excellent

Return ONLY JSON:

{{
  "score": 1,
  "pass": true,
  "reason": "short explanation"
}}

Use pass=true only when the response is acceptable overall.
"""

    def evaluate_suite(self, suite, configuration):

        results = []

        for index, case in enumerate(
            suite["cases"],
            start=1
        ):

            print(
                f"Config {configuration} "
                f"[{index}/{len(suite['cases'])}] "
                f"{case['id']}"
            )

            if configuration == "A":
                prompt = self.prompt_a(case)
            else:
                prompt = self.prompt_b(case)

            result = self.call_judge(prompt)

            results.append({
                "case_id": case["id"],
                "score": result.get("score", 0),
                "pass": bool(result.get("pass", False)),
                "reason": result.get("reason", ""),
                "raw_result": result
            })

        return results

    @staticmethod
    def calculate_metrics(results):

        total = len(results)

        passed = sum(
            1
            for r in results
            if r["pass"]
        )

        mean_score = (
            sum(r["score"] for r in results)
            / total
        )

        pass_rate = (
            passed / total
        )

        return {
            "cases": total,
            "passed": passed,
            "pass_rate": pass_rate,
            "mean_score": mean_score
        }

    @staticmethod
    def pairwise_comparison(
        results_a,
        results_b
    ):

        a_wins = 0
        b_wins = 0
        ties = 0

        for a, b in zip(
            results_a,
            results_b
        ):

            if a["score"] > b["score"]:
                a_wins += 1

            elif b["score"] > a["score"]:
                b_wins += 1

            else:
                ties += 1

        total = len(results_a)

        return {
            "A_wins": a_wins,
            "B_wins": b_wins,
            "ties": ties,
            "A_win_rate": a_wins / total,
            "B_win_rate": b_wins / total,
            "tie_rate": ties / total
        }

    def run(self):

        with open(
            "data/test_suite.json",
            "r",
            encoding="utf-8"
        ) as f:

            suite = json.load(f)

        print("=" * 60)
        print("A/B JUDGE COMPARISON")
        print("=" * 60)

        print()
        print("Running Configuration A...")
        print()

        results_a = self.evaluate_suite(
            suite,
            "A"
        )

        print()
        print("Running Configuration B...")
        print()

        results_b = self.evaluate_suite(
            suite,
            "B"
        )

        metrics_a = self.calculate_metrics(
            results_a
        )

        metrics_b = self.calculate_metrics(
            results_b
        )

        pairwise = self.pairwise_comparison(
            results_a,
            results_b
        )

        # Winner based on mean score first,
        # then pass rate.
        if (
            metrics_a["mean_score"]
            > metrics_b["mean_score"]
        ):
            winner = "Configuration A"

        elif (
            metrics_b["mean_score"]
            > metrics_a["mean_score"]
        ):
            winner = "Configuration B"

        elif (
            metrics_a["pass_rate"]
            > metrics_b["pass_rate"]
        ):
            winner = "Configuration A"

        elif (
            metrics_b["pass_rate"]
            > metrics_a["pass_rate"]
        ):
            winner = "Configuration B"

        else:
            winner = "Tie"

        report = {
            "model": self.model,
            "configuration_A": {
                "description":
                    "Basic judge prompt",
                "metrics": metrics_a,
                "results": results_a
            },
            "configuration_B": {
                "description":
                    "Bias-aware structured judge prompt",
                "metrics": metrics_b,
                "results": results_b
            },
            "pairwise_comparison": pairwise,
            "winner": winner
        }

        os.makedirs(
            "reports",
            exist_ok=True
        )

        with open(
            "reports/ab_comparison.json",
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
        print("A/B COMPARISON COMPLETE")
        print("=" * 60)

        print()
        print("CONFIGURATION A")
        print(
            f"Pass rate: "
            f"{metrics_a['pass_rate']:.1%}"
        )
        print(
            f"Mean score: "
            f"{metrics_a['mean_score']:.2f}/5"
        )

        print()
        print("CONFIGURATION B")
        print(
            f"Pass rate: "
            f"{metrics_b['pass_rate']:.1%}"
        )
        print(
            f"Mean score: "
            f"{metrics_b['mean_score']:.2f}/5"
        )

        print()
        print("PAIRWISE")
        print(
            f"A wins: "
            f"{pairwise['A_wins']}"
        )
        print(
            f"B wins: "
            f"{pairwise['B_wins']}"
        )
        print(
            f"Ties: "
            f"{pairwise['ties']}"
        )

        print()
        print(
            f"A win rate: "
            f"{pairwise['A_win_rate']:.1%}"
        )

        print(
            f"B win rate: "
            f"{pairwise['B_win_rate']:.1%}"
        )

        print()
        print(
            f"WINNER: {winner}"
        )

        print()
        print(
            "Saved: reports/ab_comparison.json"
        )


if __name__ == "__main__":
    tester = ABTester()
    tester.run()