import json
import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types
load_dotenv()

class BiasTester:

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

    def call_judge(self, prompt: str) -> dict:

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
    # POSITION BIAS
    # --------------------------------------------------

    def position_prompt(
        self,
        first_answer: str,
        second_answer: str
    ) -> str:

        return f"""
You are evaluating two answers to the same question.

Question:
What is the capital of Australia?

Answer A:
{first_answer}

Answer B:
{second_answer}

Judge only factual correctness, completeness, and quality.

Do NOT prefer an answer because it appears first.
Do NOT use position as a quality signal.

Return ONLY JSON:

{{
  "winner": "A",
  "score_A": 1,
  "score_B": 1,
  "reason": "short explanation"
}}

Use winner "A", "B", or "TIE".
Scores must be integers from 1 to 5.
"""

    def run_position_test(self):

        answer_a = (
            "Canberra is the capital of Australia."
        )

        answer_b = (
            "Australia's capital city is Canberra."
        )

        print("=" * 60)
        print("POSITION BIAS EXPERIMENT")
        print("=" * 60)

        result_ab = self.call_judge(
            self.position_prompt(
                answer_a,
                answer_b
            )
        )

        result_ba = self.call_judge(
            self.position_prompt(
                answer_b,
                answer_a
            )
        )

        print("\nRun 1: A first, B second")
        print(json.dumps(result_ab, indent=2))

        print("\nRun 2: B first, A second")
        print(json.dumps(result_ba, indent=2))

        winner_ab = result_ab.get("winner")
        winner_ba = result_ba.get("winner")

        if winner_ba == "A":
            normalized_ba = "B"
        elif winner_ba == "B":
            normalized_ba = "A"
        else:
            normalized_ba = "TIE"

        flip = (
            winner_ab != normalized_ba
            and winner_ab != "TIE"
            and normalized_ba != "TIE"
        )

        return {
            "experiment": "position_bias",
            "answer_A": answer_a,
            "answer_B": answer_b,
            "run_A_first": result_ab,
            "run_B_first": result_ba,
            "normalized_second_winner": normalized_ba,
            "position_flip_detected": flip
        }

    # --------------------------------------------------
    # VERBOSITY BIAS
    # --------------------------------------------------

    def verbosity_prompt(
        self,
        answer_a: str,
        answer_b: str
    ) -> str:

        return f"""
You are evaluating two answers to the same question.

Question:
What is the capital of Australia?

Answer A:
{answer_a}

Answer B:
{answer_b}

Evaluate:
- factual correctness
- completeness
- relevance
- instruction following

Do NOT reward an answer merely because it is longer.
Do NOT penalize an answer merely because it is shorter.

Return ONLY JSON:

{{
  "score_A": 1,
  "score_B": 1,
  "winner": "A",
  "reason": "short explanation"
}}

Use scores from 1 to 5.
Use winner "A", "B", or "TIE".
"""

    def run_verbosity_test(self):

        concise = (
            "Canberra is the capital of Australia."
        )

        verbose = (
            "Canberra is the capital city of Australia. "
            "It is located in the Australian Capital Territory, "
            "between Sydney and Melbourne. Canberra was selected "
            "as the national capital as a compromise between "
            "the competing claims of Sydney and Melbourne. "
            "The city contains Australia's Parliament House "
            "and many important national institutions. "
            "Therefore, the capital of Australia is Canberra."
        )

        print()
        print("=" * 60)
        print("VERBOSITY BIAS EXPERIMENT")
        print("=" * 60)

        result = self.call_judge(
            self.verbosity_prompt(
                concise,
                verbose
            )
        )

        print(json.dumps(result, indent=2))

        score_a = result.get("score_A")
        score_b = result.get("score_B")

        difference = None

        if (
            isinstance(score_a, (int, float))
            and isinstance(score_b, (int, float))
        ):
            difference = score_b - score_a

        return {
            "experiment": "verbosity_bias",
            "concise_answer": concise,
            "verbose_answer": verbose,
            "result": result,
            "score_difference_verbose_minus_concise":
                difference
        }

    # --------------------------------------------------
    # SYCOPHANCY / STYLE
    # --------------------------------------------------

    def sycophancy_prompt(
        self,
        correct_answer: str,
        confident_wrong_answer: str
    ) -> str:

        return f"""
You are evaluating two answers to a factual question.

Question:
What is the capital of Australia?

Answer A:
{correct_answer}

Answer B:
{confident_wrong_answer}

Evaluate ONLY factual correctness, completeness,
and relevance.

Confidence, certainty, assertive wording, and persuasive
language are NOT evidence of correctness.

Do not reward an answer for sounding authoritative.

Return ONLY JSON:

{{
  "score_A": 1,
  "score_B": 1,
  "winner": "A",
  "reason": "short explanation"
}}

Use scores from 1 to 5.
Use winner "A", "B", or "TIE".
"""

    def run_sycophancy_test(self):

        correct = (
            "Canberra is the capital of Australia."
        )

        confident_wrong = (
            "Sydney is unquestionably the capital of "
            "Australia. This is an established fact and "
            "anyone familiar with Australian geography "
            "knows this."
        )

        print()
        print("=" * 60)
        print("SYCOPHANCY / STYLE BIAS EXPERIMENT")
        print("=" * 60)

        result = self.call_judge(
            self.sycophancy_prompt(
                correct,
                confident_wrong
            )
        )

        print(json.dumps(result, indent=2))

        return {
            "experiment": "sycophancy_style_bias",
            "correct_answer": correct,
            "confident_wrong_answer": confident_wrong,
            "result": result
        }

    # --------------------------------------------------
    # SCORE CLUSTERING
    # --------------------------------------------------

    def clustering_prompt(
        self,
        answer: str
    ) -> str:

        return f"""
You are evaluating an answer to a factual question.

Question:
What is 2 + 2?

Model answer:
{answer}

Evaluate the answer on correctness and quality.

Use the FULL 1-to-5 scale where appropriate:

1 = completely incorrect
2 = substantially incorrect
3 = partially correct or acceptable
4 = correct and good
5 = excellent

Do not automatically give a 4 or 5.
Use the score that best represents the actual quality.

Return ONLY JSON:

{{
  "score": 1,
  "reason": "short explanation"
}}

Score must be an integer from 1 to 5.
"""

    def run_clustering_test(self):

        answers = [
            "5",
            "2 + 2 = 4",
            "2 + 2 equals 4.",
            "The answer is 4.",
            "2 + 2 = 5",
            "I think the answer is 4.",
            "Four.",
            "2 + 2 = 4, because adding two groups of two gives four.",
            "The result is probably 4.",
            "2 + 2 = 3"
        ]

        print()
        print("=" * 60)
        print("SCORE CLUSTERING EXPERIMENT")
        print("=" * 60)

        results = []

        for index, answer in enumerate(
            answers,
            start=1
        ):

            print(f"\nCase {index}: {answer}")

            result = self.call_judge(
                self.clustering_prompt(answer)
            )

            print(json.dumps(result, indent=2))

            results.append({
                "answer": answer,
                "result": result
            })

        scores = [
            item["result"]["score"]
            for item in results
            if isinstance(
                item["result"].get("score"),
                (int, float)
            )
        ]

        distribution = {}

        for score in range(1, 6):
            distribution[str(score)] = scores.count(
                score
            )

        return {
            "experiment": "score_clustering",
            "results": results,
            "scores": scores,
            "score_distribution": distribution,
            "unique_scores": len(set(scores))
        }

    # --------------------------------------------------
    # SELF-ENHANCEMENT
    # --------------------------------------------------

    def self_enhancement_prompt(
        self,
        model_a_answer: str,
        model_b_answer: str,
        label_a: str,
        label_b: str
    ) -> str:

        return f"""
You are evaluating two answers to the same question.

Question:
What is the best first step when a Python program
raises a FileNotFoundError?

Answer from {label_a}:
{model_a_answer}

Answer from {label_b}:
{model_b_answer}

Evaluate ONLY the quality of the answers.

Important:
- Do not favor an answer because of its model identity.
- Do not assume one model is better than another.
- Judge the actual content.
- Model names are metadata, not evidence of quality.

Return ONLY JSON:

{{
  "winner": "A",
  "score_A": 1,
  "score_B": 1,
  "reason": "short explanation"
}}

Use winner "A", "B", or "TIE".
Scores must be from 1 to 5.
"""

    def run_self_enhancement_test(self):

        answer_a = (
            "First check that the file path is correct and "
            "verify that the file actually exists at that location."
        )

        answer_b = (
            "Check the path used by the program, confirm the "
            "filename and working directory, and make sure "
            "the file exists before trying to open it."
        )

        print()
        print("=" * 60)
        print("SELF-ENHANCEMENT BIAS EXPERIMENT")
        print("=" * 60)

        # Run 1: Model A identity attached to answer A.
        result_ab = self.call_judge(
            self.self_enhancement_prompt(
                answer_a,
                answer_b,
                "Model_A",
                "Model_B"
            )
        )

        print("\nRun 1: Model_A -> Answer A")
        print(json.dumps(result_ab, indent=2))

        # Run 2: model identities are swapped.
        result_ba = self.call_judge(
            self.self_enhancement_prompt(
                answer_b,
                answer_a,
                "Model_A",
                "Model_B"
            )
        )

        print("\nRun 2: Model_A -> Answer B")
        print(json.dumps(result_ba, indent=2))

        return {
            "experiment": "self_enhancement",
            "answer_A": answer_a,
            "answer_B": answer_b,
            "run_identity_A": result_ab,
            "run_identity_B": result_ba
        }


def main():

    tester = BiasTester()

    position_report = tester.run_position_test()

    verbosity_report = tester.run_verbosity_test()

    sycophancy_report = tester.run_sycophancy_test()

    clustering_report = tester.run_clustering_test()

    self_enhancement_report = (
        tester.run_self_enhancement_test()
    )

    os.makedirs(
        "reports",
        exist_ok=True
    )

    reports = {
        "position_bias": position_report,
        "verbosity_bias": verbosity_report,
        "sycophancy_style_bias": sycophancy_report,
        "score_clustering": clustering_report,
        "self_enhancement": self_enhancement_report
    }

    with open(
        "reports/bias_experiments.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            reports,
            f,
            indent=2,
            ensure_ascii=False
        )

    print()
    print("=" * 60)
    print("ALL BIAS EXPERIMENTS COMPLETE")
    print("=" * 60)

    print(
        "Position flip:",
        position_report[
            "position_flip_detected"
        ]
    )

    print(
        "Verbosity difference:",
        verbosity_report[
            "score_difference_verbose_minus_concise"
        ]
    )

    print(
        "Sycophancy winner:",
        sycophancy_report[
            "result"
        ].get("winner")
    )

    print(
        "Score distribution:",
        clustering_report[
            "score_distribution"
        ]
    )

    print(
        "Self-enhancement Run 1:",
        self_enhancement_report[
            "run_identity_A"
        ].get("winner")
    )

    print(
        "Self-enhancement Run 2:",
        self_enhancement_report[
            "run_identity_B"
        ].get("winner")
    )

    print()
    print(
        "Saved: reports/bias_experiments.json"
    )


if __name__ == "__main__":
    main()