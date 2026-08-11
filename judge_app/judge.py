import json
import os
import time
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types


load_dotenv()


class LLMJudge:
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

    def build_prompt(self, case: dict[str, Any]) -> str:
        criteria = ", ".join(case["criteria"])

        return f"""
You are an impartial evaluator judging the quality of an LLM response.

Evaluate the response using ONLY the information provided below.

INPUT:
{case["input"]}

SYSTEM INSTRUCTION:
{case["system_prompt"]}

MODEL OUTPUT:
{case["model_output"]}

EXPECTED OUTPUT:
{case.get("expected_output", "Not provided")}

EVALUATION CRITERIA:
{criteria}

Scoring:
1 = very poor
2 = poor
3 = acceptable
4 = good
5 = excellent

Judge correctness, completeness, and instruction following.
Do not reward verbosity, confidence, or writing style unless those are
explicitly part of the criteria.

Return ONLY a JSON object with this structure:

{{
  "correctness": {{
    "score": 1,
    "rationale": "short explanation"
  }},
  "completeness": {{
    "score": 1,
    "rationale": "short explanation"
  }},
  "instruction_following": {{
    "score": 1,
    "rationale": "short explanation"
  }},
  "overall_score": 1,
  "pass": true,
  "overall_rationale": "short explanation"
}}

Use scores from 1 to 5.
The pass value should be true when the response is generally acceptable.
"""

    def judge_case(
        self,
        case: dict[str, Any]
    ) -> dict[str, Any]:

        prompt = self.build_prompt(case)

        start = time.perf_counter()

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )

        latency_ms = round(
            (time.perf_counter() - start) * 1000,
            2
        )

        raw_text = response.text or ""

        os.makedirs("reports/logs", exist_ok=True)

        with open(
            f"reports/logs/{case['id']}.json",
            "w",
            encoding="utf-8"
        ) as log_file:
            json.dump(
                {
                    "case": case,
                    "judge_model": self.model,
                    "prompt": prompt,
                    "raw_response": raw_text,
                },
                log_file,
                indent=2,
                ensure_ascii=False
            )

        usage = getattr(
            response,
            "usage_metadata",
            None
        )

        result = self.parse_json(raw_text)

        result["_meta"] = {
            "case_id": case["id"],
            "model": self.model,
            "latency_ms": latency_ms,
            "prompt_tokens": getattr(
                usage,
                "prompt_token_count",
                0
            ) or 0,
            "completion_tokens": getattr(
                usage,
                "candidates_token_count",
                0
            ) or 0,
            "total_tokens": getattr(
                usage,
                "total_token_count",
                0
            ) or 0
        }

        return result

    @staticmethod
    def parse_json(raw_text: str) -> dict[str, Any]:

        # First attempt: direct JSON
        try:
            return json.loads(raw_text)

        except json.JSONDecodeError:
            pass

        # Second attempt: extract JSON object
        start = raw_text.find("{")
        end = raw_text.rfind("}")

        if start != -1 and end > start:
            try:
                return json.loads(
                    raw_text[start:end + 1]
                )
            except json.JSONDecodeError:
                pass

        # Nothing worked
        raise ValueError(
            "Judge returned malformed JSON:\n"
            + raw_text
        )


if __name__ == "__main__":

    with open(
        "data/test_suite.json",
        "r",
        encoding="utf-8"
    ) as f:
        suite = json.load(f)

    judge = LLMJudge()

    first_case = suite["cases"][0]

    result = judge.judge_case(first_case)

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False
        )
    )