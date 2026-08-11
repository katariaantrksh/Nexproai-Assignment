\# LLM-as-Judge Evaluation Pipeline



\## Overview



This project implements an LLM-as-Judge evaluation pipeline for measuring

the quality of model-generated responses.



The pipeline:



1\. Loads a JSON test suite.

2\. Constructs a structured judging prompt.

3\. Sends each case to an LLM judge.

4\. Parses a structured JSON verdict.

5\. Records per-criterion scores and rationale.

6\. Aggregates results into a suite-level report.

7\. Logs judge prompts and raw responses for auditability.

8\. Measures judge latency and token usage.

9\. Runs controlled bias experiments.

10\. Validates judge consistency.

11\. Compares two judge prompt configurations.



The main judging mode is pointwise scoring.



\---



\## Project Structure



```text

llm\_judge\_takehome/

│

├── data/

│   └── test\_suite.json

│

├── judge\_app/

│   ├── \_\_init\_\_.py

│   ├── judge.py

│   ├── parser.py

│   ├── report.py

│   ├── cli.py

│   ├── bias.py

│   ├── validation.py

│   └── ab\_test.py

│

├── reports/

│   ├── judge\_results.json

│   ├── position\_bias.json

│   ├── verbosity\_bias.json

│   ├── bias\_experiments.json

│   ├── judge\_validation.json

│   ├── ab\_comparison.json

│   └── logs/

│       ├── case\_01.json

│       ├── ...

│       └── case\_10.json

│

├── .env

├── .gitignore

├── requirements.txt

└── README.md

