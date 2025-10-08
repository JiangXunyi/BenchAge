# When Benchmarks Age: Temporal Misalignment through Large Language Model Factuality Evaluation

This repository contains the code for the paper "When Benchmarks Age: Temporal Misalignment through Large Language Model Factuality Evaluation".

![image](assets/figures/ARROct-Benchmark.jpg)
## Setup
To setup package and dependencies in conda env, run:
```python
conda env create -f environment.yml -n BenchAge
```
and to activate the created environment, run:
```python
conda activate BenchAge
```
## Data
As for time-sensitive detection, store your data under `assets/data/<DATASET_NAME>`, for example, `assets/data/TruthfulQA`. The data format is json file, containing `question` as the key. For example:
```json
[
    {
        "question": "Who is the current president of the United States?"
    }
]
```


## Run
### Step1: Extract time-sensitive questions
If you want to deploy your own model, you can first run:
```bash
vllm serve \
  <YOUR_MODEL_PATH> \
  --served-model-name <YOUR_MODEL_NAME> \
  --port 9090
```
Otherwise, you can use OpenAI API directly.
Then you can run the following command to extract time-sensitive questions:
```python
python -m src.detector.detector \
--file_path <YOUR_FILE_PATH> \
--mode majority_vote \
--output_path <YOUR_OUTPUT_PATH> \
--temperature 1 \
--rounds 3 \
--use_vllm \
--vllm_base_url http://localhost:9090/v1 \
--model_name <YOUR_MODEL_NAME>
```

The results will contain the following files:

**For single mode:**
- `<run_id>/TruthfulQA_response.json` - All processed questions with GPT reasoning, answer, and confidence
- `<run_id>/TruthfulQA_time_sensitive.json` - Questions classified as time-sensitive
- `<run_id>/TruthfulQA_time_not_sensitive.json` - Questions classified as not time-sensitive
- `<run_id>/TruthfulQA_all_gpt_response.json` - Complete raw LLM responses with usage statistics

**For majority_vote mode:**
- `majority_vote_<timestamp>/run_<i>_<uuid>/` - Individual run results (same structure as single mode)
- `majority_vote_<timestamp>/majority_vote_results.json` - Final majority vote decisions for each question
- `majority_vote_<timestamp>/config.json` - Configuration used for the majority vote run


### Step2: Web search run
Before you run the web search, you need to set the [brave search API](https://brave.com/search/api/) and also [Google search API](https://developers.google.com/custom-search/v1/overview).

After you set the API keys, you can run the web search.


To run all the web search task, you can use the bash script provided in the `scripts` directory, you can run this command line under the root directory of the project:
```bash
bash scripts/run_web_search.sh
```
Then the search results and log will automatically be saved in the `data/` directory under the corresponding dataset folder.

To run the web search, you can use the following command:
```python
python -m src.runners.websearch TruthfulQA --model gpt-4o-mini-2024-07-18
```

### Step 3: Evaluator
To run the evaluator, you need to change the open source model results to yours then run the following command line:
```bash
python -m src.evaluator.evaluator <dataset_name>
```
or you can run the following bash script
```bash
bash scripts/run_evaluator.sh
```