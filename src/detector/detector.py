from tqdm import tqdm
import json
from openai import OpenAI
from src.detector.llmjudgeall import check_time_sensitivity
import os
import argparse
from datetime import datetime
import importlib
import concurrent.futures
import uuid
from typing import List, Dict, Any
from pathlib import Path

# client = OpenAI()
import time
import requests

def wait_for_model_ready(base_url: str, model_name: str, timeout: int = 300):
    print(f"Waiting for model '{model_name}' to be ready...")
    url = f"{base_url}/models"
    for _ in range(timeout):
        try:
            resp = requests.get(url).json()
            models = [m["id"] for m in resp.get("data", [])]
            if model_name in models:
                print(f"Model '{model_name}' is now ready.")
                return
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError(f"Timeout: Model '{model_name}' not ready after {timeout} seconds.")

def run_single(args: Dict[str, Any]) -> str:
    """Run a single instance with given parameters"""
    client = args["client"]
    file_path = args["file_path"]
    temperature = args["temperature"]
    output_path = args["output_path"]
    run_id = args.get("run_id")  # Get run_id if provided
    checkpoint_interval = args.get("checkpoint_interval", 10)
    disable_checkpoint = args.get("disable_checkpoint", False)
    
    # Load prompt configuration directly
    from src.detector.prompts.prompt_config import FEWSHOT_EXAMPLES, PROMPT_TEMPLATE
    fewshot_examples, prompt_template = FEWSHOT_EXAMPLES, PROMPT_TEMPLATE
    
    with open(file_path, 'r') as f:
        data = json.load(f)

    response_all_info = []
    response_time_sensitive = []
    response_not_time = []
    response = []

    cnt_time_sensitive = 0
    token_input = 0
    token_output = 0

    # Checkpoint file path
    checkpoint_file = f"{output_path}/checkpoint_{run_id}.json"
    
    # Load checkpoint if exists and checkpoint is not disabled
    start_index = 0
    if not disable_checkpoint and os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
                start_index = checkpoint.get('last_processed_index', 0)
                response_all_info = checkpoint.get('response_all_info', [])
                response_time_sensitive = checkpoint.get('response_time_sensitive', [])
                response_not_time = checkpoint.get('response_not_time', [])
                response = checkpoint.get('response', [])
                cnt_time_sensitive = checkpoint.get('cnt_time_sensitive', 0)
                token_input = checkpoint.get('token_input', 0)
                token_output = checkpoint.get('token_output', 0)
                print(f"Resuming from index {start_index} with {len(response)} processed items")
        except Exception as e:
            print(f"Error loading checkpoint: {e}, starting from beginning")

    if "gpt" in args["model_name"]:
        for i, item in enumerate(tqdm(data[start_index:], initial=start_index, total=len(data))):
            question = item["question"]
            try:
                raw = check_time_sensitivity(question, client, fewshot_examples, prompt_template, temperature, model_name=args["model_name"])
                response_info = raw.model_dump()
                response_all_info.append(response_info)
                raw = raw.output[0].content[0].text
                try:
                    parsed = json.loads(raw)
                    reason = parsed["reasoning"]
                    answer = parsed["answer"]
                    confidence = parsed["confidence"]
                except json.JSONDecodeError:
                    reason = raw.strip()
                    answer = "No"
                    confidence = 0.0
                    
                item["gpt_reason"] = reason
                item["gpt_answer"] = answer
                item["gpt_confidence"] = confidence
                response.append(item)
                
                # Access usage from the dictionary
                token_input += response_info["usage"]["input_tokens"]
                token_output += response_info["usage"]["output_tokens"]
                
                if answer == "Yes":
                    cnt_time_sensitive += 1
                    response_time_sensitive.append(item)
                else:
                    response_not_time.append(item)
                    
                # Save checkpoint every N items if not disabled
                if not disable_checkpoint and (i + 1) % checkpoint_interval == 0:
                    checkpoint_data = {
                        'last_processed_index': i + 1,
                        'response_all_info': response_all_info,
                        'response_time_sensitive': response_time_sensitive,
                        'response_not_time': response_not_time,
                        'response': response,
                        'cnt_time_sensitive': cnt_time_sensitive,
                        'token_input': token_input,
                        'token_output': token_output
                    }
                    with open(checkpoint_file, 'w') as f:
                        json.dump(checkpoint_data, f, indent=4)
                        
            except Exception as e:
                print(f"Error processing item {i}: {e}")
                # Save checkpoint on error if not disabled
                if not disable_checkpoint:
                    checkpoint_data = {
                        'last_processed_index': i,
                        'response_all_info': response_all_info,
                        'response_time_sensitive': response_time_sensitive,
                        'response_not_time': response_not_time,
                        'response': response,
                        'cnt_time_sensitive': cnt_time_sensitive,
                        'token_input': token_input,
                        'token_output': token_output
                    }
                    with open(checkpoint_file, 'w') as f:
                        json.dump(checkpoint_data, f, indent=4)
                raise e

    elif "qwen" in args["model_name"]:
        for i, item in enumerate(tqdm(data[start_index:], initial=start_index, total=len(data))):
            question = item["question"]
            try:
                raw = check_time_sensitivity(question, client, fewshot_examples, prompt_template, temperature, model_name=args["model_name"])
                response_info = raw.model_dump()
                response_all_info.append(response_info)
                raw = raw.choices[0].message.content
                try:
                    parsed = json.loads(raw)
                    reason = parsed["reasoning"]
                    answer = parsed["answer"]
                    confidence = parsed["confidence"]
                except json.JSONDecodeError:
                    reason = raw.strip()
                    answer = "No"
                    confidence = 0.0
                    
                item["gpt_reason"] = reason
                item["gpt_answer"] = answer
                item["gpt_confidence"] = confidence
                response.append(item)
                
                # Access usage from the dictionary
                token_input += response_info["usage"]["prompt_tokens"]
                token_output += response_info["usage"]["completion_tokens"]
                
                if answer == "Yes":
                    cnt_time_sensitive += 1
                    response_time_sensitive.append(item)
                else:
                    response_not_time.append(item)
                    
                # Save checkpoint every N items if not disabled
                if not disable_checkpoint and (i + 1) % checkpoint_interval == 0:
                    checkpoint_data = {
                        'last_processed_index': i + 1,
                        'response_all_info': response_all_info,
                        'response_time_sensitive': response_time_sensitive,
                        'response_not_time': response_not_time,
                        'response': response,
                        'cnt_time_sensitive': cnt_time_sensitive,
                        'token_input': token_input,
                        'token_output': token_output
                    }
                    with open(checkpoint_file, 'w') as f:
                        json.dump(checkpoint_data, f, indent=4)
                        
            except Exception as e:
                print(f"Error processing item {i}: {e}")
                # Save checkpoint on error if not disabled
                if not disable_checkpoint:
                    checkpoint_data = {
                        'last_processed_index': i,
                        'response_all_info': response_all_info,
                        'response_time_sensitive': response_time_sensitive,
                        'response_not_time': response_not_time,
                        'response': response,
                        'cnt_time_sensitive': cnt_time_sensitive,
                        'token_input': token_input,
                        'token_output': token_output
                    }
                    with open(checkpoint_file, 'w') as f:
                        json.dump(checkpoint_data, f, indent=4)
                raise e

    print(f"Time sensitive ratio: {cnt_time_sensitive / len(data)}")
    print(f"The input token number is: {token_input}, output token number is {token_output}, overall token number is: {token_input+token_output}. Used credit is: {token_output*1.100*10e-6 + token_output*4.400*10e-6}")
    
    # Clean up checkpoint file after successful completion
    if not disable_checkpoint and os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        
    return save_results(response_all_info, response_time_sensitive, response_not_time, response, output_path, run_id)

def majority_vote(question_filepath: str, output_path: str, rounds: int, temperature: float, client=None, model_name: str = "qwen"):
    """
    Perform majority voting across multiple runs with same prompts using multithreading
    
    Args:
        question_filepath: Path to the question file
        output_path: Base output path
        rounds: Number of rounds to run
        temperature: Temperature for model generation
        client: OpenAI client instance
        model_name: Name of the model to use
    """
    # Create root folder with timestamp for this majority vote run
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    root_folder = f"{output_path}/majority_vote_{current_time}"
    os.makedirs(root_folder, exist_ok=True)

    
    # Checkpoint file for majority vote
    mv_checkpoint_file = f"{root_folder}/majority_vote_checkpoint.json"
    
    # Load majority vote checkpoint if exists
    completed_runs = []
    if os.path.exists(mv_checkpoint_file):
        try:
            with open(mv_checkpoint_file, 'r') as f:
                mv_checkpoint = json.load(f)
                completed_runs = mv_checkpoint.get('completed_runs', [])
                print(f"Found checkpoint with {len(completed_runs)} completed runs")
        except Exception as e:
            print(f"Error loading majority vote checkpoint: {e}")
    
    # Save run configuration
    config = {
        "timestamp": current_time,
        "rounds": rounds,
        "temperature": temperature,
        "question_file": question_filepath,
        "model_name": model_name
    }
    with open(f"{root_folder}/config.json", "w") as f:
        json.dump(config, f, indent=4)
    
    # Prepare arguments for each thread (only for incomplete runs)
    thread_args = []
    for i in range(rounds):
        run_id = f"run_{i+1}_{str(uuid.uuid4())[:8]}"
        # Skip if this run is already completed
        if any(run_id in run for run in completed_runs):
            continue
        thread_args.append({
            "file_path": question_filepath,
            "temperature": temperature,
            "output_path": root_folder,
            "run_id": run_id,
            "client": client,
            "model_name": model_name,
            "checkpoint_interval": 10,  # Default checkpoint interval
            "disable_checkpoint": False  # Enable checkpoint by default
        })
    
    if not thread_args:
        print("All runs already completed!")
        result_paths = [run['result_path'] for run in completed_runs]
    else:
        # Run multiple instances in parallel
        result_paths = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(thread_args), 5)) as executor:
            # Submit all tasks
            future_to_args = {executor.submit(run_single, args): args for args in thread_args}
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_args):
                try:
                    result_path = future.result()
                    result_paths.append(result_path)
                    # Update checkpoint
                    completed_runs.append({
                        'run_id': future_to_args[future]['run_id'],
                        'result_path': result_path
                    })
                    with open(mv_checkpoint_file, 'w') as f:
                        json.dump({'completed_runs': completed_runs}, f, indent=4)
                except Exception as e:
                    print(f"Error in thread: {e}")
    
    # Create a dictionary to store votes for each question
    votes = {}
    for result_path in result_paths:
        with open(f"{result_path}/TruthfulQA_response.json", 'r') as f:
            results = json.load(f)
            for item in results:
                question = item["question"]
                if question not in votes:
                    votes[question] = {"Yes": 0, "No": 0}
                votes[question][item["gpt_answer"]] += 1
    
    # Calculate final decisions based on majority vote
    final_results = []
    for question, vote_counts in votes.items():
        final_answer = "Yes" if vote_counts["Yes"] > vote_counts["No"] else "No"
        final_results.append({
            "question": question,
            "final_answer": final_answer,
            "yes_votes": vote_counts["Yes"],
            "no_votes": vote_counts["No"],
            "total_votes": len(result_paths)
        })
    
    # Save majority vote results in the root folder
    with open(f"{root_folder}/majority_vote_results.json", "w") as f:
        json.dump(final_results, f, indent=4)
    
    # Clean up checkpoint file after successful completion
    if os.path.exists(mv_checkpoint_file):
        os.remove(mv_checkpoint_file)
    
    return final_results, root_folder

def save_results(response_all_info, response_time_sensitive, response_not_time, response, output_path, run_id=None):
    """Helper function to save results with unique folder name
    
    Args:
        response_all_info: All response information
        response_time_sensitive: Time sensitive responses
        response_not_time: Non-time sensitive responses
        response: Processed responses
        output_path: Base output path
        run_id: Optional UUID for this specific run. If None, a new one will be generated.
    """
    if run_id is None:
        run_id = str(uuid.uuid4())[:8]
    
    folder_name = f"{output_path}/{run_id}"
    os.makedirs(folder_name, exist_ok=True)

    with open(f"{folder_name}/TruthfulQA_time_sensitive.json", "w") as f:
        json.dump(response_time_sensitive, f, indent=4)

    with open(f"{folder_name}/TruthfulQA_all_gpt_response.json", "w") as f:
        json.dump(response_all_info, f, indent=4)
        
    with open(f"{folder_name}/TruthfulQA_time_not_sensitive.json", "w") as f:
        json.dump(response_not_time, f, indent=4)

    with open(f"{folder_name}/TruthfulQA_response.json", "w") as f:
        json.dump(response, f, indent=4)

    return folder_name

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check time sensitivity of questions.")
    parser.add_argument("--file_path", type=str, default="None", help="Path to the file containing questions.")
    parser.add_argument("--mode", type=str, choices=["single", "majority_vote"], default="single",
                      help="Mode to run the script in")
    parser.add_argument("--rounds", type=int, default=3, help="Number of rounds for majority vote")
    parser.add_argument("--output_path", type=str, default="")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--use_vllm", action="store_true", help="Use vLLM to run the model")
    parser.add_argument("--vllm_base_url", type=str, default="http://localhost:8000/v1", help="Base URL for vLLM server")
    parser.add_argument("--model_name", type=str, default="qwen", help="Model name to use")
    parser.add_argument("--checkpoint_interval", type=int, default=100, help="Save checkpoint every N items (default: 100)")
    parser.add_argument("--disable_checkpoint", action="store_true", help="Disable checkpoint functionality")
    args = parser.parse_args()

    if args.use_vllm:
        client = OpenAI(
            api_key="EMPTY",
            base_url=args.vllm_base_url,
        )
        wait_for_model_ready(args.vllm_base_url, args.model_name)
    else:
        client = OpenAI()

    if args.mode == "majority_vote":
        results, root_folder = majority_vote(args.file_path, args.output_path, args.rounds, args.temperature, client=client, model_name=args.model_name)
        print(f"Majority vote results saved in: {root_folder}")
    else:  # single
        run_single({
            "file_path": args.file_path,
            "temperature": args.temperature,
            "output_path": args.output_path,
            "client": client,
            "model_name": args.model_name,
            "checkpoint_interval": args.checkpoint_interval,
            "disable_checkpoint": args.disable_checkpoint
        })
