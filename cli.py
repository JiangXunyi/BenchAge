import argparse
from src.runners import simpleqa # add more runners as needed

def run():
    parser = argparse.ArgumentParser(description="Run factuality benchmark.")
    parser.add_argument("--dataset", required=True, choices=["simpleqa"], help="Which dataset to benchmark?")
    parser.add_argument("--output", default="data/results.csv", help="Path to save results.")
    
    args = parser.parse_args()

    if args.dataset == "simpleqa":
        simpleqa.run(output_path=args.output)

if __name__ == "__main__":
    run()