import argparse
import json
import csv
from trace import trace


def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM agent dialogues.")
    parser.add_argument('--eval_llm', default='gpt-4.1', 
                        choices=['gpt-4.1', 'o3-mini', 'claude', ...],
                        help='LLM model to use for evaluation.')
    parser.add_argument('--input_files', nargs='+', required=True,
                        help='List of input JSON file paths to evaluate.')
    parser.add_argument('--output_file', default='evaluation_results.csv',
                        help='Path to save the CSV output file.')
    
    args = parser.parse_args()

    with open(args.output_file, 'w', newline='') as csvfile:
        fieldnames = ['file', 'item_key', 'hallucination_score', 'inefficiency_score', 'adaptivity_score', 'correctness']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for file_path in args.input_files:
            print(f"Processing file: {file_path}...")
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)

                for key, dialog_data in data.items():
                    try:
                        results = trace(dialog_data, args.eval_llm)
                        
                        writer.writerow({
                            'file': file_path,
                            'item_key': key,
                            **results
                        })
                    except Exception as e:
                        print(f"  - Error processing item '{key}': {e}")
                        continue
            except Exception as e:
                print(f"Error reading or processing file {file_path}: {e}")
                continue
    
    print(f"\nEvaluation finished. Results saved to {args.output_file}")

if __name__ == '__main__':
    main()