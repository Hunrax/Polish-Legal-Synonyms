import sys
import csv
from pathlib import Path
from datetime import datetime
import numpy as np

# Add the workspace root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from simple_bert_dbscan import run_workflow
from models.bert.bert import model_names


def run_batch_experiments(model_name, epsilons, output_csv=None):
    """
    Run BERT+DBSCAN workflow for all files with different epsilon values.
    
    Args:
        model_name: Name of the BERT model to use
        epsilons: List of epsilon values to test
        output_csv: Path to output CSV file (default: auto-generated)
    """
    
    # Validate model
    if model_name not in model_names:
        print(f"Error: Model '{model_name}' not found.")
        print("Available models:")
        for name in model_names:
            print(f"  - {name}")
        return
    
    # Find all PDF files in input directory
    input_dir = Path("input")
    if not input_dir.exists():
        print(f"Input directory not found: {input_dir}")
        return
    
    pdf_files = sorted(list(input_dir.glob("*.pdf")))
    if not pdf_files:
        print(f"No PDF files found in {input_dir}")
        return
    
    print(f"Found {len(pdf_files)} PDF files")
    print(f"Testing with {len(epsilons)} epsilon values: {epsilons}")
    print(f"Model: {model_name}\n")
    
    # Generate output CSV filename if not provided
    if output_csv is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        safe_model_name = model_name.replace("/", "_").replace(".", "_")
        output_csv = f"results_bert_dbscan_{safe_model_name}_{timestamp}.csv"
        output_csv = Path("results") / output_csv
    else:
        output_csv = Path(output_csv)
    
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    
    # Prepare CSV file
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow([
            'file', 'model', 'epsilon', 'average_score', 'median_score',
            'acceptance_rate', 'coverage', 'score_distribution'
        ])
        f.flush()
        
        # Run experiments
        total_runs = len(pdf_files) * len(epsilons)
        current_run = 0
        
        for pdf_file in pdf_files:
            pdf_filename = pdf_file.name
            print(f"\n{'='*60}")
            print(f"Processing file: {pdf_filename}")
            print(f"{'='*60}")
            
            for epsilon in epsilons:
                current_run += 1
                print(f"\n[{current_run}/{total_runs}] Running with epsilon={epsilon}")
                
                try:
                    metrics = run_workflow(pdf_filename, model_name, epsilon)
                    
                    if metrics is None:
                        print(f"Failed to run workflow for {pdf_filename} with epsilon={epsilon}")
                        continue
                    
                    # Extract metrics
                    avg_score = metrics.get('average_score', 0)
                    median_score = metrics.get('median_score', 0)
                    acceptance_rate = metrics.get('acceptance_rate', 0)
                    coverage = metrics.get('coverage', 0)
                    score_dist = metrics.get('score_distribution', {})
                    
                    # Write result to CSV
                    writer.writerow([
                        pdf_filename,
                        model_name,
                        round(epsilon, 1),
                        f"{avg_score:.4f}",
                        median_score,
                        f"{acceptance_rate:.4f}",
                        f"{coverage:.4f}",
                        str(score_dist)
                    ])
                    f.flush()
                    
                    print(f"✓ Saved results for {pdf_filename} (eps={epsilon})")
                    
                except Exception as e:
                    print(f"✗ Error processing {pdf_filename} with epsilon={epsilon}: {e}")
                    continue
    
    print(f"\n{'='*60}")
    print(f"Experiments completed!")
    print(f"Results saved to: {output_csv}")
    print(f"{'='*60}")


if __name__ == "__main__":
    print("Available models:")
    for i, name in enumerate(model_names, start=1):
        print(f"{i}. {name}")
    
    # If model_number is provided as argument, use it
    if len(sys.argv) >= 2:
        try:
            model_idx = int(sys.argv[1]) - 1
            if model_idx < 0 or model_idx >= len(model_names):
                print(f"Error: Model number must be between 1 and {len(model_names)}")
                sys.exit(1)
            model_name = model_names[model_idx]
        except ValueError:
            print("Error: Model number must be an integer")
            sys.exit(1)
    else:
        try:
            model_idx = int(input("\nEnter the number of your choice: ")) - 1
            if model_idx < 0 or model_idx >= len(model_names):
                print(f"Error: Model number must be between 1 and {len(model_names)}")
                sys.exit(1)
            model_name = model_names[model_idx]
        except ValueError:
            print("Error: Model number must be an integer")
            sys.exit(1)
    
    print(f"Selected model: {model_name}\n")
    
    # Generate epsilon values: from 0.2 to 1.0 with step 0.1
    epsilons = list(np.arange(0.2, 1.1, 0.1))
    print(f"Epsilon values: {[f'{e:.1f}' for e in epsilons]}")
    
    run_batch_experiments(model_name, epsilons)
