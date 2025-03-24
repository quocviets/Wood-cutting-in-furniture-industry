# main.py

import numpy as np
import json
import datetime
import time
import os
import matplotlib.pyplot as plt
from policy.BestFit import best_fit_policy
from policy.FirstFit import first_fit_policy
from policy.Greedy import greedy_policy
from utils.evaluate import evaluate_solution, print_evaluation, compare_algorithms
from utils.visual import visualize_comparison, visualize_stock_heatmap, visualize_stocks, create_radar_chart

def load_data(filepath):
    """
    Load data from JSON file
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Convert JSON data to observation format
    observation = {
        "products": data["products"],
        "stocks": []
    }
    
    # Create numpy arrays for stocks
    for stock in data["stocks"]:
        h, w = stock["size"]
        observation["stocks"].append(np.full((h, w), -1))
    
    return observation, {}

def run_algorithm(algorithm_func, algorithm_name, observation, info):
    """
    Run algorithm and measure runtime
    """
    print(f"\nRunning {algorithm_name} algorithm...")
    start_time = time.time()
    actions = algorithm_func(observation, info)
    end_time = time.time()
    runtime = end_time - start_time
    
    # Evaluate solution
    metrics = evaluate_solution(observation, runtime)
    print_evaluation(metrics, algorithm_name)
    
    return metrics

def main():
    # Create results directory if it doesn't exist
    results_dir = "HoangTH_qe170011_submit_code/results"
    os.makedirs(results_dir, exist_ok=True)
    
    # Change the default save location for matplotlib figures
    original_savefig = plt.savefig
    
    def custom_savefig(fname, *args, **kwargs):
        # Extracting just the filename without path
        base_filename = os.path.basename(fname)
        # Joining with our desired output directory
        new_path = os.path.join(results_dir, base_filename)
        return original_savefig(new_path, *args, **kwargs)
    
    # Override plt.savefig with our custom function
    plt.savefig = custom_savefig
    
    # Load data from JSON file
    input_file = "HoangTH_qe170011_submit_code/data/data_4.json"
    print(f"Loading data from {input_file}...")
    
    try:
        observation_ff, info_ff = load_data(input_file)
        observation_bf, info_bf = load_data(input_file)
        observation_greedy, info_greedy = load_data(input_file)
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        print("Please check the file path and try again.")
        return
    except json.JSONDecodeError:
        print("Error: Invalid JSON format in input file.")
        return
    
    # Run each algorithm and collect metrics
    ff_metrics = run_algorithm(first_fit_policy, "First-Fit", observation_ff, info_ff)
    visualize_stocks(observation_ff["stocks"], "First-Fit Algorithm")
    
    bf_metrics = run_algorithm(best_fit_policy, "Best-Fit", observation_bf, info_bf)
    visualize_stocks(observation_bf["stocks"], "Best-Fit Algorithm")
    
    greedy_metrics = run_algorithm(greedy_policy, "Greedy", observation_greedy, info_greedy)
    visualize_stocks(observation_greedy["stocks"], "Greedy Algorithm")
    
    # Compare algorithms
    compare_algorithms(ff_metrics, bf_metrics, greedy_metrics)

    # Create comparison charts
    print("\nGenerating comparison visualizations...")
    visualize_comparison(ff_metrics, bf_metrics, greedy_metrics)
    create_radar_chart(ff_metrics, bf_metrics, greedy_metrics)
    
    # Find best algorithm based on fitness score
    algorithms = [
        ("First-Fit", ff_metrics, observation_ff["stocks"]),
        ("Best-Fit", bf_metrics, observation_bf["stocks"]),
        ("Greedy", greedy_metrics, observation_greedy["stocks"])
    ]
    
    best_algorithm, best_metrics, best_stocks = max(
        algorithms, 
        key=lambda x: x[1]['fitness_score']
    )
    
    print(f"\nBest algorithm based on fitness score: {best_algorithm} ({best_metrics['fitness_score']:.2f}/100)")
    
    # Display heatmap for best result
    print(f"\nDisplaying heatmap for best algorithm ({best_algorithm}):")
    for i, stock in enumerate(best_stocks):
        visualize_stock_heatmap(stock, f"{best_algorithm} - Stock {i} Heatmap")
    
    # Restore original savefig function
    plt.savefig = original_savefig
    
    # Save results to file
    save_results(ff_metrics, bf_metrics, greedy_metrics, results_dir)

def save_results(ff_metrics, bf_metrics, greedy_metrics, output_dir):
    """
    Save evaluation results to JSON file with NumPy type conversion
    """
    # Function to convert NumPy types to Python native types
    def convert_numpy_types(obj):
        if isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return convert_numpy_types(obj.tolist())
        else:
            return obj
    
    # Convert all metrics to Python native types
    ff_metrics_converted = convert_numpy_types(ff_metrics)
    bf_metrics_converted = convert_numpy_types(bf_metrics)
    greedy_metrics_converted = convert_numpy_types(greedy_metrics)
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    results = {
        "first_fit": ff_metrics_converted,
        "best_fit": bf_metrics_converted,
        "greedy": greedy_metrics_converted,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Save to results directory with timestamp
    results_file = os.path.join(output_dir, f"evaluation_results_{timestamp}.json")
    
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"\nEvaluation results saved to '{results_file}'")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()