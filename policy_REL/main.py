import numpy as np
import json
import datetime
from policy.BestFit import best_fit_policy
from policy.FirstFit import first_fit_policy
from policy.Greedy import greedy_policy
from evaluate import evaluate_solution ,print_evaluation ,compare_algorithms
from visual import visualize_comparison ,visualize_stock_heatmap ,visualize_stocks

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

def main():
    # Load data from JSON file
    observation_ff, info_ff = load_data("policy_REL\input_data.json")
    observation_bf, info_bf = load_data("policy_REL\input_data.json")
    observation_greedy, info_greedy = load_data("policy_REL\input_data.json")

    # Run algorithms
    print("Running First-Fit algorithm...")
    first_fit_actions = first_fit_policy(observation_ff, info_ff)  
    visualize_stocks(observation_ff["stocks"], "First-Fit Algorithm")
    ff_metrics = evaluate_solution(observation_ff)
    print_evaluation(ff_metrics, "First-Fit")

    print("\nRunning Best-Fit algorithm...")
    best_fit_actions = best_fit_policy(observation_bf, info_bf)  
    visualize_stocks(observation_bf["stocks"], "Best-Fit Algorithm")
    bf_metrics = evaluate_solution(observation_bf)
    print_evaluation(bf_metrics, "Best-Fit")

    print("\nRunning Greedy algorithm...")
    greedy_actions = greedy_policy(observation_greedy, info_greedy)
    visualize_stocks(observation_greedy["stocks"], "Greedy Algorithm")
    greedy_metrics = evaluate_solution(observation_greedy)
    print_evaluation(greedy_metrics, "Greedy")
    
    # Compare algorithms
    compare_algorithms(ff_metrics, bf_metrics, greedy_metrics)

    # Create comparison charts
    visualize_comparison(ff_metrics, bf_metrics, greedy_metrics)
    
    # Display heatmap for best result
    best_algorithm = "First-Fit"
    best_stocks = observation_ff["stocks"]
    
    if bf_metrics['utilization_rate'] > ff_metrics['utilization_rate'] and bf_metrics['utilization_rate'] > greedy_metrics['utilization_rate']:
        best_algorithm = "Best-Fit"
        best_stocks = observation_bf["stocks"]
    elif greedy_metrics['utilization_rate'] > ff_metrics['utilization_rate'] and greedy_metrics['utilization_rate'] > bf_metrics['utilization_rate']:
        best_algorithm = "Greedy"
        best_stocks = observation_greedy["stocks"]
    
    print(f"\nDisplaying heatmap for best algorithm ({best_algorithm}):")
    for i, stock in enumerate(best_stocks):
        visualize_stock_heatmap(stock, f"{best_algorithm} - Stock {i} Heatmap")
    
    # Save results to file
    save_results(ff_metrics, bf_metrics, greedy_metrics)

def save_results(ff_metrics, bf_metrics, greedy_metrics):
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
    
    results = {
        "first_fit": ff_metrics_converted,
        "best_fit": bf_metrics_converted,
        "greedy": greedy_metrics_converted,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=4)
    
    print("\nEvaluation results saved to 'evaluation_results.json'")
if __name__ == "__main__":
    main()