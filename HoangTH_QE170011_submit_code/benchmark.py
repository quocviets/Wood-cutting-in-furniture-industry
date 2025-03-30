import json
import numpy as np
import time
import csv
from policy.BestFit import best_fit_policy
from policy.Greedy import greedy_policy
from policy.FirstFit import first_fit_policy

#  dùng file chứa nhiều data 1 lần
with open("HoangTH_qe170011_submit_code/data/formatted_orders.json", "r") as f:
    orders = json.load(f)

def calculate_waste_rate(stocks, stock_size):
    total_area = len(stocks) * stock_size[0] * stock_size[1]
    used_area = sum(np.count_nonzero(stock != -1) for stock in stocks)
    return (total_area - used_area) / total_area

def calculate_fitness(waste_rate):
    return 1 - waste_rate

def encode_product_id(product_id):
    """ Chuyển product_id thành số nguyên duy nhất """
    return hash(product_id) % (10**6)  

def run_algorithm(algorithm_name, algorithm, orders):
    results = []
    for order in orders:
        order_id = order["order_id"]
        stock_size = order["stock_size"]
        
        products = [
            {"id": encode_product_id(item_id), "size": item_data["size"], "quantity": item_data["quantity"]}
            for item_id, item_data in order["items"].items()
        ]
        
        stocks = [np.full(stock_size, -1)]
        
        observation = {"products": products, "stocks": stocks}
        info = {}
        
        start_time = time.time()
        actions = algorithm(observation, info)
        runtime = time.time() - start_time
        
        stock_count = len(observation["stocks"])
        waste_rate = calculate_waste_rate(observation["stocks"], stock_size)
        fitness = calculate_fitness(waste_rate)
        
        result = [algorithm_name, order_id, stock_count, waste_rate, fitness, runtime]
        results.append(result)
        print(result)  # In ra màn hình
    
    return results

def main():
    benchmark_results = []
    # phần này bạn có thể chọn thuật toán để thực hiện benchmark 
    # benchmark_results += run_algorithm("Best-Fit", best_fit_policy, orders)
    # benchmark_results += run_algorithm("First-Fit", first_fit_policy, orders)
    benchmark_results += run_algorithm("Greedy", greedy_policy, orders)
    
    with open("benchmark_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Algorithm", "Order ID", "Stock Count", "Waste Rate", "Fitness", "Runtime"])
        writer.writerows(benchmark_results)
    
if __name__ == "__main__":
    main()
