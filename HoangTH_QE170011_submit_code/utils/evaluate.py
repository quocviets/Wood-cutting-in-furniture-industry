import numpy as np
import time

def evaluate_solution(observation, runtime=None):
    """
    Evaluate solution based on multiple criteria including fitness, waste, and runtime
    """
    total_stocks = len(observation["stocks"])
    total_area = 0
    used_area = 0
    wasted_area = 0
    
    # Utilization rate by stock
    stock_utilization = []
    
    # Evaluate fragmentation
    total_fragmentation = 0
    
    for stock in observation["stocks"]:
        stock_height, stock_width = stock.shape
        stock_area = stock_height * stock_width
        total_area += stock_area
        
        # Count used cells
        used_cells = np.sum(stock != -1)
        used_area += used_cells
        
        # Count wasted cells
        wasted_cells = stock_area - used_cells
        wasted_area += wasted_cells
        
        # Utilization rate of this stock
        stock_util = (used_cells / stock_area * 100) if stock_area > 0 else 0
        stock_utilization.append(stock_util)
        
        # Calculate fragmentation
        # Use 4-connected method to count non-continuous regions
        visited = np.zeros_like(stock, dtype=bool)
        empty_regions = 0
        
        for i in range(stock_height):
            for j in range(stock_width):
                if stock[i, j] == -1 and not visited[i, j]:
                    # Found an unvisited empty cell
                    empty_regions += 1
                    # Use BFS to mark all connected cells
                    queue = [(i, j)]
                    visited[i, j] = True
                    
                    while queue:
                        x, y = queue.pop(0)
                        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                            nx, ny = x + dx, y + dy
                            if (0 <= nx < stock_height and 0 <= ny < stock_width and 
                                stock[nx, ny] == -1 and not visited[nx, ny]):
                                visited[nx, ny] = True
                                queue.append((nx, ny))
        
        total_fragmentation += empty_regions
    
    # Overall utilization rate
    utilization_rate = used_area / total_area * 100 if total_area > 0 else 0
    
    # Standard deviation of stock utilization (measure of balance)
    std_utilization = np.std(stock_utilization) if len(stock_utilization) > 0 else 0
    
    # Fragmentation rate
    fragmentation_rate = total_fragmentation / total_stocks if total_stocks > 0 else 0
    
    # Evaluate concentration of same type products
    product_dispersion = calculate_product_dispersion(observation["stocks"])
    
    # Calculate fitness score - weighted combination of metrics
    # Higher is better (0-100 scale)
    fitness_score = calculate_fitness_score(
        utilization_rate, 
        fragmentation_rate, 
        product_dispersion, 
        std_utilization
    )
    
    # Calculate waste ratio
    waste_ratio = wasted_area / total_area * 100 if total_area > 0 else 0
    
    return {
        "total_stocks": total_stocks,
        "total_area": total_area,
        "used_area": used_area,
        "wasted_area": wasted_area,
        "utilization_rate": utilization_rate,
        "waste_ratio": waste_ratio,
        "stock_utilization": stock_utilization,
        "utilization_std": std_utilization,
        "fragmentation_count": total_fragmentation,
        "fragmentation_rate": fragmentation_rate,
        "min_utilization": min(stock_utilization) if stock_utilization else 0,
        "max_utilization": max(stock_utilization) if stock_utilization else 0,
        "product_dispersion": product_dispersion,
        "fitness_score": fitness_score,
        "runtime": runtime  # in seconds, None if not measured
    }

def calculate_fitness_score(utilization_rate, fragmentation_rate, product_dispersion, std_utilization):
    """
    Calculate a composite fitness score (0-100)
    Higher is better
    
    Parameters:
    - utilization_rate: Percentage of area used (higher is better)
    - fragmentation_rate: Average number of empty regions per stock (lower is better)
    - product_dispersion: Average number of stocks each product is distributed across (lower is better)
    - std_utilization: Standard deviation of utilization rates (lower is better)
    """
    # Define weights for each component (sum should be 1.0)
    w_utilization = 0.6  # Utilization is most important
    w_fragmentation = 0.2  # Fragmentation is next most important
    w_dispersion = 0.1  # Product dispersion
    w_std = 0.1  # Even distribution
    
    # Normalize metrics to 0-100 scale (where 100 is best)
    util_score = utilization_rate  # Already 0-100
    
    # For fragmentation, assume 5 is worst case, 0 is best
    frag_score = max(0, 100 - (fragmentation_rate * 20))
    
    # For dispersion, assume 3 is worst case, 1 is best
    disp_score = max(0, 100 - ((product_dispersion - 1) * 50))
    
    # For std deviation, assume 30 is worst case, 0 is best
    std_score = max(0, 100 - (std_utilization * 100 / 30))
    
    # Calculate weighted sum
    fitness = (
        w_utilization * util_score +
        w_fragmentation * frag_score +
        w_dispersion * disp_score +
        w_std * std_score
    )
    
    return round(fitness, 2)

def calculate_product_dispersion(stocks):
    """
    Calculate dispersion of products of the same type
    Lower value is better (products of same type placed closer)
    """
    # Count number of stocks containing each product type
    product_counts = {}
    
    for stock in stocks:
        # Find all unique product IDs in this stock
        unique_products = set(np.unique(stock))
        
        # Remove -1 value (empty cell)
        if -1 in unique_products:
            unique_products.remove(-1)
        
        # Update counts
        for prod_id in unique_products:
            if prod_id not in product_counts:
                product_counts[prod_id] = 0
            product_counts[prod_id] += 1
    
    # Calculate average number of stocks each product is distributed across
    if not product_counts:
        return 0
    
    total_dispersion = sum(product_counts.values())
    num_products = len(product_counts)
    
    return total_dispersion / num_products

def print_evaluation(metrics, algorithm_name):
    """
    Print detailed evaluation for a solution
    """
    print(f"\nEvaluation for {algorithm_name}:")
    print(f"- Fitness Score: {metrics['fitness_score']:.2f}/100")
    print(f"- Number of stocks used: {metrics['total_stocks']}")
    print(f"- Total area: {metrics['total_area']}")
    print(f"- Used area: {metrics['used_area']} ({metrics['utilization_rate']:.2f}%)")
    print(f"- Wasted area: {metrics['wasted_area']} ({metrics['waste_ratio']:.2f}%)")
    print(f"- Runtime: {metrics['runtime']:.4f} seconds" if metrics['runtime'] is not None else "- Runtime: Not measured")
    print(f"- Standard deviation of utilization: {metrics['utilization_std']:.2f}")
    print(f"- Lowest/highest utilization: {metrics['min_utilization']:.2f}% / {metrics['max_utilization']:.2f}%")
    print(f"- Number of fragmented regions: {metrics['fragmentation_count']}")
    print(f"- Fragmentation rate per stock: {metrics['fragmentation_rate']:.2f}")
    print(f"- Product dispersion: {metrics['product_dispersion']:.2f}")
    
    # Detail by stock
    print("\nDetails by stock:")
    for i, util in enumerate(metrics['stock_utilization']):
        print(f"  Stock {i}: {util:.2f}% utilized")
        
def compare_algorithms(ff_metrics, bf_metrics, greedy_metrics):
    """
    Compare algorithms using tables and charts
    """
    # Create comparison table
    print("\nAlgorithm Comparison:")
    print("-" * 100)
    headers = ["Algorithm", "Fitness", "Stocks", "Utilization", "Waste", "Runtime", "Fragmentation"]
    print(f"{headers[0]:<15} {headers[1]:<10} {headers[2]:<10} {headers[3]:<15} {headers[4]:<10} {headers[5]:<15} {headers[6]:<15}")
    print("-" * 100)
    
    runtime_ff = f"{ff_metrics['runtime']:.4f}s" if ff_metrics['runtime'] is not None else "N/A"
    runtime_bf = f"{bf_metrics['runtime']:.4f}s" if bf_metrics['runtime'] is not None else "N/A"
    runtime_greedy = f"{greedy_metrics['runtime']:.4f}s" if greedy_metrics['runtime'] is not None else "N/A"
    
    print(f"{'First-Fit':<15} {ff_metrics['fitness_score']:<10.2f} {ff_metrics['total_stocks']:<10} "
          f"{ff_metrics['utilization_rate']:.2f}%{'':<8} "
          f"{ff_metrics['waste_ratio']:.2f}%{'':<4} "
          f"{runtime_ff:<15} "
          f"{ff_metrics['fragmentation_rate']:.2f}{'':<8}")
    
    print(f"{'Best-Fit':<15} {bf_metrics['fitness_score']:<10.2f} {bf_metrics['total_stocks']:<10} "
          f"{bf_metrics['utilization_rate']:.2f}%{'':<8} "
          f"{bf_metrics['waste_ratio']:.2f}%{'':<4} "
          f"{runtime_bf:<15} "
          f"{bf_metrics['fragmentation_rate']:.2f}{'':<8}")
    
    print(f"{'Greedy':<15} {greedy_metrics['fitness_score']:<10.2f} {greedy_metrics['total_stocks']:<10} "
          f"{greedy_metrics['utilization_rate']:.2f}%{'':<8} "
          f"{greedy_metrics['waste_ratio']:.2f}%{'':<4} "
          f"{runtime_greedy:<15} "
          f"{greedy_metrics['fragmentation_rate']:.2f}{'':<8}")
    
    # Determine best algorithm
    best_fitness = max(
        [ff_metrics['fitness_score'], bf_metrics['fitness_score'], greedy_metrics['fitness_score']]
    )
    best_utilization = max(
        [ff_metrics['utilization_rate'], bf_metrics['utilization_rate'], greedy_metrics['utilization_rate']]
    )
    lowest_waste = min(
        [ff_metrics['waste_ratio'], bf_metrics['waste_ratio'], greedy_metrics['waste_ratio']]
    )
    fastest_runtime = None
    if all(m['runtime'] is not None for m in [ff_metrics, bf_metrics, greedy_metrics]):
        fastest_runtime = min(
            [ff_metrics['runtime'], bf_metrics['runtime'], greedy_metrics['runtime']]
        )
    min_stocks = min(
        [ff_metrics['total_stocks'], bf_metrics['total_stocks'], greedy_metrics['total_stocks']]
    )
    
    print("\nOverall assessment:")
    if ff_metrics['fitness_score'] == best_fitness:
        print(f"- First-Fit has best overall fitness ({best_fitness:.2f}/100)")
    elif bf_metrics['fitness_score'] == best_fitness:
        print(f"- Best-Fit has best overall fitness ({best_fitness:.2f}/100)")
    else:
        print(f"- Greedy has best overall fitness ({best_fitness:.2f}/100)")
    
    if ff_metrics['utilization_rate'] == best_utilization:
        print(f"- First-Fit has best utilization rate ({best_utilization:.2f}%)")
    elif bf_metrics['utilization_rate'] == best_utilization:
        print(f"- Best-Fit has best utilization rate ({best_utilization:.2f}%)")
    else:
        print(f"- Greedy has best utilization rate ({best_utilization:.2f}%)")
    
    if ff_metrics['waste_ratio'] == lowest_waste:
        print(f"- First-Fit has lowest waste ({lowest_waste:.2f}%)")
    elif bf_metrics['waste_ratio'] == lowest_waste:
        print(f"- Best-Fit has lowest waste ({lowest_waste:.2f}%)")
    else:
        print(f"- Greedy has lowest waste ({lowest_waste:.2f}%)")
    
    if fastest_runtime is not None:
        if ff_metrics['runtime'] == fastest_runtime:
            print(f"- First-Fit has fastest runtime ({fastest_runtime:.4f}s)")
        elif bf_metrics['runtime'] == fastest_runtime:
            print(f"- Best-Fit has fastest runtime ({fastest_runtime:.4f}s)")
        else:
            print(f"- Greedy has fastest runtime ({fastest_runtime:.4f}s)")
    
    if ff_metrics['total_stocks'] == min_stocks:
        print(f"- First-Fit uses fewest stocks ({min_stocks})")
    elif bf_metrics['total_stocks'] == min_stocks:
        print(f"- Best-Fit uses fewest stocks ({min_stocks})")
    else:
        print(f"- Greedy uses fewest stocks ({min_stocks})")