import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import datetime
import seaborn as sns

def visualize_stocks(stocks, title="Stock Visualization"):
    """
    Display stocks with maximum 4 stocks per row, automatically wrapping to new rows if needed
    """
    num_stocks = len(stocks)
    cols = min(4, num_stocks)  # Maximum 4 columns per row
    rows = math.ceil(num_stocks / 4)  # Required number of rows

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    fig.suptitle(title, fontsize=16)

    # If only one row and column, axes is not array so convert to list format
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]  # Convert to list containing 1 row
    elif cols == 1:
        axes = [[ax] for ax in axes]  # Convert to list containing single rows

    product_colors = {
        1: "#0000FF",  # Blue
        2: "#FFD700",  # Gold
        3: "#FF0000",  # Red
        4: "#b57edc",  # Purple
        5: "#7fffd4",  # Aquamarine
    }

    for i, stock in enumerate(stocks):
        row, col = divmod(i, 4)  # Determine position (row, column)
        ax = axes[row][col]

        ax.set_xticks(range(stock.shape[1] + 1))
        ax.set_yticks(range(stock.shape[0] + 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_title(f"Stock {i}")

        visited = np.full(stock.shape, False)  # Mark cells already drawn

        for x in range(stock.shape[0]):
            for y in range(stock.shape[1]):
                product_id = stock[x, y]

                if product_id >= 1 and not visited[x, y]:
                    width, height = 1, 1
                    while y + width < stock.shape[1] and stock[x, y + width] == product_id:
                        width += 1
                    while x + height < stock.shape[0] and np.all(stock[x:x + height, y:y + width] == product_id):
                        height += 1

                    visited[x:x + height, y:y + width] = True

                    color = product_colors.get(product_id, "gray")
                    rect = patches.Rectangle(
                        (y, stock.shape[0] - x - height),
                        width, height,
                        linewidth=2, edgecolor="black", facecolor=color
                    )
                    ax.add_patch(rect)

    # Hide empty cells if number of stocks is not divisible by 4
    for i in range(num_stocks, rows * cols):
        row, col = divmod(i, 4)
        fig.delaxes(axes[row][col])

    plt.tight_layout()
    plt.show()

def visualize_stock_heatmap(stock, title="Stock Utilization Heatmap"):
    """
    Create a heatmap visualization of a single stock
    """
    plt.figure(figsize=(10, 8))
    
    # Convert stock to binary map (used vs unused)
    binary_map = (stock != -1).astype(int)
    
    ax = sns.heatmap(binary_map, cmap="YlGnBu", cbar_kws={'label': 'Utilized'})
    ax.set_title(title)
    
    # Add product boundaries
    visited = np.full(stock.shape, False)
    for x in range(stock.shape[0]):
        for y in range(stock.shape[1]):
            product_id = stock[x, y]
            
            if product_id >= 1 and not visited[x, y]:
                width, height = 1, 1
                while y + width < stock.shape[1] and stock[x, y + width] == product_id:
                    width += 1
                while x + height < stock.shape[0] and np.all(stock[x:x + height, y:y + width] == product_id):
                    height += 1
                    
                visited[x:x + height, y:y + width] = True
                
                # Add rectangle outline
                rect = patches.Rectangle(
                    (y, x), width, height,
                    linewidth=2, edgecolor="red", facecolor="none"
                )
                ax.add_patch(rect)
                
                # Add product ID in the center
                plt.text(y + width/2, x + height/2, f"ID:{product_id}", 
                         horizontalalignment='center', verticalalignment='center')
    
    plt.show()

def visualize_comparison(ff_metrics, bf_metrics, greedy_metrics):
    """
    Create charts comparing metrics across algorithms
    """
    # Prepare data
    algorithms = ['First-Fit', 'Best-Fit', 'Greedy']
    
    # Main metrics
    fitness = [
        ff_metrics['fitness_score'],
        bf_metrics['fitness_score'],
        greedy_metrics['fitness_score']
    ]
    
    utilization = [
        ff_metrics['utilization_rate'],
        bf_metrics['utilization_rate'],
        greedy_metrics['utilization_rate']
    ]
    
    waste = [
        ff_metrics['waste_ratio'],
        bf_metrics['waste_ratio'],
        greedy_metrics['waste_ratio']
    ]
    
    runtime = [0, 0, 0]  # Default to 0 if runtime not measured
    if all(m.get('runtime') is not None for m in [ff_metrics, bf_metrics, greedy_metrics]):
        runtime = [
            ff_metrics['runtime'],
            bf_metrics['runtime'],
            greedy_metrics['runtime']
        ]
    
    # Secondary metrics
    fragmentation = [
        ff_metrics['fragmentation_rate'],
        bf_metrics['fragmentation_rate'],
        greedy_metrics['fragmentation_rate']
    ]
    
    dispersion = [
        ff_metrics['product_dispersion'],
        bf_metrics['product_dispersion'],
        greedy_metrics['product_dispersion']
    ]
    
    total_stocks = [
        ff_metrics['total_stocks'],
        bf_metrics['total_stocks'],
        greedy_metrics['total_stocks']
    ]
    
    # Create charts for the three main criteria
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Algorithm Performance: Fitness, Waste, Runtime', fontsize=16)
    
    # Fitness score chart
    axes[0].bar(algorithms, fitness, color=['blue', 'green', 'red'])
    axes[0].set_title('Fitness Score (higher is better)')
    axes[0].set_ylim(0, 100)
    for i, v in enumerate(fitness):
        axes[0].text(i, v + 1, f"{v:.1f}", ha='center')
    
    # Waste rate chart
    axes[1].bar(algorithms, waste, color=['blue', 'green', 'red'])
    axes[1].set_title('Waste Rate % (lower is better)')
    axes[1].set_ylim(0, max(waste) * 1.2)
    for i, v in enumerate(waste):
        axes[1].text(i, v + 1, f"{v:.1f}%", ha='center')
    
    # Runtime chart
    axes[2].bar(algorithms, runtime, color=['blue', 'green', 'red'])
    axes[2].set_title('Runtime in seconds (lower is better)')
    if any(runtime):  # Only if there's actual runtime data
        axes[2].set_ylim(0, max(runtime) * 1.2)
        for i, v in enumerate(runtime):
            axes[2].text(i, v + 0.01, f"{v:.3f}s", ha='center')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    
    # Save chart
    plt.savefig(f"main_metrics_comparison_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.show()
    
    # Create charts for secondary metrics
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Additional Algorithm Performance Metrics', fontsize=16)
    
    # Utilization rate chart
    axes[0, 0].bar(algorithms, utilization, color=['blue', 'green', 'red'])
    axes[0, 0].set_title('Utilization Rate (%)')
    axes[0, 0].set_ylim(0, 100)
    for i, v in enumerate(utilization):
        axes[0, 0].text(i, v + 1, f"{v:.1f}%", ha='center')
    
    # Fragmentation chart
    axes[0, 1].bar(algorithms, fragmentation, color=['blue', 'green', 'red'])
    axes[0, 1].set_title('Fragmentation Rate (lower is better)')
    for i, v in enumerate(fragmentation):
        axes[0, 1].text(i, v + 0.1, f"{v:.2f}", ha='center')
    
    # Stocks used chart
    axes[1, 0].bar(algorithms, total_stocks, color=['blue', 'green', 'red'])
    axes[1, 0].set_title('Number of Stocks Used (lower is better)')
    for i, v in enumerate(total_stocks):
        axes[1, 0].text(i, v + 0.1, str(v), ha='center')
    
    # Product dispersion chart
    axes[1, 1].bar(algorithms, dispersion, color=['blue', 'green', 'red'])
    axes[1, 1].set_title('Product Dispersion (lower is better)')
    for i, v in enumerate(dispersion):
        axes[1, 1].text(i, v + 0.1, f"{v:.2f}", ha='center')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Save chart
    plt.savefig(f"additional_metrics_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.show()

def create_radar_chart(ff_metrics, bf_metrics, greedy_metrics):
    """
    Create a radar chart comparing different algorithms
    """
    # Prepare data - normalize all metrics to 0-1 scale (1 is best)
    categories = ['Fitness', 'Utilization', 'Low Waste', 'Speed', 'Low Fragmentation', 'Low Dispersion']
    
    # Normalize metrics (higher is better)
    max_fitness = max([ff_metrics['fitness_score'], bf_metrics['fitness_score'], greedy_metrics['fitness_score']])
    norm_fitness_ff = ff_metrics['fitness_score'] / 100  # Already 0-100
    norm_fitness_bf = bf_metrics['fitness_score'] / 100
    norm_fitness_gr = greedy_metrics['fitness_score'] / 100
    
    norm_util_ff = ff_metrics['utilization_rate'] / 100  # Already 0-100
    norm_util_bf = bf_metrics['utilization_rate'] / 100
    norm_util_gr = greedy_metrics['utilization_rate'] / 100
    
    # Waste - invert so higher is better
    norm_waste_ff = 1 - (ff_metrics['waste_ratio'] / 100)
    norm_waste_bf = 1 - (bf_metrics['waste_ratio'] / 100)
    norm_waste_gr = 1 - (greedy_metrics['waste_ratio'] / 100)
    
    # Runtime - invert and normalize
    if all(m.get('runtime') is not None for m in [ff_metrics, bf_metrics, greedy_metrics]):
        max_runtime = max([ff_metrics['runtime'], bf_metrics['runtime'], greedy_metrics['runtime']])
        if max_runtime > 0:
            norm_runtime_ff = 1 - (ff_metrics['runtime'] / max_runtime)
            norm_runtime_bf = 1 - (bf_metrics['runtime'] / max_runtime)
            norm_runtime_gr = 1 - (greedy_metrics['runtime'] / max_runtime)
        else:
            norm_runtime_ff = norm_runtime_bf = norm_runtime_gr = 1.0
    else:
        norm_runtime_ff = norm_runtime_bf = norm_runtime_gr = 0.5  # Middle if not measured
    
    # Fragmentation - invert and normalize (assume 5 is worst case)
    max_frag = max([ff_metrics['fragmentation_rate'], bf_metrics['fragmentation_rate'], 
                   greedy_metrics['fragmentation_rate'], 5])
    norm_frag_ff = 1 - (ff_metrics['fragmentation_rate'] / max_frag)
    norm_frag_bf = 1 - (bf_metrics['fragmentation_rate'] / max_frag)
    norm_frag_gr = 1 - (greedy_metrics['fragmentation_rate'] / max_frag)
    
    # Dispersion - invert and normalize (assume 3 is worst case)
    max_disp = max([ff_metrics['product_dispersion'], bf_metrics['product_dispersion'], 
                   greedy_metrics['product_dispersion'], 3])
    norm_disp_ff = 1 - (ff_metrics['product_dispersion'] / max_disp)
    norm_disp_bf = 1 - (bf_metrics['product_dispersion'] / max_disp)
    norm_disp_gr = 1 - (greedy_metrics['product_dispersion'] / max_disp)
    
    # Create data arrays
    ff_data = [norm_fitness_ff, norm_util_ff, norm_waste_ff, norm_runtime_ff, norm_frag_ff, norm_disp_ff]
    bf_data = [norm_fitness_bf, norm_util_bf, norm_waste_bf, norm_runtime_bf, norm_frag_bf, norm_disp_bf]
    gr_data = [norm_fitness_gr, norm_util_gr, norm_waste_gr, norm_runtime_gr, norm_frag_gr, norm_disp_gr]
    
    # Repeat first value to close the polygon
    ff_data.append(ff_data[0])
    bf_data.append(bf_data[0])
    gr_data.append(gr_data[0])
    
    # Add categories with the first one repeated
    cat = categories.copy()
    cat.append(categories[0])
    
    # Create angles
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    angles += [angles[0]]  # Close the loop
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
    
    # Plot data
    ax.plot(angles, ff_data, 'o-', linewidth=2, label='First-Fit', color='blue')
    ax.fill(angles, ff_data, alpha=0.1, color='blue')
    
    ax.plot(angles, bf_data, 'o-', linewidth=2, label='Best-Fit', color='green')
    ax.fill(angles, bf_data, alpha=0.1, color='green')
    
    ax.plot(angles, gr_data, 'o-', linewidth=2, label='Greedy', color='red')
    ax.fill(angles, gr_data, alpha=0.1, color='red')
    
    # Set category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    
    # Add legend
    ax.legend(loc='upper right')
    
    plt.title('Algorithm Performance Radar Chart', size=15)
    
    # Set y-axis to go from 0 to 1
    ax.set_ylim(0, 1)
    
    # Add grid
    ax.grid(True)
    
    # Save and show
    plt.savefig(f"radar_chart_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.show()