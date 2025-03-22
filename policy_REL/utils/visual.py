import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import datetime

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

def visualize_comparison(ff_metrics, bf_metrics, greedy_metrics):
    """
    Create charts comparing metrics across algorithms
    """
    # Prepare data
    algorithms = ['First-Fit', 'Best-Fit', 'Greedy']
    
    utilization = [
        ff_metrics['utilization_rate'],
        bf_metrics['utilization_rate'],
        greedy_metrics['utilization_rate']
    ]
    
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
    
    # Create charts
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Algorithm Performance Comparison', fontsize=16)
    
    # Utilization rate chart
    axes[0, 0].bar(algorithms, utilization, color=['blue', 'green', 'red'])
    axes[0, 0].set_title('Utilization Rate (%)')
    axes[0, 0].set_ylim(0, 100)
    
    # Fragmentation chart
    axes[0, 1].bar(algorithms, fragmentation, color=['blue', 'green', 'red'])
    axes[0, 1].set_title('Fragmentation Rate (lower is better)')
    
    # Stocks used chart
    axes[1, 0].bar(algorithms, total_stocks, color=['blue', 'green', 'red'])
    axes[1, 0].set_title('Number of Stocks Used (lower is better)')
    
    # Product dispersion chart
    axes[1, 1].bar(algorithms, dispersion, color=['blue', 'green', 'red'])
    axes[1, 1].set_title('Product Dispersion (lower is better)')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Save chart
    plt.savefig(f"algorithm_comparison_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.show()

def visualize_stock_heatmap(stock, title="Stock Heatmap"):
    """
    Display heatmap for a stock to clearly see distribution
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(stock, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Product ID (-1: empty)')
    plt.title(title)
    plt.grid(True, color='white', linestyle='-', linewidth=0.5)
    plt.tight_layout()
    plt.show()