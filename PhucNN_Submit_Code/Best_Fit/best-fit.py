import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Data for wooden stock sheets and required pieces
stock_sheets = [
    {"id": 1, "length": 120, "width": 60},
    {"id": 2, "length": 100, "width": 50},
    {"id": 3, "length": 90, "width": 40},
    {"id": 4, "length": 80, "width": 30}
]

demand_pieces = [
    {"id": 1, "length": 50, "width": 30, "quantity": 4},
    {"id": 2, "length": 40, "width": 20, "quantity": 6},
    {"id": 3, "length": 60, "width": 50, "quantity": 2},
    {"id": 4, "length": 30, "width": 20, "quantity": 8},
    {"id": 5, "length": 70, "width": 40, "quantity": 3}
]

def best_fit(stock_sheets, demand_pieces):
    placements = []
    used_stocks = set()
    stock_layouts = {stock['id']: np.zeros((stock['width'], stock['length'])) for stock in stock_sheets}
    leftover_pieces = 0
    
    for piece in sorted(demand_pieces, key=lambda p: p['length'] * p['width'], reverse=True):
        for _ in range(piece['quantity']):
            best_stock = None
            min_waste = float('inf')
            best_x, best_y = None, None
            
            for stock in stock_sheets:
                for y in range(stock['width'] - piece['width'] + 1):
                    for x in range(stock['length'] - piece['length'] + 1):
                        if np.all(stock_layouts[stock['id']][y:y+piece['width'], x:x+piece['length']] == 0):
                            waste = (stock['length'] * stock['width']) - (piece['length'] * piece['width'])
                            if waste < min_waste:
                                min_waste = waste
                                best_stock = stock
                                best_x, best_y = x, y
            
            if best_stock:
                stock_layouts[best_stock['id']][best_y:best_y+piece['width'], best_x:best_x+piece['length']] = piece['id']
                placements.append({"stock_id": best_stock['id'], "piece_id": piece['id'],
                                   "x": best_x, "y": best_y, "length": piece['length'], "width": piece['width']})
                used_stocks.add(best_stock['id'])
            else:
                leftover_pieces += 1
                print(f"Cannot place piece {piece['id']}")
    
    return placements, used_stocks, leftover_pieces

def print_summary(placements):
    """
    Prints a summary table of the cutting stock results.
    """
    print("\nSummary of Best Fit Cutting:")
    print("+------------+----------+------------+-----------+")
    print("| Stock ID   | Piece ID | Dimensions | Position  |")
    print("+------------+----------+------------+-----------+")
    for place in placements:
        print(f"| {place['stock_id']:^10} | {place['piece_id']:^8} | {place['length']}x{place['width']:^5} | ({place['x']},{place['y']}) |")
    print("+------------+----------+------------+-----------+")

def calculate_waste(stock_sheets, placements):
    """
    Calculates and prints the waste for each stock sheet used.
    """
    total_waste = 0
    stock_usage = {stock['id']: 0 for stock in stock_sheets}
    
    for place in placements:
        stock_usage[place['stock_id']] += place['length'] * place['width']
    
    print("\nWaste Summary:")
    print("+------------+------------+------------+")
    print("| Stock ID   | Used Area  | Waste Area |")
    print("+------------+------------+------------+")
    
    for stock in stock_sheets:
        if stock['id'] in stock_usage:
            total_area = stock['length'] * stock['width']
            used_area = stock_usage[stock['id']]
            waste_area = total_area - used_area
            total_waste += waste_area
            print(f"| {stock['id']:^10} | {used_area:^10} | {waste_area:^10} |")
    
    print("+------------+------------+------------+")
    print(f"Total Waste Area: {total_waste} square units\n")
    return total_waste

def visualize_cutting(stock_sheets, placements, used_stocks, title):
    """
    Visualizes the cutting layout on used stock sheets only with size annotations.
    """
    for stock in stock_sheets:
        if stock['id'] in used_stocks:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.set_xlim(0, stock['length'])
            ax.set_ylim(0, stock['width'])
            ax.set_title(f"Stock {stock['id']} Layout ({stock['length']}x{stock['width']})")
            ax.set_xticks([])
            ax.set_yticks([])
            
            ax.add_patch(plt.Rectangle((0, 0), stock['length'], stock['width'],
                                       edgecolor='black', facecolor='gray', alpha=0.3))
            
            colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow']
            
            for i, place in enumerate(placements):
                if place['stock_id'] == stock['id']:
                    rect = plt.Rectangle((place['x'], place['y']), place['length'], place['width'],
                                         edgecolor='black', facecolor=colors[i % len(colors)], alpha=0.5)
                    ax.add_patch(rect)
                    ax.text(place['x'] + place['length']/2, place['y'] + place['width']/2,
                            f"{place['piece_id']}\n({place['length']}x{place['width']})", ha='center', va='center', fontsize=8, color='black')
            
            plt.gca().invert_yaxis()
            plt.show()


def calculate_reward(leftover_pieces):
    return -10 * leftover_pieces

# Run Best Fit
bf_placements, bf_used_stocks, leftover_pieces = best_fit(stock_sheets, demand_pieces)
reward = calculate_reward(leftover_pieces)

print(f"Total Reward: {reward}")

# Visualize results with explanations
visualize_cutting(stock_sheets, bf_placements, bf_used_stocks, "Best Fit Cutting Layout")

# Print Summary Table
print_summary(bf_placements)

# Calculate and Print Waste
calculate_waste(stock_sheets, bf_placements)