import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Define available wooden stock sheets with their dimensions
stock_sheets = [
    {"id": 1, "length": 120, "width": 60},
    {"id": 2, "length": 100, "width": 50},
    {"id": 3, "length": 90, "width": 40},
    {"id": 4, "length": 80, "width": 30}
]

# Define the demand pieces with their dimensions and required quantity
demand_pieces = [
    {"id": 1, "length": 50, "width": 30, "quantity": 4},
    {"id": 2, "length": 40, "width": 20, "quantity": 6},
    {"id": 3, "length": 60, "width": 50, "quantity": 2},
    {"id": 4, "length": 30, "width": 20, "quantity": 8},
    {"id": 5, "length": 70, "width": 40, "quantity": 3}
]

def first_fit(stock_sheets, demand_pieces):
    """
    First Fit heuristic: Place each piece in the first available position on a stock sheet.
    """
    placements = []  # List to store placement details
    used_stocks = set()  # Set to track used stock sheets
    reward = 0  # Initialize reward
    
    # Create a layout representation for each stock sheet
    stock_layouts = {stock['id']: np.zeros((stock['width'], stock['length'])) for stock in stock_sheets}
    
    # Iterate through demand pieces
    for piece in demand_pieces:
        for _ in range(piece['quantity']):
            placed = False  # Flag to check if piece is placed
            
            # Try to place the piece in the first available stock sheet
            for stock in stock_sheets:
                for y in range(stock['width'] - piece['width'] + 1):
                    for x in range(stock['length'] - piece['length'] + 1):
                        # Check if the space is available (empty)
                        if np.all(stock_layouts[stock['id']][y:y+piece['width'], x:x+piece['length']] == 0):
                            # Place the piece on the stock sheet
                            stock_layouts[stock['id']][y:y+piece['width'], x:x+piece['length']] = piece['id']
                            placements.append({"stock_id": stock['id'], "piece_id": piece['id'],
                                               "x": x, "y": y, "length": piece['length'], "width": piece['width']})
                            used_stocks.add(stock['id'])  # Mark stock as used
                            placed = True  # Set flag to true
                            break
                    if placed:
                        break
                if placed:
                    break
            
            # If no placement was found, penalize the reward
            if not placed:
                print(f"Cannot place piece {piece['id']}")
                reward -= 10  # Deduct 10 points for each unplaced piece
    
    return placements, used_stocks, reward

def calculate_waste(stock_sheets, placements):
    """
    Calculates and prints the waste for each used stock sheet.
    """
    total_waste = 0
    stock_usage = {stock['id']: 0 for stock in stock_sheets}  # Track used area
    
    # Calculate the used area for each stock
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
    Visualizes the cutting layout on used stock sheets with size annotations.
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

def print_summary(placements):
    """
    Prints a summary table of the cutting stock results.
    """
    print("\nSummary of First Fit Cutting:")
    print("+------------+----------+------------+-----------+")
    print("| Stock ID   | Piece ID | Dimensions | Position  |")
    print("+------------+----------+------------+-----------+")
    for place in placements:
        print(f"| {place['stock_id']:^10} | {place['piece_id']:^8} | {place['length']}x{place['width']:^5} | ({place['x']},{place['y']}) |")
    print("+------------+----------+------------+-----------+")

# Execute First Fit algorithm
ff_placements, ff_used_stocks, ff_reward = first_fit(stock_sheets, demand_pieces)
print(f"\nTotal Reward: {ff_reward}")

# Visualize the cutting results
visualize_cutting(stock_sheets, ff_placements, ff_used_stocks, "First Fit Cutting Layout")

# Print summary table
print_summary(ff_placements)

# Calculate and print waste
calculate_waste(stock_sheets, ff_placements)
