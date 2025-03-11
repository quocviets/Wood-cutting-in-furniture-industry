import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def visualize_cutting(stock, placements, pieces, title):
    """
    Visualizes the cutting layout by displaying the stock and placed pieces.
    """
    fig, ax = plt.subplots(figsize=(5, 10))  # Maintain aspect ratio
    ax.set_xlim(0, stock['width'])  # Keep stock width unchanged
    ax.set_ylim(0, stock['height'])
    ax.set_title(title)
    
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow']
    legend_patches = []
    
    for i, place in enumerate(placements):
        piece = next(p for p in pieces if p['id'] == place['id'])
        rect = plt.Rectangle((place['x'], place['y']),
                             piece['width'], piece['height'],
                             edgecolor='black',
                             facecolor=colors[i % len(colors)],
                             alpha=0.5)
        ax.add_patch(rect)
        
        # Add to legend list
        legend_patches.append(mpatches.Patch(color=colors[i % len(colors)], label=f"{piece['name']} ({piece['width']}x{piece['height']})"))
    
    plt.gca().invert_yaxis()
    
    # Display legend outside the stock
    fig.legend(handles=legend_patches, loc='upper right', fontsize=8)
    plt.show()

def calculate_waste(stock, placements):
    """
    Calculates the amount of wasted material after placing pieces.
    """
    used_area = sum(p['width'] * p['height'] for p in pieces for _ in range(p['quantity']))
    total_area = stock['width'] * stock['height']
    return total_area - used_area

def first_fit(stock, pieces):
    """
    First Fit heuristic: Places each piece in the first available position.
    """
    layout = np.zeros((stock['height'], stock['width']))  # Representation of the stock
    placements = []
    
    for piece in pieces:
        for _ in range(piece['quantity']):
            placed = False
            for y in range(stock['height'] - piece['height'] + 1):
                for x in range(stock['width'] - piece['width'] + 1):
                    if np.all(layout[y:y+piece['height'], x:x+piece['width']] == 0):
                        layout[y:y+piece['height'], x:x+piece['width']] = piece['id']
                        placements.append({'id': piece['id'], 'x': x, 'y': y, 'width': piece['width'], 'height': piece['height']})
                        placed = True
                        break
                if placed:
                    break
    return placements

def best_fit(stock, pieces):
    """
    Best Fit heuristic: Places each piece in the position that minimizes leftover space.
    """
    layout = np.zeros((stock['height'], stock['width']))
    placements = []
    
    for piece in sorted(pieces, key=lambda p: p['width'] * p['height'], reverse=True):  # Sort by decreasing area
        for _ in range(piece['quantity']):
            best_pos = None
            min_waste = float('inf')
            
            for y in range(stock['height'] - piece['height'] + 1):
                for x in range(stock['width'] - piece['width'] + 1):
                    if np.all(layout[y:y+piece['height'], x:x+piece['width']] == 0):
                        waste = np.count_nonzero(layout[y:y+piece['height'], x:x+piece['width']])
                        if waste < min_waste:
                            min_waste = waste
                            best_pos = (x, y)
            
            if best_pos:
                x, y = best_pos
                layout[y:y+piece['height'], x:x+piece['width']] = piece['id']
                placements.append({'id': piece['id'], 'x': x, 'y': y, 'width': piece['width'], 'height': piece['height']})
    return placements

# Define stock and pieces
stock = {"width": 100, "height": 200}
pieces = [
    {"id": 1, "name": "Table", "width": 40, "height": 75, "quantity": 2},
    {"id": 2, "name": "Chair", "width": 20, "height": 25, "quantity": 4},
    {"id": 3, "name": "Chair Leg", "width": 4, "height": 25, "quantity": 8}
]

# Execute First Fit and Best Fit algorithms
ff_placements = first_fit(stock, pieces)
bf_placements = best_fit(stock, pieces)

# Visualize results
visualize_cutting(stock, ff_placements, pieces, "First Fit Cutting Layout")
visualize_cutting(stock, bf_placements, pieces, "Best Fit Cutting Layout")

# Compare efficiency
ff_waste = calculate_waste(stock, ff_placements)
bf_waste = calculate_waste(stock, bf_placements)

print(f"First Fit Waste: {ff_waste} square units")
print(f"Best Fit Waste: {bf_waste} square units")