# utils/animation.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import os
from matplotlib.colors import to_rgba

def create_packing_frames(algorithm_func, observation, info, output_dir="frames"):
    """
    Run algorithm and capture frames at each step
    
    Args:
        algorithm_func: Function implementing the algorithm
        observation: Initial observation state
        info: Additional info
        output_dir: Directory to save frames
    
    Returns:
        List of stock states at each step
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Make a deep copy of the initial observation
    import copy
    obs = copy.deepcopy(observation)
    
    # Modify the policy function to capture intermediate states
    original_place_product = None
    frames = []
    
    # Add first frame (empty stocks)
    frames.append(copy.deepcopy(obs["stocks"]))
    
    # Define function to intercept product placement
    def place_product_with_capture(product_id, product, stock_idx, pos):
        # Call original function
        result = original_place_product(product_id, product, stock_idx, pos)
        
        # If placement was successful, capture the frame
        if result:
            frames.append(copy.deepcopy(obs["stocks"]))
        
        return result
    
    # Replace the place_product function with our capturing version
    # This works based on the assumption that algorithms use a place_product function
    # that modifies the observation
    if hasattr(algorithm_func, "__globals__") and "place_product" in algorithm_func.__globals__:
        original_place_product = algorithm_func.__globals__["place_product"]
        algorithm_func.__globals__["place_product"] = place_product_with_capture
    
    # Run the algorithm
    actions = algorithm_func(obs, info)
    
    # Restore original function if we replaced it
    if original_place_product:
        algorithm_func.__globals__["place_product"] = original_place_product
    
    return frames

def create_packing_gif(frames, algorithm_name, output_file=None, fps=2):
    """
    Create a GIF from packing frames
    
    Args:
        frames: List of stock states at each step
        algorithm_name: Name of the algorithm for the title
        output_file: File path to save the GIF
        fps: Frames per second
    """
    if not output_file:
        output_file = f"results/{algorithm_name.lower().replace(' ', '_')}_animation.gif"
    
    # Create figure
    fig = plt.figure(figsize=(12, 8))
    plt.title(f"{algorithm_name} Algorithm")
    
    # Product colors dictionary
    product_colors = {
        1: "#0000FF",  # Blue
        2: "#FFD700",  # Gold
        3: "#FF0000",  # Red
        4: "#b57edc",  # Purple
        5: "#7fffd4",  # Aquamarine
    }
    
    # Function to draw a single frame
    def draw_frame(frame_idx):
        plt.clf()
        plt.suptitle(f"{algorithm_name} Algorithm - Step {frame_idx}/{len(frames)-1}", fontsize=16)
        
        stocks = frames[frame_idx]
        num_stocks = len(stocks)
        cols = min(4, num_stocks)
        rows = (num_stocks + cols - 1) // cols
        
        # Create subplots
        for i, stock in enumerate(stocks):
            ax = plt.subplot(rows, cols, i + 1)
            ax.set_xticks(range(stock.shape[1] + 1))
            ax.set_yticks(range(stock.shape[0] + 1))
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_title(f"Stock {i}")
            ax.set_aspect('equal')
            
            # Fill background with light gray for empty cells
            background = patches.Rectangle(
                (0, 0), stock.shape[1], stock.shape[0],
                linewidth=0, facecolor="#EEEEEE"
            )
            ax.add_patch(background)
            
            # Draw grid lines
            for x in range(stock.shape[1] + 1):
                ax.plot([x, x], [0, stock.shape[0]], 'k-', lw=0.5, alpha=0.3)
            for y in range(stock.shape[0] + 1):
                ax.plot([0, stock.shape[1]], [y, y], 'k-', lw=0.5, alpha=0.3)
            
            visited = np.full(stock.shape, False)
            
            # Draw products
            for x in range(stock.shape[0]):
                for y in range(stock.shape[1]):
                    product_id = stock[x, y]
                    
                    if product_id >= 1 and not visited[x, y]:
                        # Find dimensions of this product piece
                        width, height = 1, 1
                        while y + width < stock.shape[1] and stock[x, y + width] == product_id:
                            width += 1
                        while x + height < stock.shape[0] and np.all(stock[x:x + height, y:y + width] == product_id):
                            height += 1
                        
                        visited[x:x + height, y:y + width] = True
                        
                        # Draw rectangle
                        color = product_colors.get(product_id, "gray")
                        rect = patches.Rectangle(
                            (y, stock.shape[0] - x - height),
                            width, height,
                            linewidth=2, edgecolor="black", facecolor=color
                        )
                        ax.add_patch(rect)
                        
                        # Add product ID text
                        ax.text(
                            y + width/2, 
                            stock.shape[0] - x - height/2, 
                            f"{product_id}", 
                            horizontalalignment='center',
                            verticalalignment='center',
                            color='black',
                            fontweight='bold',
                            fontsize=10
                        )
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout with room for title
    
    # Create animation
    ani = animation.FuncAnimation(
        fig, draw_frame, frames=range(len(frames)),
        interval=1000/fps, blit=False
    )
    
    # Save as GIF
    ani.save(output_file, writer='pillow', fps=fps)
    plt.close(fig)
    
    print(f"Animation saved to {output_file}")
    return output_file

def visualize_algorithm_steps(algorithm_func, algorithm_name, observation, info):
    """
    Create and save a GIF showing algorithm execution steps
    
    Args:
        algorithm_func: Function implementing the algorithm
        algorithm_name: Name of the algorithm
        observation: Initial observation state
        info: Additional info
    
    Returns:
        Path to created GIF file
    """
    print(f"\nCreating animation for {algorithm_name} algorithm...")
    
    # Capture frames
    frames = create_packing_frames(algorithm_func, observation, info)
    
    # If no frames were captured (usually because the algorithm doesn't use the expected placement function)
    if len(frames) <= 1:
        print(f"Warning: Could not capture steps for {algorithm_name} algorithm.")
        print("The algorithm may not use the expected placement function.")
        return None
    
    # Create GIF
    output_file = f"results/{algorithm_name.lower().replace(' ', '_')}_animation.gif"
    return create_packing_gif(frames, algorithm_name, output_file)