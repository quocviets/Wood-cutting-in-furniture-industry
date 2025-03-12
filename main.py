import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from policy.BestFit import best_fit_policy
from policy.FirstFit import first_fit_policy
from policy.Greedy import greedy_policy

def create_sample_data():
    """
    Tạo dữ liệu giả lập gồm danh sách sản phẩm và danh sách kho chứa.
    """
    return {
        "products": [
            {"width": 3, "height": 3, "quantity": 2, "id": 1},
            {"width": 2, "height": 2, "quantity": 3, "id": 2},
            {"width": 4, "height": 4, "quantity": 1, "id": 3}
        ],
        "stocks": [
            {"width": 10, "height": 10, "matrix": np.full((10, 10), -1)},
            {"width": 8, "height": 8, "matrix": np.full((8, 8), -1)}
        ]
    }

def visualize_steps(stocks, steps, title="Stock Visualization"):
    """ Hiển thị từng bước xếp sản phẩm vào stock. """
    fig, axes = plt.subplots(1, len(stocks), figsize=(5 * len(stocks), 5))
    if len(stocks) == 1:
        axes = [axes]
    
    colors = ["#0000FF", "#FFD700", "#FF0000", "#008000", "#800080"]
    
    for i, stock in enumerate(stocks):
        stock_matrix = stock["matrix"]
        ax = axes[i]
        ax.set_xticks(range(stock["width"] + 1))
        ax.set_yticks(range(stock["height"] + 1))
        ax.grid(True, color="black", linewidth=0.5)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_title(f"{title} - Stock {i}")
        
        for step in steps:
            if isinstance(step, dict) and "stock_idx" in step:
                if step["stock_idx"] == i:
                    x, y = step["position"]
                    w, h = step["size"]
                    color = colors[(step["product_id"] - 1) % len(colors)]
                    rect = patches.Rectangle((y, stock["height"] - x - 1), h, w,
                                            linewidth=1.5, edgecolor="black",
                                            facecolor=color, alpha=0.7)
                    ax.add_patch(rect)
    
    plt.show()

def apply_and_visualize(policy_func, policy_name, observation):
    """ Áp dụng thuật toán, hiển thị từng bước thực hiện. """
    print(f"Chạy thuật toán {policy_name}...")
    result = policy_func(observation, {})  
    if isinstance(result, tuple):
        stocks, steps = result
    else:
        stocks, steps = result, []
    
    visualize_steps(stocks, steps, f"{policy_name} Algorithm")

def main():
    policies = [
        (first_fit_policy, "First-Fit"),
        (best_fit_policy, "Best-Fit"),
        (greedy_policy, "Greedy")
    ]
    
    for policy_func, policy_name in policies:
        observation = create_sample_data()
        observation["products"] = [
            {"size": (p["width"], p["height"]), "quantity": p["quantity"], "id": p["id"]} 
            for p in observation["products"]
        ]
        apply_and_visualize(policy_func, policy_name, observation)
    
if __name__ == "__main__":
    main()
