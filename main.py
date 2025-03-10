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
    observation = {
        "products": [
            {"size": (3, 3), "quantity": 2, "id": 1},
            {"size": (2, 2), "quantity": 3, "id": 2},
            {"size": (4, 4), "quantity": 1, "id": 3}
        ],
        "stocks": [
            np.full((6, 6), -1),  # Kho trống kích thước 6x6
            np.full((5, 5), -1)   # Kho trống kích thước 5x5
        ]
    }
    return observation, {}

def place_product(stock, pos_x, pos_y, size, product_id):
    """ Đặt sản phẩm vào kho ở vị trí cụ thể """
    prod_w, prod_h = size
    stock[pos_x:pos_x + prod_w, pos_y:pos_y + prod_h] = product_id

def visualize_stocks(stocks, products, title="Stock Visualization"):
    """
    Hiển thị các kho với màu sắc cố định cho từng loại sản phẩm.
    """
    fig, axes = plt.subplots(1, len(stocks), figsize=(5 * len(stocks), 5))

    if len(stocks) == 1:
        axes = [axes]  # Đảm bảo axes là danh sách
    
    # Định nghĩa màu cố định cho từng sản phẩm
    product_colors = {
        1: "#0000FF",  # Blue
        2: "#FFD700",  # Gold (Yellow)
        3: "#FF0000"   # Red
    }


    for i, stock in enumerate(stocks):
        ax = axes[i]
        ax.set_xticks(range(stock.shape[1] + 1))
        ax.set_yticks(range(stock.shape[0] + 1))
        ax.grid(True, color="black", linewidth=0.5)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_title(f"Stock {i}")

        for x in range(stock.shape[0]):
            for y in range(stock.shape[1]):
                cell_value = stock[x, y]
                print(f"Stock {i} at ({x},{y}): {cell_value}")  # Debug giá trị

                if cell_value >= 1:  # Đảm bảo ID hợp lệ
                    color = product_colors.get(int(cell_value), "gray")  # Chuyển sang int tránh lỗi
                    rect = patches.Rectangle((y, stock.shape[0] - x - 1), 1, 1,
                                            linewidth=1.5, edgecolor="black",
                                            facecolor=color)
                    ax.add_patch(rect)


    plt.show()


def main():
    observation_ff, info_ff = create_sample_data()
    observation_bf, info_bf = create_sample_data()
    observation_greedy, info_greedy = create_sample_data()

    # observation, info = create_sample_data()
    # print("Dữ liệu đầu vào:", observation, info)
    # updated_warehouse = first_fit_policy(observation, info)
    # print("Kho sau khi áp dụng First-Fit:", updated_warehouse)


    print("Chạy thuật toán First-Fit...")
    first_fit_policy(observation_ff, info_ff)  
    visualize_stocks(observation_ff["stocks"], "First-Fit Algorithm")

    print("Chạy thuật toán Best-Fit...")
    best_fit_policy(observation_bf, info_bf)  
    visualize_stocks(observation_bf["stocks"], "Best-Fit Algorithm")


    print("Chạy thuật toán Greedy...")
    greedy_policy(observation_greedy, info_greedy)
    visualize_stocks(observation_greedy["stocks"], observation_greedy["products"], "Greedy Algorithm")
    
if __name__ == "__main__":
    main()
