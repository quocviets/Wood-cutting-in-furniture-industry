import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from policy.BestFit import best_fit_policy
from policy.FirstFit import first_fit_policy

def create_sample_data():
    """
    Tạo dữ liệu giả lập gồm danh sách sản phẩm và danh sách kho chứa.
    """
    observation = {
        "products": [
            {"size": (3, 3), "quantity": 2},
            {"size": (2, 2), "quantity": 3},
            {"size": (4, 4), "quantity": 1}
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
    Hiển thị các stocks với màu sắc khác nhau cho từng loại sản phẩm.
    """
    fig, axes = plt.subplots(1, len(stocks), figsize=(5 * len(stocks), 5))
    
    if len(stocks) == 1:
        axes = [axes]  # Đảm bảo axes là danh sách
    
    # Tạo màu cho từng loại sản phẩm
    product_colors = {}
    cmap = plt.get_cmap("tab10")  # Chọn 10 màu khác nhau
    color_index = 0

    for i, stock in enumerate(stocks):
        ax = axes[i]
        ax.set_xticks(range(stock.shape[1] + 1))
        ax.set_yticks(range(stock.shape[0] + 1))
        ax.grid(True, color="black", linewidth=0.5)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_title(f"{title} - Stock {i}")
        ax.set_title(f"Stock {i}")

        for x in range(stock.shape[0]):
            for y in range(stock.shape[1]):
                cell_value = stock[x, y]
                
                if cell_value > 0:
                    # Ánh xạ sản phẩm vào màu sắc
                    if cell_value not in product_colors:
                        product_colors[cell_value] = cmap(color_index % 10)
                        color_index += 1
                    
                    color = product_colors[cell_value]

                    # Vẽ hình chữ nhật với màu sản phẩm
                    rect = patches.Rectangle((y, stock.shape[0] - x - 1), 1, 1,
                                             linewidth=1.5, edgecolor="black",
                                             facecolor=color)
                    ax.add_patch(rect)
    
    plt.show()

def main():
    observation_ff, info_ff = create_sample_data()
    observation_bf, info_bf = create_sample_data()

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

if __name__ == "__main__":
    main()