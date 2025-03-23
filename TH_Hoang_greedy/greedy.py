import numpy as np
import json
import matplotlib.pyplot as plt
from prettytable import PrettyTable
import time

def greedy_policy(observation, info):
    """
    Thuật toán Greedy cho bài toán 2D Cutting Stock:
    - Xếp sản phẩm vào stock đầu tiên có đủ chỗ trống.
    - Nếu không có stock phù hợp, tạo một stock mới.
    - Ưu tiên đặt sản phẩm lớn trước để tối ưu không gian.
    """
    list_prods = sorted(observation["products"], key=lambda p: p["size"][0] * p["size"][1], reverse=True)
    list_stocks = observation["stocks"]
    
    actions = []  # Danh sách các hành động đặt sản phẩm

    for prod in list_prods:
        while prod["quantity"] > 0:  # Đặt hết tất cả sản phẩm cùng loại
            prod_w, prod_h = prod["size"]
            placed = False

            for idx, stock in enumerate(list_stocks):
                stock_h, stock_w = stock.shape  # Kích thước của kho hàng

                # Kiểm tra xem stock có đủ chỗ không
                if stock_h < prod_h or stock_w < prod_w:
                    continue  # Kho quá nhỏ -> bỏ qua

                # Duyệt từng vị trí có thể đặt
                for x in range(stock_h - prod_h + 1):
                    for y in range(stock_w - prod_w + 1):
                        if np.all(stock[x:x + prod_h, y:y + prod_w] == -1):
                            #  Đặt sản phẩm vào stock (đánh dấu bằng ID sản phẩm)
                            stock[x:x + prod_h, y:y + prod_w] = prod["id"]

                            # Lưu hành động
                            actions.append({
                                "stock_idx": idx,
                                "size": (prod_w, prod_h),
                                "position": (x, y),
                                "product_id": prod["id"]
                            })

                            prod["quantity"] -= 1  # Giảm số lượng sản phẩm
                            placed = True
                            break
                    if placed:
                        break

            # Nếu không tìm thấy chỗ đặt, mở stock mới
            if not placed:
                new_stock_idx = len(list_stocks)
                new_stock = np.full((6, 6), -1)  # Tạo kho mới (kích thước mặc định 6x6)
                list_stocks.append(new_stock)

                # Đặt sản phẩm vào kho mới
                new_stock[0:prod_h, 0:prod_w] = prod["id"]

                actions.append({
                    "stock_idx": new_stock_idx,
                    "size": (prod_w, prod_h),
                    "position": (0, 0),
                    "product_id": prod["id"]
                })

                prod["quantity"] -= 1  # Giảm số lượng sản phẩm
    
    return actions

def calculate_waste_summary(list_stocks, products):
    summary = []
    for idx, stock in enumerate(list_stocks):
        total_area = stock.shape[0] * stock.shape[1]
        used_area = np.sum(stock != -1)
        waste_area = total_area - used_area

        summary.append({
            "Stock Index": idx + 1,
            "Total Area": total_area,
            "Used Area": used_area,
            "Waste Area": waste_area
        })

    return summary

def plot_stocks(list_stocks):
    fig, axes = plt.subplots(1, len(list_stocks), figsize=(5 * len(list_stocks), 5))
    if len(list_stocks) == 1:
        axes = [axes]

    for idx, stock in enumerate(list_stocks):
        axes[idx].imshow(stock, cmap="tab20", origin="upper")
        axes[idx].set_title(f"Stock {idx+1}")
        axes[idx].axis("off")

    plt.tight_layout()
    plt.show()

def print_waste_summary(summary):
    table = PrettyTable()
    table.field_names = ["Stock ID", "Total Area", "Used Area", "Waste Area", "Waste Rate"]
    
    total_waste = 0
    total_area = 0
    for item in summary:
        waste_rate = item["Waste Area"] / item["Total Area"] if item["Total Area"] != 0 else 0
        table.add_row([item["Stock Index"], item["Total Area"], item["Used Area"], item["Waste Area"], f"{waste_rate:.4f}"])
        total_waste += item["Waste Area"]
        total_area += item["Total Area"]
    
    print("Waste Summary:")
    print(table)
    print(f"Total Waste Area: {total_waste} square units")
    print(f"Total Waste Rate: {total_waste / total_area:.4f}")

def calculate_fitness(summary):
    total_area = sum(item["Total Area"] for item in summary)
    used_area = sum(item["Used Area"] for item in summary)
    waste_area = total_area - used_area
    utilization_rate = used_area / total_area if total_area != 0 else 0

    fitness_score = utilization_rate / len(summary) if len(summary) != 0 else 0

    print(f"\nFitness Score: {fitness_score:.4f}")
    print(f"Utilization Rate: {utilization_rate:.4f}")
    print(f"Total Stocks Used: {len(summary)}")

def main():
    start_time = time.time()

    # Đọc dữ liệu từ file JSON
    with open("TH_Hoang_greedy/results_Heuristic_Algorithms/data_input/data_3.json", "r") as file:
        data = json.load(file)

    # Chuyển đổi dữ liệu thành dạng quan sát
    observation = {
        "stocks": [np.full((stock["size"][0], stock["size"][1]), -1) for stock in data["stocks"]],
        "products": data["products"],
    }

    info = {}
    actions = greedy_policy(observation, info)

    print("Danh sách hành động:")
    for action in actions:
        print(action)

    plot_stocks(observation["stocks"])

    summary = calculate_waste_summary(observation["stocks"], observation["products"])
    print_waste_summary(summary)
    calculate_fitness(summary)

    end_time = time.time()
    print(f"Runtime: {end_time - start_time:.4f} seconds")

if __name__ == "__main__":
    main()
