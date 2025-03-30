import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.animation as animation
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import json
import copy
import os

# data nhận vào ***cai này chỉ nên dùng 1 data thôi *** data_1.json, data_2.json, data_3.json, data_4.json
data_file = "HoangTH_qe170011_submit_code\data\data_1.json"

# Định nghĩa lại các thuật toán với khả năng ghi lại từng bước
def first_fit_policy_visualize(observation, info):
    """First-Fit với logging từng bước"""
    list_prods = observation["products"]
    list_stocks = observation["stocks"]
    
    actions = []
    history = []  # Lưu lại trạng thái sau mỗi bước
    
    # Lưu trạng thái ban đầu
    history.append({
        "stocks": copy.deepcopy(list_stocks),
        "action": None,
        "step": 0
    })
    
    step = 1
    for prod in list_prods:
        prod_w, prod_h = prod["size"]
        
        while prod["quantity"] > 0:
            placed = False

            for idx, stock in enumerate(list_stocks):
                stock_h, stock_w = stock.shape

                if stock_h < prod_w or stock_w < prod_h:
                    continue

                for x in range(stock_h - prod_w + 1):
                    for y in range(stock_w - prod_h + 1):
                        if np.all(stock[x:x + prod_w, y:y + prod_h] == -1):
                            # Đặt sản phẩm
                            stock[x:x + prod_w, y:y + prod_h] = prod["id"]
                            
                            action = {
                                "stock_idx": idx,
                                "size": (prod_w, prod_h),
                                "position": (x, y),
                                "product_id": prod["id"]
                            }
                            
                            actions.append(action)
                            
                            # Lưu trạng thái sau khi đặt
                            history.append({
                                "stocks": copy.deepcopy(list_stocks),
                                "action": action,
                                "step": step
                            })
                            step += 1
                            
                            prod["quantity"] -= 1
                            placed = True
                            break
                    if placed:
                        break
                if placed:
                    break

            if not placed:
                # Tạo stock mới với kích thước 100x100
                new_stock_idx = len(list_stocks)
                new_stock = np.full((100, 100), -1)
                list_stocks.append(new_stock)

                new_stock[0:prod_w, 0:prod_h] = prod["id"]
                
                action = {
                    "stock_idx": new_stock_idx,
                    "size": (prod_w, prod_h),
                    "position": (0, 0),
                    "product_id": prod["id"]
                }
                
                actions.append(action)
                
                # Lưu trạng thái sau khi đặt vào stock mới
                history.append({
                    "stocks": copy.deepcopy(list_stocks),
                    "action": action,
                    "step": step
                })
                step += 1
                
                prod["quantity"] -= 1
    
    return actions, history

def best_fit_policy_visualize(observation, info):
    """Best-Fit với logging từng bước"""
    list_prods = observation["products"]
    list_stocks = observation["stocks"]
    
    actions = []
    history = []
    
    # Lưu trạng thái ban đầu
    history.append({
        "stocks": copy.deepcopy(list_stocks),
        "action": None,
        "step": 0
    })
    
    step = 1
    for prod in list_prods:
        prod_w, prod_h = prod["size"]
        
        while prod["quantity"] > 0:
            best_position = None
            best_stock_idx = None
            min_waste = float('inf')
            
            for idx, stock in enumerate(list_stocks):
                stock_h, stock_w = stock.shape
                
                if stock_h < prod_w or stock_w < prod_h:
                    continue
                
                for x in range(stock_h - prod_w + 1):
                    for y in range(stock_w - prod_h + 1):
                        if np.all(stock[x:x + prod_w, y:y + prod_h] == -1):
                            waste = 0
                            
                            if x > 0:
                                waste += sum(1 for i in range(y, y + prod_h) if stock[x-1, i] == -1)
                            
                            if x + prod_w < stock_h:
                                waste += sum(1 for i in range(y, y + prod_h) if stock[x+prod_w, i] == -1)
                            
                            if y > 0:
                                waste += sum(1 for i in range(x, x + prod_w) if stock[i, y-1] == -1)
                            
                            if y + prod_h < stock_w:
                                waste += sum(1 for i in range(x, x + prod_w) if stock[i, y+prod_h] == -1)
                            
                            if waste < min_waste:
                                min_waste = waste
                                best_position = (x, y)
                                best_stock_idx = idx
            
            if best_position is not None:
                x, y = best_position
                stock = list_stocks[best_stock_idx]
                
                stock[x:x + prod_w, y:y + prod_h] = prod["id"]
                
                action = {
                    "stock_idx": best_stock_idx,
                    "size": (prod_w, prod_h),
                    "position": (x, y),
                    "product_id": prod["id"]
                }
                
                actions.append(action)
                
                # Lưu trạng thái sau khi đặt
                history.append({
                    "stocks": copy.deepcopy(list_stocks),
                    "action": action,
                    "step": step
                })
                step += 1
                
                prod["quantity"] -= 1
            else:
                new_stock_idx = len(list_stocks)
                # Sử dụng kích thước 100x100 cho stock mới
                new_stock = np.full((100, 100), -1)
                list_stocks.append(new_stock)
                
                new_stock[0:prod_w, 0:prod_h] = prod["id"]
                
                action = {
                    "stock_idx": new_stock_idx,
                    "size": (prod_w, prod_h),
                    "position": (0, 0),
                    "product_id": prod["id"]
                }
                
                actions.append(action)
                
                # Lưu trạng thái sau khi đặt vào stock mới
                history.append({
                    "stocks": copy.deepcopy(list_stocks),
                    "action": action,
                    "step": step
                })
                step += 1
                
                prod["quantity"] -= 1
    
    return actions, history

def greedy_policy_visualize(observation, info):
    """Greedy với logging từng bước"""
    list_prods = sorted(observation["products"], key=lambda p: p["size"][0] * p["size"][1], reverse=True)
    list_stocks = observation["stocks"]
    
    actions = []
    history = []
    
    # Lưu trạng thái ban đầu
    history.append({
        "stocks": copy.deepcopy(list_stocks),
        "action": None,
        "step": 0
    })
    
    step = 1
    for prod in list_prods:
        while prod["quantity"] > 0:
            prod_w, prod_h = prod["size"]  # w, h
            placed = False

            for idx, stock in enumerate(list_stocks):
                stock_h, stock_w = stock.shape

                # Kiểm tra đủ chỗ
                if stock_h < prod_w or stock_w < prod_h:
                    continue

                # Duyệt vị trí
                for x in range(stock_h - prod_w + 1):
                    for y in range(stock_w - prod_h + 1):
                        # Kiểm tra trống
                        if np.all(stock[x:x + prod_w, y:y + prod_h] == -1):
                            # Đặt sản phẩm
                            stock[x:x + prod_w, y:y + prod_h] = prod["id"]

                            action = {
                                "stock_idx": idx,
                                "size": (prod_w, prod_h),   # (width, height)
                                "position": (x, y),         # row=x, col=y
                                "product_id": prod["id"]
                            }
                            actions.append(action)

                            # Lưu history
                            history.append({
                                "stocks": copy.deepcopy(list_stocks),
                                "action": action,
                                "step": step
                            })
                            step += 1
                            prod["quantity"] -= 1
                            placed = True
                            break
                    if placed:
                        break
                if placed:
                    break

            # Nếu không placed, tạo stock mới
            if not placed:
                new_stock_idx = len(list_stocks)
                new_stock = np.full((100, 100), -1)
                list_stocks.append(new_stock)

                # Đặt sản phẩm ở (0,0)
                new_stock[0:prod_w, 0:prod_h] = prod["id"]

                action = {
                    "stock_idx": new_stock_idx,
                    "size": (prod_w, prod_h),  # (width, height)
                    "position": (0, 0),
                    "product_id": prod["id"]
                }
                actions.append(action)

                history.append({
                    "stocks": copy.deepcopy(list_stocks),
                    "action": action,
                    "step": step
                })
                step += 1
                prod["quantity"] -= 1
    return actions, history

def load_data(filepath):
    """
    Load data from JSON file
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    observation = {
        "products": data["products"],
        "stocks": []
    }
    
    for stock in data["stocks"]:
        h, w = stock["size"]
        observation["stocks"].append(np.full((h, w), -1))
    
    return observation, {}

def create_animation(history, algorithm_name):
    """
    Tạo animation từ lịch sử các bước
    """
    # Tạo bảng màu cho các sản phẩm
    num_products = max([np.max(stock[stock >= 0]) if stock.size > 0 and np.max(stock) >= 0 else 0 
                        for step in history for stock in step["stocks"]]) + 1
    
    cmap = plt.cm.get_cmap('tab10', num_products)
    colors = [cmap(i) for i in range(num_products)]
    colors.insert(0, (0.9, 0.9, 0.9, 1.0))  # Màu cho ô trống (-1)
    
    # Tạo colormap tùy chỉnh
    custom_cmap = mcolors.ListedColormap(colors)
    bounds = list(range(-1, num_products + 1))
    norm = mcolors.BoundaryNorm(bounds, custom_cmap.N)
    
    # Tìm tổng số stock tối đa trong lịch sử
    max_stocks = max(len(step["stocks"]) for step in history)
    
    # Xác định số cột/hàng để hiển thị tất cả stocks
    cols = min(3, max_stocks)
    rows = (max_stocks + cols - 1) // cols
    
    fig, ax = plt.subplots(rows, cols, figsize=(12, 8))
    fig.suptitle(f"{algorithm_name} Algorithm Visualization", fontsize=16)
    
    # Làm phẳng mảng axes nếu cần
    if rows == 1 and cols == 1:
        ax = np.array([ax])
    elif rows == 1 or cols == 1:
        ax = ax.flatten()
    
    # Ẩn các axes không sử dụng
    for i in range(max_stocks, rows * cols):
        row, col = i // cols, i % cols
        if rows > 1:
            ax[row, col].axis('off')
        else:
            ax[i].axis('off')
    
    # Tạo text object cho thông tin bước
    step_text_obj = fig.text(0.5, 0.02, "", ha='center', fontsize=12)
    
    # Hàm cập nhật frame
    def update(frame):
        step_data = history[frame]
        stocks = step_data["stocks"]
        action = step_data["action"]
        step = step_data["step"]
        
        # Xóa nội dung cũ
        for a in ax.flatten():
            a.clear()
            a.set_xticks([])
            a.set_yticks([])
        
        # Tạo danh sách tất cả actions đến thời điểm frame
        actions_up_to_frame = []
        for idx in range(frame + 1):
            step_d = history[idx]
            if step_d["action"] is not None:
                actions_up_to_frame.append(step_d["action"])
        
        # Vẽ các stock (heatmap)
        for i, stock in enumerate(stocks):
            row, col = i // cols, i % cols
            
            if rows > 1:
                axes = ax[row, col]
            else:
                axes = ax[i]
            
            # Vẽ heatmap cho stock
            im = axes.imshow(stock, cmap=custom_cmap, norm=norm, origin='upper')
            
            # Vẽ đường lưới + tiêu đề
            axes.set_title(f"Stock {i}")
            axes.grid(True, which='both', color='black', linewidth=0.5)
            
            # Vẽ hình chữ nhật và ID cho từng action thuộc stock i
            for act in actions_up_to_frame:
                if act["stock_idx"] == i:
                    x, y = act["position"]
                    w, h = act["size"]
                    pid = act["product_id"]
                    
                    # Vẽ hình chữ nhật (chú ý Rectangle((left, top), width, height))
                    # Trong ma trận, x là row, y là column => rect((col, row), width, height)
                    rect = patches.Rectangle((y, x), h, w, 
                                            linewidth=2, edgecolor='black', facecolor='none')
                    axes.add_patch(rect)
                    
                    # Tọa độ trung tâm
                    center_x = x + w / 2
                    center_y = y + h / 2
                    
                    # In ID ở giữa
                    axes.text(center_y, center_x, f'{int(pid)}', 
                            ha='center', va='center',
                            color='white' if pid > 2 else 'black',
                            fontsize=14, fontweight='bold',
                            bbox=dict(boxstyle="round,pad=0.2", 
                                        edgecolor="black", facecolor="none"))
        
        # Cập nhật thông tin bước
        step_info = f"Step: {step}/{len(history)-1}"
        if action:
            pid = action["product_id"]
            stock_idx = action["stock_idx"]
            step_info += f" | Placed product {pid} in stock {stock_idx}"
        
        step_text_obj.set_text(step_info)
        
        return ax.flatten()

        
    # Tạo animation
    ani = animation.FuncAnimation(fig, update, frames=len(history), 
                                 interval=800, blit=False)
    
    return fig, ani

def save_animation_as_gif(ani, filename, fps=1):
    """
    Lưu animation thành file GIF
    """
    # Tạo thư mục results nếu chưa tồn tại
    os.makedirs("results", exist_ok=True)
    
    # Đường dẫn đầy đủ
    filepath = os.path.join("results", filename)
    
    # Lưu animation
    print(f"Saving animation to {filepath}...")
    ani.save(filepath, writer='pillow', fps=fps)
    print(f"Animation saved to {filepath}")

def run_all_algorithms(data_file):
    """
    Chạy tất cả các thuật toán và tạo GIF
    """
    print(f"Loading data from {data_file}...")
    
    # Tạo một bản sao riêng cho mỗi thuật toán
    observation_ff, info_ff = load_data(data_file)
    observation_bf, info_bf = load_data(data_file)
    observation_greedy, info_greedy = load_data(data_file)
    
    # Chạy First-Fit
    print("Running First-Fit algorithm...")
    _, ff_history = first_fit_policy_visualize(observation_ff, info_ff)
    
    # Chạy Best-Fit
    print("Running Best-Fit algorithm...")
    _, bf_history = best_fit_policy_visualize(observation_bf, info_bf)
    
    # Chạy Greedy
    print("Running Greedy algorithm...")
    _, greedy_history = greedy_policy_visualize(observation_greedy, info_greedy)
    
    # Tạo và lưu GIF
    print("Creating animations...")
    
    # First-Fit
    _, ff_ani = create_animation(ff_history, "First-Fit")
    save_animation_as_gif(ff_ani, "first_fit_animation.gif", fps=1)
    
    # Best-Fit
    _, bf_ani = create_animation(bf_history, "Best-Fit")
    save_animation_as_gif(bf_ani, "best_fit_animation.gif", fps=1)
    
    # Greedy
    _, greedy_ani = create_animation(greedy_history, "Greedy")
    save_animation_as_gif(greedy_ani, "greedy_animation.gif", fps=1)
    
    print("All animations created successfully!")

if __name__ == "__main__":
<<<<<<< HEAD:HoangTH_qe170011_submit_code/create_animation.py
=======
    # Để chạy từ file dữ liệu của bạn, đổi đường dẫn này:
    data_file = "policy_REL/data/data_2.json"
    
>>>>>>> 7e2b3847f1c17b86aafdc9da88ad0f74c2c9d2ef:HoangTH_QE170011_submit_code/test.py
    try:
        run_all_algorithms(data_file)
    except FileNotFoundError:
        print(f"Error: Input file '{data_file}' not found.")
        print("Looking for alternative data file...")
        
        # Thử tạo dữ liệu mẫu nếu không tìm thấy file
        try:
            # Tạo thư mục nếu chưa tồn tại
            os.makedirs("sample_data", exist_ok=True)
            
            # Tạo dữ liệu mẫu
            sample_data = {
                "products": [
                    {"id": 0, "size": [2, 3], "quantity": 3},
                    {"id": 1, "size": [1, 2], "quantity": 4},
                    {"id": 2, "size": [3, 2], "quantity": 2}
                ],
                "stocks": [
                    {"size": [6, 6]}
                ]
            }
            
            # Lưu dữ liệu mẫu
            sample_file = "sample_data/sample_data.json"
            with open(sample_file, 'w') as f:
                json.dump(sample_data, f, indent=4)
            
            print(f"Created sample data file: {sample_file}")
            print("Running algorithms with sample data...")
            
            run_all_algorithms(sample_file)
            
        except Exception as e:
            print(f"Error creating sample data: {e}")
            import traceback
            traceback.print_exc()
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()