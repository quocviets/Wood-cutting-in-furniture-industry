import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Định nghĩa các stock sheet (tấm nguyên liệu) với kích thước
stock_sheets = [
    {"id": 1, "length": 120, "width": 60},  # Tấm nguyên liệu 1 có kích thước 120x60
    {"id": 2, "length": 100, "width": 50},  # Tấm nguyên liệu 2 có kích thước 100x50
    {"id": 3, "length": 90, "width": 40},   # Tấm nguyên liệu 3 có kích thước 90x40
    {"id": 4, "length": 80, "width": 30}    # Tấm nguyên liệu 4 có kích thước 80x30
]

# Định nghĩa các sản phẩm yêu cầu với kích thước và số lượng cần cắt
demand_pieces = [
    {"id": 1, "length": 50, "width": 30, "quantity": 4},  # Sản phẩm 1: Kích thước 50x30, cần 4 cái
    {"id": 2, "length": 40, "width": 20, "quantity": 6},  # Sản phẩm 2: Kích thước 40x20, cần 6 cái
    {"id": 3, "length": 60, "width": 50, "quantity": 2},  # Sản phẩm 3: Kích thước 60x50, cần 2 cái
    {"id": 4, "length": 30, "width": 20, "quantity": 8},  # Sản phẩm 4: Kích thước 30x20, cần 8 cái
    {"id": 5, "length": 70, "width": 40, "quantity": 3}   # Sản phẩm 5: Kích thước 70x40, cần 3 cái
]

def first_fit(stock_sheets, demand_pieces):
    """
    Thuật toán First Fit: Đặt mỗi sản phẩm vào vị trí đầu tiên có thể trên tấm nguyên liệu.
    """
    placements = []  # Danh sách lưu trữ thông tin về vị trí các sản phẩm đã được đặt
    used_stocks = set()  # Tập lưu trữ các tấm nguyên liệu đã được sử dụng
    reward = 0  # Khởi tạo điểm thưởng (reward)

    # Tạo đại diện cho bố cục của mỗi tấm nguyên liệu
    stock_layouts = {stock['id']: np.zeros((stock['width'], stock['length'])) for stock in stock_sheets}
    
    # Lặp qua các sản phẩm yêu cầu
    for piece in demand_pieces:
        for _ in range(piece['quantity']):
            placed = False  # Cờ kiểm tra xem sản phẩm đã được đặt chưa
            
            # Cố gắng đặt sản phẩm vào vị trí đầu tiên trên tấm nguyên liệu có thể chứa
            for stock in stock_sheets:
                for y in range(stock['width'] - piece['width'] + 1):
                    for x in range(stock['length'] - piece['length'] + 1):
                        # Kiểm tra nếu không gian còn trống (rỗng)
                        if np.all(stock_layouts[stock['id']][y:y+piece['width'], x:x+piece['length']] == 0):
                            # Đặt sản phẩm vào tấm nguyên liệu
                            stock_layouts[stock['id']][y:y+piece['width'], x:x+piece['length']] = piece['id']
                            placements.append({"stock_id": stock['id'], "piece_id": piece['id'],
                                               "x": x, "y": y, "length": piece['length'], "width": piece['width']})
                            used_stocks.add(stock['id'])  # Đánh dấu tấm nguyên liệu là đã sử dụng
                            placed = True  # Đánh dấu sản phẩm đã được đặt
                            break
                    if placed:
                        break
                if placed:
                    break
            
            # Nếu không tìm được vị trí đặt, giảm điểm thưởng
            if not placed:
                print(f"Không thể đặt sản phẩm {piece['id']}")
                reward -= 10  # Trừ 10 điểm cho mỗi sản phẩm không thể đặt
    
    return placements, used_stocks, reward

def calculate_waste(stock_sheets, placements):
    """
    Tính toán và in ra lượng lãng phí trên mỗi tấm nguyên liệu đã sử dụng.
    """
    total_waste = 0
    stock_usage = {stock['id']: 0 for stock in stock_sheets}  # Theo dõi diện tích đã sử dụng trên mỗi tấm nguyên liệu
    
    # Tính diện tích đã sử dụng cho mỗi tấm nguyên liệu
    for place in placements:
        stock_usage[place['stock_id']] += place['length'] * place['width']
    
    print("\nTóm tắt Lãng phí:")
    print("+------------+------------+------------+")
    print("| Stock ID   | Diện tích  | Diện tích  |")
    print("+------------+------------+------------+")
    
    for stock in stock_sheets:
        if stock['id'] in stock_usage:
            total_area = stock['length'] * stock['width']
            used_area = stock_usage[stock['id']]
            waste_area = total_area - used_area
            total_waste += waste_area
            print(f"| {stock['id']:^10} | {used_area:^10} | {waste_area:^10} |")
    
    print("+------------+------------+------------+")
    print(f"Tổng diện tích lãng phí: {total_waste} đơn vị diện tích\n")
    return total_waste

def visualize_cutting(stock_sheets, placements, used_stocks, title):
    """
    Hiển thị đồ họa bố cục cắt trên các tấm nguyên liệu đã sử dụng với các kích thước của sản phẩm.
    """
    for stock in stock_sheets:
        if stock['id'] in used_stocks:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.set_xlim(0, stock['length'])
            ax.set_ylim(0, stock['width'])
            ax.set_title(f"Bố cục Tấm {stock['id']} ({stock['length']}x{stock['width']})")
            ax.set_xticks([])  # Ẩn thang đo trục X
            ax.set_yticks([])  # Ẩn thang đo trục Y
            
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
            
            plt.gca().invert_yaxis()  # Lật trục Y để phù hợp với đồ họa
            plt.show()

def print_summary(placements):
    """
    In ra bảng tóm tắt kết quả cắt.
    """
    print("\nTóm tắt Cutting First Fit:")
    print("+------------+----------+------------+-----------+")
    print("| Stock ID   | Piece ID | Kích thước | Vị trí    |")
    print("+------------+----------+------------+-----------+")
    for place in placements:
        print(f"| {place['stock_id']:^10} | {place['piece_id']:^8} | {place['length']}x{place['width']:^5} | ({place['x']},{place['y']}) |")
    print("+------------+----------+------------+-----------+")

# Thực thi thuật toán First Fit
ff_placements, ff_used_stocks, ff_reward = first_fit(stock_sheets, demand_pieces)
print(f"\nTổng điểm thưởng: {ff_reward}")

# Hiển thị đồ họa cắt
visualize_cutting(stock_sheets, ff_placements, ff_used_stocks, "First Fit Cutting Layout")

# In bảng tóm tắt kết quả
print_summary(ff_placements)

# Tính toán và in lãng phí
calculate_waste(stock_sheets, ff_placements)
