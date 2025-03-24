import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Dữ liệu về các tấm nguyên liệu và yêu cầu cắt
stock_sheets = [
    {"id": 1, "length": 120, "width": 60},  # Tấm nguyên liệu 1 có kích thước 120x60
    {"id": 2, "length": 100, "width": 50},  # Tấm nguyên liệu 2 có kích thước 100x50
    {"id": 3, "length": 90, "width": 40},   # Tấm nguyên liệu 3 có kích thước 90x40
    {"id": 4, "length": 80, "width": 30}    # Tấm nguyên liệu 4 có kích thước 80x30
]

demand_pieces = [
    {"id": 1, "length": 50, "width": 30, "quantity": 4},  # Sản phẩm 1: Kích thước 50x30, cần 4 cái
    {"id": 2, "length": 40, "width": 20, "quantity": 6},  # Sản phẩm 2: Kích thước 40x20, cần 6 cái
    {"id": 3, "length": 60, "width": 50, "quantity": 2},  # Sản phẩm 3: Kích thước 60x50, cần 2 cái
    {"id": 4, "length": 30, "width": 20, "quantity": 8},  # Sản phẩm 4: Kích thước 30x20, cần 8 cái
    {"id": 5, "length": 70, "width": 40, "quantity": 3}   # Sản phẩm 5: Kích thước 70x40, cần 3 cái
]

def best_fit(stock_sheets, demand_pieces):
    """
    Thuật toán Best Fit: Đặt mỗi sản phẩm vào vị trí có lãng phí ít nhất trên tấm nguyên liệu.
    """
    placements = []  # Danh sách lưu trữ thông tin về vị trí các sản phẩm đã cắt
    used_stocks = set()  # Tập lưu trữ các tấm nguyên liệu đã sử dụng
    stock_layouts = {stock['id']: np.zeros((stock['width'], stock['length'])) for stock in stock_sheets}  # Đại diện bố cục tấm nguyên liệu
    leftover_pieces = 0  # Biến đếm số sản phẩm không thể đặt

    # Lặp qua các sản phẩm yêu cầu (sắp xếp theo diện tích để cắt các sản phẩm lớn trước)
    for piece in sorted(demand_pieces, key=lambda p: p['length'] * p['width'], reverse=True):
        for _ in range(piece['quantity']):
            best_stock = None  # Biến lưu tấm nguyên liệu phù hợp nhất
            min_waste = float('inf')  # Biến lưu lãng phí ít nhất (khởi tạo với giá trị vô cùng)
            best_x, best_y = None, None  # Vị trí tốt nhất để đặt sản phẩm
            
            # Lặp qua các tấm nguyên liệu để tìm vị trí đặt sản phẩm với lãng phí ít nhất
            for stock in stock_sheets:
                for y in range(stock['width'] - piece['width'] + 1):  # Duyệt qua các vị trí theo chiều cao
                    for x in range(stock['length'] - piece['length'] + 1):  # Duyệt qua các vị trí theo chiều dài
                        # Kiểm tra không gian có còn trống không (rỗng)
                        if np.all(stock_layouts[stock['id']][y:y+piece['width'], x:x+piece['length']] == 0):
                            waste = (stock['length'] * stock['width']) - (piece['length'] * piece['width'])  # Tính lãng phí
                            if waste < min_waste:  # Nếu lãng phí ít hơn, chọn vị trí này
                                min_waste = waste
                                best_stock = stock
                                best_x, best_y = x, y
            
            # Nếu tìm được tấm nguyên liệu và vị trí đặt, thực hiện cắt
            if best_stock:
                stock_layouts[best_stock['id']][best_y:best_y+piece['width'], best_x:best_x+piece['length']] = piece['id']
                placements.append({"stock_id": best_stock['id'], "piece_id": piece['id'],
                                   "x": best_x, "y": best_y, "length": piece['length'], "width": piece['width']})
                used_stocks.add(best_stock['id'])  # Đánh dấu tấm nguyên liệu là đã sử dụng
            else:
                leftover_pieces += 1  # Nếu không thể đặt, tăng số sản phẩm không thể đặt
                print(f"Không thể đặt sản phẩm {piece['id']}")

    return placements, used_stocks, leftover_pieces  # Trả về kết quả cắt, tấm nguyên liệu đã sử dụng, và số sản phẩm không thể đặt

def print_summary(placements):
    """
    In bảng tóm tắt kết quả cắt sản phẩm.
    """
    print("\nTóm tắt kết quả Best Fit Cutting:")
    print("+------------+----------+------------+-----------+")
    print("| Stock ID   | Piece ID | Kích thước | Vị trí    |")
    print("+------------+----------+------------+-----------+")
    for place in placements:
        print(f"| {place['stock_id']:^10} | {place['piece_id']:^8} | {place['length']}x{place['width']:^5} | ({place['x']},{place['y']}) |")
    print("+------------+----------+------------+-----------+")

def calculate_waste(stock_sheets, placements):
    """
    Tính toán và in ra lượng lãng phí trên mỗi tấm nguyên liệu đã sử dụng.
    """
    total_waste = 0  # Biến tổng lãng phí
    stock_usage = {stock['id']: 0 for stock in stock_sheets}  # Theo dõi diện tích đã sử dụng trên mỗi tấm nguyên liệu
    
    # Tính diện tích đã sử dụng cho mỗi tấm nguyên liệu
    for place in placements:
        stock_usage[place['stock_id']] += place['length'] * place['width']
    
    print("\nTóm tắt Lãng phí:")
    print("+------------+------------+------------+")
    print("| Stock ID   | Diện tích  | Diện tích  |")
    print("+------------+------------+------------+")
    
    # In ra diện tích đã sử dụng và lãng phí cho mỗi tấm nguyên liệu
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
            
            # Vẽ tấm nguyên liệu
            ax.add_patch(plt.Rectangle((0, 0), stock['length'], stock['width'],
                                       edgecolor='black', facecolor='gray', alpha=0.3))
            
            # Màu sắc cho các sản phẩm
            colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan', 'magenta', 'yellow']
            
            # Vẽ các sản phẩm đã cắt trên tấm nguyên liệu
            for i, place in enumerate(placements):
                if place['stock_id'] == stock['id']:
                    rect = plt.Rectangle((place['x'], place['y']), place['length'], place['width'],
                                         edgecolor='black', facecolor=colors[i % len(colors)], alpha=0.5)
                    ax.add_patch(rect)
                    ax.text(place['x'] + place['length']/2, place['y'] + place['width']/2,
                            f"{place['piece_id']}\n({place['length']}x{place['width']})", ha='center', va='center', fontsize=8, color='black')
            
            plt.gca().invert_yaxis()  # Lật trục Y để phù hợp với đồ họa
            plt.show()

def calculate_reward(leftover_pieces): 
    return -10 * leftover_pieces

# Thực thi thuật toán Best Fit
bf_placements, bf_used_stocks, leftover_pieces = best_fit(stock_sheets, demand_pieces)
reward = calculate_reward(leftover_pieces)  # Tính điểm thưởng dựa trên số sản phẩm không thể đặt

print(f"Total Reward: {reward}")  # In ra tổng điểm thưởng

# Hiển thị đồ họa cắt
visualize_cutting(stock_sheets, bf_placements, bf_used_stocks, "Best Fit Cutting Layout")

# In bảng tóm tắt kết quả cắt
print_summary(bf_placements)

# Tính toán và in ra lãng phí
calculate_waste(stock_sheets, bf_placements)
