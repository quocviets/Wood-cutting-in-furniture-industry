import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Định nghĩa lớp Piece - đại diện cho mỗi mảnh cần cắt
class Piece:
    def __init__(self, length, width, id):
        self.length = length
        self.width = width
        self.id = id
        self.placed = False  # Trạng thái đã được đặt chưa
        self.x = 0  # Vị trí x trên tấm nguyên liệu
        self.y = 0  # Vị trí y trên tấm nguyên liệu
        self.rotated = False  # Trạng thái xoay của mảnh

    def area(self):
        return self.length * self.width  # Diện tích của mảnh cần cắt

    def rotate(self):
        # Xoay mảnh cắt (đổi chiều dài và rộng)
        self.length, self.width = self.width, self.length
        self.rotated = not self.rotated  # Cập nhật trạng thái xoay

# Định nghĩa lớp StockSheet - đại diện cho tấm nguyên liệu
class StockSheet:
    def __init__(self, length, width, stock_id):
        self.length = length
        self.width = width
        self.stock_id = stock_id
        self.placed_pieces = []  # Danh sách các mảnh đã được đặt lên tấm
        self.layout = np.zeros((length, width), dtype=int)  # Ma trận bố trí mảnh cắt trên tấm

    # Kiểm tra có thể đặt mảnh lên tấm không
    def can_place(self, piece):
        for rotate in [False, True]:  # Kiểm tra hai trạng thái: không xoay và có xoay
            if rotate:
                piece.rotate()
            for x in range(self.length - piece.length + 1):
                for y in range(self.width - piece.width + 1):
                    if self.check_fit(piece, x, y):  # Kiểm tra xem mảnh có vừa không
                        return True, x, y, piece.rotated
            if rotate:  # Trả lại trạng thái cũ nếu đã xoay
                piece.rotate()
        return False, -1, -1, False  # Không đặt được

    # Kiểm tra mảnh cắt có đè lên mảnh khác không
    def check_fit(self, piece, x, y):
        return np.all(self.layout[x:x + piece.length, y:y + piece.width] == 0)

    # Đặt mảnh cắt vào vị trí x, y trên tấm nguyên liệu
    def place(self, piece, x, y):
        self.layout[x:x + piece.length, y:y + piece.width] = piece.id
        piece.x = x
        piece.y = y
        piece.placed = True  # Đánh dấu đã đặt
        self.placed_pieces.append(piece)  # Thêm mảnh vào danh sách đã đặt

    # Tính diện tích đã sử dụng trên tấm nguyên liệu
    def used_area(self):
        return sum([p.area() for p in self.placed_pieces])

class Order:
    def __init__(self, order_id, stock_size, items):
        self.order_id = order_id
        self.stock_size = stock_size
        self.pieces = []  # Danh sách các mảnh cần cắt
        self.total_area = 0  # Tổng diện tích các mảnh cần cắt
        self.create_pieces(items)  # Tạo các mảnh dựa trên yêu cầu của đơn hàng

    # Tạo các mảnh cần cắt từ yêu cầu của đơn hàng
    def create_pieces(self, items):
        pid = 1
        for name, info in items.items():
            l, w = info['size']
            for _ in range(info['quantity']):
                self.pieces.append(Piece(l, w, pid))
                self.total_area += l * w  # Cộng diện tích các mảnh cần cắt
            pid += 1

# Hàm thực thi giải thuật kết hợp First-Fit và Best-Fit
def run_combination_heuristic(order):
    start = time.time()
    length, width = order.stock_size
    stock_sheets = []  # Danh sách các tấm nguyên liệu

    for piece in order.pieces:
        placed = False  # Trạng thái đặt mảnh cắt

        # Áp dụng thuật toán First-Fit
        for stock in stock_sheets:
            fit, x, y, rotated = stock.can_place(piece)
            if fit:
                stock.place(piece, x, y)
                placed = True
                break

        # Nếu không đặt được, áp dụng Best-Fit
        if not placed:
            best_stock = None
            best_x, best_y, best_rotated = -1, -1, False
            min_waste = float('inf')  # Lãng phí tối thiểu

            for stock in stock_sheets:
                fit, x, y, rotated = stock.can_place(piece)
                if fit:
                    projected_used = stock.used_area() + piece.area()  # Diện tích dự kiến
                    projected_waste = (stock.length * stock.width) - projected_used
                    if projected_waste < min_waste:
                        min_waste = projected_waste
                        best_stock = stock
                        best_x, best_y = x, y
                        best_rotated = rotated

            if best_stock:
                best_stock.place(piece, best_x, best_y)
                placed = True

        # Mở tấm nguyên liệu mới nếu không đặt được
        if not placed:
            new_stock = StockSheet(length, width, f"{order.order_id}_{len(stock_sheets) + 1}")
            fit, x, y, rotated = new_stock.can_place(piece)
            if fit:
                new_stock.place(piece, x, y)
                stock_sheets.append(new_stock)

    end = time.time()

     # Tính toán hiệu suất và tỷ lệ lãng phí
    total_stock_area = len(stock_sheets) * length * width
    total_used_area = sum([s.used_area() for s in stock_sheets])
    total_cut_items_area = order.total_area
    total_unused_area = total_stock_area - total_used_area

    waste_rate = round(total_unused_area / total_cut_items_area, 4)
    fitness = round(total_cut_items_area / total_stock_area, 4)
    runtime = round(end - start, 4)

    return {
        'Order ID': order.order_id,
        'Stock Count': len(stock_sheets),
        'Waste Rate': waste_rate,
        'Fitness': fitness,
        'Runtime (s)': runtime
    }




orders = [
    {
        "order_id": "order_001",
        "stock_size": [100, 100],
        "items": {
            "table": { "size": [40, 50], "quantity": 2 },
            "chair": { "size": [20, 25], "quantity": 4 },
            "leg":   { "size": [4, 25],  "quantity": 10 }
        }
    },
    {
        "order_id": "order_002",
        "stock_size": [100, 100],
        "items": {
            "table": { "size": [40, 50], "quantity": 1 },
            "chair": { "size": [20, 25], "quantity": 6 },
            "leg":   { "size": [4, 25],  "quantity": 20 }
        }
    },
    {
        "order_id": "order_003",
        "stock_size": [100, 100],
        "items": {
            "table": { "size": [40, 50], "quantity": 3 },
            "chair": { "size": [20, 25], "quantity": 6 },
            "leg":   { "size": [4, 25],  "quantity": 25 }
        }
    },
    {
        "order_id": "order_004",
        "stock_size": [100, 100],
        "items": {
            "table": { "size": [40, 50], "quantity": 4 },
            "chair": { "size": [20, 25], "quantity": 8 },
            "leg":   { "size": [4, 25],  "quantity": 30 }
        }
    }
]

print("\n📦 Combination Heuristic Benchmark Results")
print("-" * 50)
print(f"{'Order ID':<12}{'Stock Count':<15}{'Waste Rate':<15}{'Fitness':<10}{'Runtime (s)':<12}")
print("-" * 50)

for o in orders:
    order_obj = Order(o["order_id"], o["stock_size"], o["items"])
    result = run_combination_heuristic(order_obj)
    print(f"{result['Order ID']:<12}{result['Stock Count']:<15}{result['Waste Rate']:<15}"
          f"{result['Fitness']:<10}{result['Runtime (s)']:<12}")



