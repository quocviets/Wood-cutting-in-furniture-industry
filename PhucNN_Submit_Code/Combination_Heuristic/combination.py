import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class Piece:
    def __init__(self, length, width, id):
        self.length = length
        self.width = width
        self.id = id
        self.placed = False
        self.x = 0
        self.y = 0
        self.rotated = False

    def area(self):
        return self.length * self.width

    def rotate(self):
        self.length, self.width = self.width, self.length
        self.rotated = not self.rotated

class StockSheet:
    def __init__(self, length, width, stock_id):
        self.length = length
        self.width = width
        self.stock_id = stock_id
        self.placed_pieces = []
        self.layout = np.zeros((length, width), dtype=int)

    def can_place(self, piece):
        for rotate in [False, True]:
            if rotate:
                piece.rotate()
            for x in range(self.length - piece.length + 1):
                for y in range(self.width - piece.width + 1):
                    if self.check_fit(piece, x, y):
                        return True, x, y, piece.rotated
            if rotate:
                piece.rotate()
        return False, -1, -1, False

    def check_fit(self, piece, x, y):
        return np.all(self.layout[x:x+piece.length, y:y+piece.width] == 0)

    def place(self, piece, x, y):
        self.layout[x:x+piece.length, y:y+piece.width] = piece.id
        piece.x = x
        piece.y = y
        piece.placed = True
        self.placed_pieces.append(piece)

    def used_area(self):
        return sum([p.area() for p in self.placed_pieces])

class Order:
    def __init__(self, order_id, stock_size, items):
        self.order_id = order_id
        self.stock_size = stock_size
        self.pieces = []
        self.total_area = 0
        self.create_pieces(items)

    def create_pieces(self, items):
        pid = 1
        for name, info in items.items():
            l, w = info['size']
            for _ in range(info['quantity']):
                self.pieces.append(Piece(l, w, pid))
                self.total_area += l * w
            pid += 1

def run_combination_heuristic(order):
    import time
    start = time.time()
    length, width = order.stock_size
    stock_sheets = []

    for piece in order.pieces:
        placed = False

        # Try First Fit
        for stock in stock_sheets:
            fit, x, y, rotated = stock.can_place(piece)
            if fit:
                stock.place(piece, x, y)
                placed = True
                break

        if not placed:
            # Try Best Fit
            best_stock = None
            best_x, best_y, best_rotated = -1, -1, False
            min_waste = float('inf')

            for stock in stock_sheets:
                fit, x, y, rotated = stock.can_place(piece)
                if fit:
                    projected_used = stock.used_area() + piece.area()
                    projected_waste = (stock.length * stock.width) - projected_used
                    if projected_waste < min_waste:
                        min_waste = projected_waste
                        best_stock = stock
                        best_x, best_y = x, y
                        best_rotated = rotated

            if best_stock:
                best_stock.place(piece, best_x, best_y)
                placed = True

        if not placed:
            # Open new stock if all else fails
            new_stock = StockSheet(length, width, f"{order.order_id}_{len(stock_sheets)+1}")
            fit, x, y, rotated = new_stock.can_place(piece)
            if fit:
                new_stock.place(piece, x, y)
                stock_sheets.append(new_stock)

    end = time.time()

    total_stock_area = len(stock_sheets) * length * width
    total_used_area = sum([s.used_area() for s in stock_sheets])
    total_cut_items_area = order.total_area
    total_unused_area = total_stock_area - total_used_area

    # Theo công thức chuẩn
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




# ==== Đơn hàng mẫu ====
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

# ==== Thực thi ====
print("\n📦 Combination Heuristic Benchmark Results")
print("-" * 50)
print(f"{'Order ID':<12}{'Stock Count':<15}{'Waste Rate':<15}{'Fitness':<10}{'Runtime (s)':<12}")
print("-" * 50)

for o in orders:
    order_obj = Order(o["order_id"], o["stock_size"], o["items"])
    result = run_combination_heuristic(order_obj)
    print(f"{result['Order ID']:<12}{result['Stock Count']:<15}{result['Waste Rate']:<15}"
          f"{result['Fitness']:<10}{result['Runtime (s)']:<12}")



