import matplotlib.pyplot as plt
import numpy as np

def sort_pieces_by_area(pieces):
    return sorted(pieces, key=lambda p: p['width'] * p['height'], reverse=True)

def first_fit_cutting(stocks, pieces):
    layouts = []
    
    for stock in stocks:
        stock_width, stock_height = stock['width'], stock['height']
        layout = []
        occupied = np.zeros((stock_height, stock_width), dtype=int)
        
        def can_place(x, y, w, h):
            if x + w > stock_width or y + h > stock_height:
                return False
            return np.sum(occupied[y:y+h, x:x+w]) == 0
        
        def place_piece(x, y, w, h, piece_id):
            occupied[y:y+h, x:x+w] = piece_id
            layout.append((piece_id, x, y, w, h))
        
        current_x, current_y = 0, 0
        max_row_height = 0
        
        for piece in pieces:
            for _ in range(piece['quantity']):
                w, h = piece['width'], piece['height']
                if can_place(current_x, current_y, w, h):
                    place_piece(current_x, current_y, w, h, piece['id'])
                    max_row_height = max(max_row_height, h)
                    current_x += w
                else:
                    current_x = 0
                    current_y += max_row_height
                    max_row_height = h
                    if can_place(current_x, current_y, w, h):
                        place_piece(current_x, current_y, w, h, piece['id'])
                        current_x += w
                    else:
                        print(f"Không thể đặt mảnh {piece['name']} vào stock!")
                        break
        layouts.append((stock, layout))
    return layouts

def visualize_cutting(stocks, layouts):
    for i, (stock, layout) in enumerate(layouts):
        fig, ax = plt.subplots(figsize=(6, 10))
        ax.set_xlim(0, stock['width'])
        ax.set_ylim(0, stock['height'])
        ax.set_xticks(range(0, stock['width']+1, 10))
        ax.set_yticks(range(0, stock['height']+1, 10))
        ax.grid(True, linestyle='--', alpha=0.7)
        
        for piece_id, x, y, w, h in layout:
            ax.add_patch(plt.Rectangle((x, y), w, h, edgecolor='black', facecolor=np.random.rand(3,), lw=2))
            ax.text(x + w/2, y + h/2, str(piece_id), ha='center', va='center', fontsize=12, color='white', weight='bold')
        
        ax.set_title(f"Cutting Stock Optimization - Stock {i+1}")
        plt.gca().invert_yaxis()
        plt.show()

# Dữ liệu đầu vào
stocks = [
    {"width": 50, "height": 100},
    {"width": 50, "height": 100},
    {"width": 30, "height": 50}
]

pieces = [
    {"id": 1, "name": "Table", "width": 40, "height": 75, "quantity": 2},
    {"id": 2, "name": "Chair", "width": 20, "height": 25, "quantity": 4},
    {"id": 3, "name": "Chair Leg", "width": 4, "height": 25, "quantity": 8}
]

# Chạy thuật toán
sorted_pieces = sort_pieces_by_area(pieces)
layouts = first_fit_cutting(stocks, sorted_pieces)
visualize_cutting(stocks, layouts)
