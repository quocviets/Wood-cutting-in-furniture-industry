




class BestFit:
    def __init__(self, stock_sizes, products):
        self.stock_sizes = stock_sizes  # Danh sách kích thước kho chứa
        self.products = products  # Danh sách sản phẩm cần cắt
        self.stocks = []  # Danh sách các kho chứa đã sử dụng
    
    def solve(self):
        """Duyệt qua từng sản phẩm và đặt vào stock phù hợp nhất."""
        for product in self.products:
            best_stock = None
            min_space_left = float('inf')
            
            # Tìm stock phù hợp nhất
            for stock in self.stocks:
                space_left = stock.can_fit(product)
                if space_left is not None and space_left < min_space_left:
                    best_stock = stock
                    min_space_left = space_left
            
            # Nếu không có stock nào phù hợp, tạo một stock mới
            if best_stock is None:
                best_stock = StockSheet(self.stock_sizes[0])
                self.stocks.append(best_stock)
            
            best_stock.place_product(product)
    
    def visualize(self):
        """Trả về danh sách hình ảnh của các stock đã sử dụng."""
        return [stock.visualize() for stock in self.stocks]




class FirstFit:
    def __init__(self, stock_sizes, products):
        self.stock_sizes = stock_sizes  # Danh sách kích thước kho chứa
        self.products = sorted(products, key=lambda p: p[0] * p[1], reverse=True)  # Sắp xếp sản phẩm theo diện tích
        self.stocks = []  # Danh sách các kho chứa đã sử dụng
    
    def solve(self):
        """Duyệt từng sản phẩm và đặt vào stock đầu tiên có thể chứa nó."""
        for product in self.products:
            placed = False
            
            # Thử đặt vào stock có sẵn
            for stock in self.stocks:
                if stock.can_fit(product):
                    stock.place_product(product)
                    placed = True
                    break
            
            # Nếu không có stock phù hợp, mở một stock mới
            if not placed:
                new_stock = StockSheet(self.stock_sizes[0])
                new_stock.place_product(product)
                self.stocks.append(new_stock)
    
    def visualize(self):
        """Trả về danh sách hình ảnh của các stock đã sử dụng."""
        return [stock.visualize() for stock in self.stocks]


class StockSheet:
    def __init__(self, size):
        self.size = size
        self.occupied = []
    
    def can_fit(self, product):
        """Kiểm tra xem sản phẩm có thể đặt vào không."""
        # Giả định đơn giản: Nếu chưa có sản phẩm nào -> có thể đặt
        return (self.size[0] >= product[0] and self.size[1] >= product[1])
    
    def place_product(self, product):
        """Thêm sản phẩm vào stock."""
        self.occupied.append(product)
    
    def visualize(self):
        """Trả về một hình ảnh minh họa (để trống)."""
        return f"Stock {self.size} with {len(self.occupied)} products"
