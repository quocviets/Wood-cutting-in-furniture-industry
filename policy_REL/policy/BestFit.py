import numpy as np

def best_fit_policy(observation, info):
    """
    Thuật toán Best-Fit cho bài toán 2D Cutting Stock:
    - Duyệt lần lượt từng sản phẩm trong danh sách
    - Đặt sản phẩm vào vị trí tốt nhất (có không gian thừa ít nhất) trong tất cả các stock
    - Nếu không có stock phù hợp, tạo một stock mới
    """
    list_prods = observation["products"]
    list_stocks = observation["stocks"]
    
    actions = []  # Danh sách các hành động đặt sản phẩm

    for prod in list_prods:
        prod_w, prod_h = prod["size"]
        
        while prod["quantity"] > 0:  # Đặt hết tất cả sản phẩm cùng loại
            best_position = None
            best_stock_idx = None
            min_waste = float('inf')  # Khởi tạo giá trị lớn nhất để tìm vị trí tốt nhất
            
            # Duyệt qua tất cả các stock để tìm vị trí tốt nhất
            for idx, stock in enumerate(list_stocks):
                stock_h, stock_w = stock.shape  # Kích thước của kho hàng
                
                # Kiểm tra xem stock có đủ chỗ không
                if stock_h < prod_w or stock_w < prod_h:
                    continue  # Kho quá nhỏ -> bỏ qua
                
                # Duyệt từng vị trí có thể đặt
                for x in range(stock_h - prod_w + 1):
                    for y in range(stock_w - prod_h + 1):
                        if np.all(stock[x:x + prod_w, y:y + prod_h] == -1):
                            # Tính toán "waste" (lãng phí) cho vị trí này
                            # Phương pháp đơn giản: tính số ô bị cô lập xung quanh
                            waste = 0
                            
                            # Kiểm tra biên trên
                            if x > 0:
                                waste += sum(1 for i in range(y, y + prod_h) if stock[x-1, i] == -1)
                            
                            # Kiểm tra biên dưới
                            if x + prod_w < stock_h:
                                waste += sum(1 for i in range(y, y + prod_h) if stock[x+prod_w, i] == -1)
                            
                            # Kiểm tra biên trái
                            if y > 0:
                                waste += sum(1 for i in range(x, x + prod_w) if stock[i, y-1] == -1)
                            
                            # Kiểm tra biên phải
                            if y + prod_h < stock_w:
                                waste += sum(1 for i in range(x, x + prod_w) if stock[i, y+prod_h] == -1)
                            
                            if waste < min_waste:
                                min_waste = waste
                                best_position = (x, y)
                                best_stock_idx = idx
            
            # Đặt sản phẩm vào vị trí tốt nhất nếu có
            if best_position is not None:
                x, y = best_position
                stock = list_stocks[best_stock_idx]
                
                stock[x:x + prod_w, y:y + prod_h] = prod["id"]
                
                actions.append({
                    "stock_idx": best_stock_idx,
                    "size": (prod_w, prod_h),
                    "position": (x, y),
                    "product_id": prod["id"]
                })
                
                prod["quantity"] -= 1  # Giảm số lượng sản phẩm
            else:
                # Nếu không tìm thấy chỗ đặt, mở stock mới
                new_stock_idx = len(list_stocks)
                new_stock = np.full((max(6, prod_w), max(6, prod_h)), -1)  # Tạo kho mới
                list_stocks.append(new_stock)
                
                # Đặt sản phẩm vào kho mới
                new_stock[0:prod_w, 0:prod_h] = prod["id"]
                
                actions.append({
                    "stock_idx": new_stock_idx,
                    "size": (prod_w, prod_h),
                    "position": (0, 0),
                    "product_id": prod["id"]
                })
                
                prod["quantity"] -= 1  # Giảm số lượng sản phẩm
    
    return actions