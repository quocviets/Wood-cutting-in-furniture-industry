import numpy as np

def first_fit_policy(observation, info):
    """
    Thuật toán First-Fit cho bài toán 2D Cutting Stock:
    - Duyệt lần lượt từng sản phẩm theo thứ tự trong danh sách
    - Đặt sản phẩm vào vị trí đầu tiên phù hợp trong stock đầu tiên có đủ chỗ trống
    - Nếu không có stock phù hợp, tạo một stock mới
    """
    list_prods = observation["products"]
    list_stocks = observation["stocks"]
    
    actions = []  # Danh sách các hành động đặt sản phẩm

    for prod in list_prods:
        prod_w, prod_h = prod["size"]
        
        while prod["quantity"] > 0:  # Đặt hết tất cả sản phẩm cùng loại
            placed = False

            for idx, stock in enumerate(list_stocks):
                stock_h, stock_w = stock.shape  # Kích thước của kho hàng

                # Kiểm tra xem stock có đủ chỗ không
                if stock_h < prod_w or stock_w < prod_h:
                    continue  # Kho quá nhỏ -> bỏ qua

                # Duyệt từng vị trí có thể đặt
                for x in range(stock_h - prod_w + 1):
                    for y in range(stock_w - prod_h + 1):
                        if np.all(stock[x:x + prod_w, y:y + prod_h] == -1):
                            # Đặt sản phẩm vào stock (đánh dấu bằng ID sản phẩm)
                            stock[x:x + prod_w, y:y + prod_h] = prod["id"]

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
                if placed:
                    break

            # Nếu không tìm thấy chỗ đặt, mở stock mới
            if not placed:
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