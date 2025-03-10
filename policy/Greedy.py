import numpy as np

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
