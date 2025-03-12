import numpy as np

def greedy_policy(observation, info):
    """
    Cải tiến thuật toán Greedy cho bài toán 2D Cutting Stock:
    - Ưu tiên đặt sản phẩm vào stock có diện tích trống gần nhất với kích thước sản phẩm.
    - Nếu không có stock phù hợp, tạo một stock mới với kích thước linh hoạt.
    - Tránh lãng phí không gian bằng cách kiểm tra kỹ các vị trí trước khi đặt.
    """
    
    # Sắp xếp sản phẩm theo diện tích giảm dần để đặt sản phẩm lớn trước
    list_prods = sorted(observation["products"], key=lambda p: p["size"][0] * p["size"][1], reverse=True)
    list_stocks = observation["stocks"]
    actions = []  # Danh sách các hành động đặt sản phẩm

    for prod in list_prods:
        prod_w, prod_h = prod["size"]
        quantity = prod["quantity"]
        
        for _ in range(quantity):
            best_stock_idx = None
            best_pos = None
            min_waste = float("inf")
            
            # Tìm stock phù hợp nhất
            for idx, stock in enumerate(list_stocks):
                stock_w, stock_h = stock["width"], stock["height"]  # Sửa lỗi shape
                stock_matrix = stock["matrix"]

                if stock_h < prod_h or stock_w < prod_w:
                    continue  # Nếu stock không đủ lớn, bỏ qua

                for y in range(stock_h - prod_h + 1):
                    for x in range(stock_w - prod_w + 1):
                        if np.all(stock_matrix[x:x + prod_h, y:y + prod_w] == -1):
                            waste = np.sum(stock_matrix == -1) - (prod_w * prod_h)
                            if waste < min_waste:
                                best_stock_idx = idx
                                best_pos = (x, y)
                                min_waste = waste

            if best_stock_idx is not None:
                # Đặt sản phẩm vào vị trí tối ưu
                x, y = best_pos
                list_stocks[best_stock_idx]["matrix"][x:x + prod_h, y:y + prod_w] = prod["id"]
                actions.append({
                    "stock_idx": best_stock_idx,
                    "size": (prod_w, prod_h),
                    "position": (x, y),
                    "product_id": prod["id"]
                })
            else:
                # Nếu không có stock phù hợp, tạo kho mới
                new_stock = {
                    "width": max(prod_w, 10),
                    "height": max(prod_h, 10),
                    "matrix": np.full((max(prod_w, 10), max(prod_h, 10)), -1)
                }
                new_stock["matrix"][0:prod_h, 0:prod_w] = prod["id"]
                new_stock_idx = len(list_stocks)
                list_stocks.append(new_stock)
                actions.append({
                    "stock_idx": new_stock_idx,
                    "size": (prod_w, prod_h),
                    "position": (0, 0),
                    "product_id": prod["id"],
                    "new_stock": True
                })
    
    return list_stocks, actions
