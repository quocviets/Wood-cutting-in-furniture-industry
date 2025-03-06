import numpy as np

def best_fit_policy(observation, info):
    """
    Best-Fit Algorithm for 2D Cutting-Stock Problem.
    - Tìm stock để lại ít diện tích dư thừa nhất sau khi đặt sản phẩm.
    - Nếu không có stock phù hợp, mở một stock mới.
    """
    list_prods = sorted(
        observation["products"], key=lambda x: x["size"][0] * x["size"][1], reverse=True
    )
    
    list_stocks = observation["stocks"]
    
    for prod in list_prods:
        prod_w, prod_h = prod["size"]
        
        for _ in range(prod["quantity"]):  # Xử lý từng sản phẩm
            best_stock_idx, best_pos_x, best_pos_y = None, None, None
            min_remaining_area = float("inf")
            
            # Tìm stock phù hợp nhất
            for idx, stock in enumerate(list_stocks):
                stock_w, stock_h = stock.shape
                
                if stock_w < prod_w or stock_h < prod_h:
                    continue  # Stock quá nhỏ
                
                for x in range(stock_w - prod_w + 1):
                    for y in range(stock_h - prod_h + 1):
                        if np.all(stock[x:x + prod_w, y:y + prod_h] == -1):
                            remaining_area = np.sum(stock == -1) - (prod_w * prod_h)
                            if remaining_area < min_remaining_area:
                                best_stock_idx = idx
                                best_pos_x, best_pos_y = x, y
                                min_remaining_area = remaining_area
            
            if best_stock_idx is not None:
                # Đặt sản phẩm vào stock tốt nhất
                list_stocks[best_stock_idx][best_pos_x:best_pos_x + prod_w, best_pos_y:best_pos_y + prod_h] = 1
            else:
                # Nếu không có stock nào phù hợp, tạo stock mới
                new_stock = np.full((max(prod_w, 5), max(prod_h, 5)), -1)
                new_stock[0:prod_w, 0:prod_h] = 1
                list_stocks.append(new_stock)
    
    return list_stocks  # Trả về danh sách stock đã cập nhật
