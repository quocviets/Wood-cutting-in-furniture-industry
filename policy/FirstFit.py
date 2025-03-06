import numpy as np

def first_fit_policy(observation, info):
    """
    First-Fit Algorithm for 2D Cutting Stock Problem
    - Duyệt qua từng sản phẩm và đặt vào stock đầu tiên có thể chứa nó.
    - Nếu không tìm thấy stock phù hợp, mở một stock mới.
    """
    list_prods = sorted(
        observation["products"], key=lambda x: x["size"][0] * x["size"][1], reverse=True
    )
    
    list_stocks = sorted(
        enumerate(observation["stocks"]),
        key=lambda x: np.sum(x[1] != -2),  # Đếm số ô khả dụng trong stock
        reverse=True
    )
    
    for prod in list_prods:
        prod_w, prod_h = prod["size"]
        
        for _ in range(prod["quantity"]):  # Đặt từng sản phẩm
            placed = False  # Đánh dấu xem sản phẩm có được đặt không
            
            for idx, stock in list_stocks:
                stock_w, stock_h = stock.shape

                if stock_w < prod_w or stock_h < prod_h:
                    continue  # Nếu stock không đủ lớn, bỏ qua
                
                for x in range(stock_w - prod_w + 1):
                    for y in range(stock_h - prod_h + 1):
                        if np.all(stock[x:x + prod_w, y:y + prod_h] == -1):
                            # Đặt sản phẩm vào stock
                            stock[x:x + prod_w, y:y + prod_h] = 1  
                            placed = True
                            break
                    if placed:
                        break
                
                if placed:
                    break  # Nếu đã đặt sản phẩm, thoát vòng lặp stock

            if not placed:
                # Nếu không tìm thấy vị trí, tạo kho mới
                new_stock = np.full((max(prod_w, 5), max(prod_h, 5)), -1)  # Kho mới có kích thước tối thiểu
                new_stock[0:prod_w, 0:prod_h] = 1
                observation["stocks"].append(new_stock)
    
    return observation["stocks"]  # Trả về danh sách kho sau khi cập nhật
