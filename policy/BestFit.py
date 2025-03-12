import numpy as np

def best_fit_policy(observation, info):
    """Best-Fit Algorithm for 2D Cutting-Stock Problem with visualization."""
    steps = []
    products = sorted(observation["products"], key=lambda x: x["width"] * x["height"], reverse=True)
    stocks = observation["stocks"]
    
    for prod in products:
        prod_w, prod_h = prod["width"], prod["height"]
        for _ in range(prod["quantity"]):
            best_stock_idx, best_x, best_y, min_waste = None, None, None, float("inf")
            
            for idx, stock in enumerate(stocks):
                stock_w, stock_h = stock["width"], stock["height"]
                stock_matrix = stock["matrix"]

                if stock_w < prod_w or stock_h < prod_h:
                    continue
                
                for x in range(stock_w - prod_w + 1):
                    for y in range(stock_h - prod_h + 1):
                        if np.all(stock_matrix[y:y+prod_h, x:x+prod_w] == -1):
                            waste = np.sum(stock_matrix == -1) - (prod_w * prod_h)
                            if waste < min_waste:
                                best_stock_idx, best_x, best_y, min_waste = idx, x, y, waste
            
            if best_stock_idx is not None:
                stock_matrix = stocks[best_stock_idx]["matrix"]
                stock_matrix[best_y:best_y+prod_h, best_x:best_x+prod_w] = prod["id"]
                steps.append({
                    "product_id": prod["id"],
                    "stock_idx": best_stock_idx,
                    "position": (best_x, best_y),
                    "size": (prod_w, prod_h)
                })
            else:
                new_stock = {
                    "width": max(prod_w, 10),
                    "height": max(prod_h, 10),
                    "matrix": np.full((max(prod_h, 10), max(prod_w, 10)), -1)
                }
                new_stock["matrix"][0:prod_h, 0:prod_w] = prod["id"]
                stocks.append(new_stock)
                steps.append({
                    "product_id": prod["id"],
                    "stock_idx": len(stocks) - 1,
                    "position": (0, 0),
                    "size": (prod_w, prod_h),
                    "new_stock": True
                })
    
    return stocks, steps
