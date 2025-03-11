import numpy as np

def first_fit_policy(observation, info):
    """First-Fit Algorithm for 2D Cutting-Stock Problem with step tracking."""
    products = sorted(observation["products"], key=lambda x: x["size"], reverse=True)
    stocks = observation["stocks"]
    steps = []

    for prod in products:
        prod_w, prod_h = prod["size"]
        for _ in range(prod["quantity"]):
            placed = False

            for idx, stock in enumerate(stocks):
                stock_w, stock_h = stock["width"], stock["height"]  # Sửa lỗi shape
                stock_matrix = stock["matrix"]

                if stock_w < prod_w or stock_h < prod_h:
                    continue

                for x in range(stock_w - prod_w + 1):
                    for y in range(stock_h - prod_h + 1):
                        if np.all(stock_matrix[x:x + prod_w, y:y + prod_h] == -1):
                            stock_matrix[x:x + prod_w, y:y + prod_h] = prod["id"]
                            placed = True
                            steps.append({
                                "product_id": prod["id"],
                                "stock_idx": idx,
                                "position": (x, y),
                                "size": (prod_w, prod_h)
                            })
                            break
                    if placed:
                        break
                if placed:
                    break

            if not placed:
                new_stock = {
                    "width": max(prod_w, 10),
                    "height": max(prod_h, 10),
                    "matrix": np.full((max(prod_w, 10), max(prod_h, 10)), -1)
                }
                new_stock["matrix"][0:prod_w, 0:prod_h] = prod["id"]
                stocks.append(new_stock)
                steps.append({
                    "product_id": prod["id"],
                    "stock_idx": len(stocks) - 1,
                    "position": (0, 0),
                    "size": (prod_w, prod_h),
                    "new_stock": True
                })

    return stocks, steps
