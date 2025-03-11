import numpy as np

def compute_reward(stocks, wasted_space_penalty=0.1, used_stock_bonus=1.0, full_stock_bonus=2.0):
    """ Tính toán phần thưởng dựa trên hiệu suất sử dụng stock. """
    total_reward = 0
    total_stocks = len(stocks)
    used_stocks = sum(1 for stock in stocks if np.any(stock != -1))
    
    for stock in stocks:
        total_area = stock.shape[0] * stock.shape[1]
        used_area = np.sum(stock != -1)
        wasted_space = total_area - used_area
        
        reward = (used_area / total_area) * used_stock_bonus - (wasted_space / total_area) * wasted_space_penalty
        if used_area == total_area:
            reward += full_stock_bonus  # Thưởng nếu sử dụng hết một stock
        
        total_reward += reward
    
    # Thưởng nếu sử dụng ít stock hơn mức tối đa (tổng số stock là 8)
    stock_usage_bonus = (8 - used_stocks) * 0.5
    total_reward += stock_usage_bonus
    
    return total_reward

def define_state(stocks):
    """ Xác định trạng thái của môi trường dựa trên stock hiện tại. """
    return np.array([stock.flatten() for stock in stocks])

def compute_cost(stock, material_cost=1.0):
    """ Tính toán chi phí dựa trên diện tích sử dụng của stock. """
    used_area = np.sum(stock != -1)
    cost = used_area * material_cost
    return cost
