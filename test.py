import numpy as np
import random
import copy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pickle
import os

# ============================
# DỮ LIỆU
# ============================
BOARD_WIDTH = 100
BOARD_LENGTH = 200

# Một tấm gỗ
stock_sheet = {
    "id": 1,
    "width": BOARD_WIDTH,
    "length": BOARD_LENGTH
}

# Danh sách sản phẩm
# Bàn x 2 (40×75)
# Chân bàn x 16 (4×25)
# Ghế x 4 (20×25)
products_data = [
    {"id": 1, "name": "Table",    "width": 40, "length": 75, "quantity": 2},
    {"id": 2, "name": "TableLeg", "width": 4,  "length": 25, "quantity": 16},
    {"id": 3, "name": "Chair",    "width": 20, "length": 25, "quantity": 4}
]

# ============================
# HÀM KIỂM TRA SÁT CẠNH
# ============================
def is_adjacent(pos1, size1, pos2, size2):
    """
    Kiểm tra xem 2 hình chữ nhật có sát cạnh nhau (share 1 biên) hay không.
    pos: (row, col)
    size: (width, height)
    """
    r1, c1 = pos1
    w1, h1 = size1
    r2, c2 = pos2
    w2, h2 = size2

    top1, bottom1, left1, right1 = r1, r1 + h1, c1, c1 + w1
    top2, bottom2, left2, right2 = r2, r2 + h2, c2, c2 + w2

    horizontal_adj = ((bottom1 > top2) and (top1 < bottom2)) and (abs(right1 - left2) == 0 or abs(right2 - left1) == 0)
    vertical_adj   = ((right1 > left2)   and (left1 < right2))   and (abs(bottom1 - top2) == 0 or abs(bottom2 - top1) == 0)
    return horizontal_adj or vertical_adj

# ============================
# MÔI TRƯỜNG Q-LEARNING
# ============================
class QLearningCuttingEnv:
    def __init__(self):
        # Board duy nhất
        self.board = {
            "id": stock_sheet["id"],
            "width": stock_sheet["width"],
            "length": stock_sheet["length"],
            "grid": np.full((stock_sheet["length"], stock_sheet["width"]), -1)
        }
        # Sao chép dữ liệu sản phẩm
        self.products = [dict(p) for p in products_data]
        # Lưu các bước đặt
        self.placements = []  # Mỗi item: {"piece_id", "pos", "size"}

    def reset(self):
        # Reset grid
        self.board["grid"] = np.full((self.board["length"], self.board["width"]), -1)
        # Reset lại quantity
        for p in self.products:
            if p["id"] == 1:
                p["quantity"] = 2
            elif p["id"] == 2:
                p["quantity"] = 16
            elif p["id"] == 3:
                p["quantity"] = 4
        self.placements = []
        return self.get_state()

    def get_state(self):
        """
        State = grid flatten + quantity các sản phẩm.
        """
        flatten_grid = self.board["grid"].flatten().tolist()
        q_list = [p["quantity"] for p in self.products]
        return tuple(flatten_grid + q_list)

    def is_done(self):
        """
        Xác định khi nào xong: không còn sản phẩm nào cần cắt.
        """
        return not any(p["quantity"] > 0 for p in self.products)

    def get_possible_actions(self):
        """
        Trả về danh sách (product_id) có quantity > 0.
        """
        actions = []
        for p in self.products:
            if p["quantity"] > 0:
                actions.append(p["id"])
        return actions

    def step(self, action):
        """
        Đặt sản phẩm (action = product_id) lên board.
        Nếu đặt được thì tính reward, nếu không => reward = -1.
        """
        product = None
        for p in self.products:
            if p["id"] == action:
                product = p
                break
        if not product:
            return self.get_state(), -1

        w = product["width"]
        h = product["length"]
        grid = self.board["grid"]
        b_w = self.board["width"]
        b_h = self.board["length"]

        placed = False
        pos_placed = None

        # Tìm chỗ trống theo thứ tự (top->down, left->right)
        for row in range(b_h - h + 1):
            for col in range(b_w - w + 1):
                if np.all(grid[row:row+h, col:col+w] == -1):
                    # Phải sát ít nhất 1 cạnh board
                    if not ((row == 0) or (col == 0) or (row + h == b_h) or (col + w == b_w)):
                        continue
                    grid[row:row+h, col:col+w] = action
                    placed = True
                    pos_placed = (row, col)
                    break
            if placed:
                break

        if placed:
            product["quantity"] -= 1
            self.placements.append({
                "piece_id": action,
                "pos": pos_placed,
                "size": (w, h)
            })
            reward = self.compute_reward(action, pos_placed, (w, h))
        else:
            reward = -1

        return self.get_state(), reward

    def compute_reward(self, piece_id, pos, size):
        """
        Tính reward theo các quy tắc:
          (a) Bàn đầu (ID=1, lần đầu): +10 nếu đặt ở góc, else +1.
          (b) Bàn thứ hai: +8 nếu sát bàn đầu và sát cạnh board, else +1.
          (c) Các sản phẩm khác: +5 nếu sát cạnh board hoặc sát 1 miếng đã đặt, else +1.
        """
        if piece_id == 1:
            placed_tables = [p for p in self.placements if p["piece_id"] == 1]
            count_table = len(placed_tables)
            if count_table == 1:
                corners = [
                    (0, 0),
                    (0, self.board["width"] - size[0]),
                    (self.board["length"] - size[1], 0),
                    (self.board["length"] - size[1], self.board["width"] - size[0])
                ]
                if pos in corners:
                    return 10
                else:
                    return 1
            elif count_table == 2:
                first_table = placed_tables[0]
                sat_first = is_adjacent(pos, size, first_table["pos"], first_table["size"])
                row, col = pos
                w, h = size
                touches_board = (row == 0) or (col == 0) or (row + h == self.board["length"]) or (col + w == self.board["width"])
                if sat_first and touches_board:
                    return 8
                else:
                    return 1
            else:
                return self.reward_for_other(pos, size)
        else:
            return self.reward_for_other(pos, size)

    def reward_for_other(self, pos, size):
        """
        +5 nếu sát cạnh board hoặc sát 1 miếng đã đặt, else +1.
        """
        row, col = pos
        w, h = size
        if (row == 0) or (col == 0) or (row + h == self.board["length"]) or (col + w == self.board["width"]):
            return 5

        for p in self.placements:
            if p["pos"] == pos and p["size"] == size:
                continue
            if is_adjacent(pos, size, p["pos"], p["size"]):
                return 5
        return 1

    def leftover_penalty(self):
        """
        Phạt theo diện tích thừa: -0.01 điểm cho mỗi đơn vị diện tích chưa cắt.
        """
        leftover = np.sum(self.board["grid"] == -1)
        return -0.01 * leftover

    def complete_bonus(self):
        """
        Thưởng +50 nếu toàn bộ demand đã được cắt.
        """
        if all(p["quantity"] == 0 for p in self.products):
            return 50
        return 0

# ============================
# TÁC NHÂN Q-LEARNING
# ============================
class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.2):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {}  # Dictionary lưu Q(s,a)

    def get_state_key(self, state):
        return tuple(state)

    def choose_action(self, state, possible_actions):
        """
        Chọn action theo epsilon-greedy.
        """
        state_key = self.get_state_key(state)
        if random.random() < self.epsilon:
            return random.choice(possible_actions)
        q_values = [self.q_table.get((state_key, a), 0) for a in possible_actions]
        max_q = max(q_values)
        max_actions = [a for a, qv in zip(possible_actions, q_values) if qv == max_q]
        return random.choice(max_actions)

    def update(self, state, action, reward, next_state, next_possible_actions):
        """
        Cập nhật Q-value theo công thức Q-learning.
        """
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        current_q = self.q_table.get((state_key, action), 0)

        if next_possible_actions:
            next_q = max(self.q_table.get((next_state_key, a), 0) for a in next_possible_actions)
        else:
            next_q = 0

        new_q = current_q + self.alpha * (reward + self.gamma * next_q - current_q)
        self.q_table[(state_key, action)] = new_q

# ============================
# LƯU VÀ TẢI MODEL (Q-TABLE)
# ============================
def save_q_table(agent, filename="q_table.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(agent.q_table, f)
    print(f"Q-table saved to {filename}")

def load_q_table(agent, filename="q_table.pkl"):
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            agent.q_table = pickle.load(f)
        print(f"Q-table loaded from {filename}")
    else:
        print(f"No Q-table file found at {filename}. Training from scratch.")

# ============================
# HÀM HUẤN LUYỆN (với agent được truyền vào)
# ============================
def train_q_learning(agent, num_episodes=500, max_steps=50):
    env = QLearningCuttingEnv()
    rewards_history = []
    best_reward = -float('inf')
    best_env = None

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0

        for step in range(max_steps):
            possible_actions = env.get_possible_actions()
            if not possible_actions:
                break

            action = agent.choose_action(state, possible_actions)
            next_state, r = env.step(action)
            next_possible_actions = env.get_possible_actions()
            agent.update(state, action, r, next_state, next_possible_actions)

            total_reward += r
            state = next_state

            if env.is_done():
                break

        total_reward += env.leftover_penalty()
        total_reward += env.complete_bonus()

        rewards_history.append(total_reward)
        if total_reward > best_reward:
            best_reward = total_reward
            best_env = copy.deepcopy(env)

        print(f"Episode {episode+1}: Total Reward = {total_reward}")

    return best_env, rewards_history

# ============================
# HÀM TÍNH TOÁN CÁC DIỆN TÍCH
# ============================
def compute_areas(env):
    board_width = env.board["width"]
    board_length = env.board["length"]
    total_area = board_width * board_length
    used_area = sum(p["size"][0] * p["size"][1] for p in env.placements)
    waste_area = total_area - used_area
    return total_area, used_area, waste_area

# ============================
# HÀM IN BẢNG TÓM TẮT
# ============================
def print_best_solution_summary(env):
    print("\nSummary of Best Fit Cutting:")
    print("+----------+----------+--------------+----------+")
    print("| Stock ID | Piece ID | Dimensions   | Position |")
    print("+----------+----------+--------------+----------+")
    for placement in env.placements:
        stock_id = env.board["id"]
        piece_id = placement["piece_id"]
        w, h = placement["size"]
        row, col = placement["pos"]
        dim_str = f"{w}x{h}"
        pos_str = f"({row},{col})"
        print(f"| {stock_id:<8} | {piece_id:<8} | {dim_str:<12} | {pos_str:<8} |")
    print("+----------+----------+--------------+----------+")

# ============================
# HÀM VẼ BIỂU ĐỒ REWARD
# ============================
def plot_training_rewards(rewards):
    episodes = range(1, len(rewards)+1)
    plt.figure(figsize=(8,4))
    plt.plot(episodes, rewards, marker='o', color='b')
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Q-Learning Rewards (Tables, Legs, Chairs)")
    plt.grid(True)
    plt.show()

# ============================
# HÀM TRỰC QUAN HÓA KẾT QUẢ
# ============================
def visualize_best_solution(env):
    board = env.board
    placements = env.placements
    fig, ax = plt.subplots(figsize=(5,10))
    ax.set_title("Best Cutting Layout")
    ax.set_xlim(0, board["width"])
    ax.set_ylim(0, board["length"])
    ax.set_xticks(range(board["width"]+1))
    ax.set_yticks(range(board["length"]+1))
    ax.grid(True, linewidth=0.5)
    ax.imshow(np.full((board["length"], board["width"]), 1), cmap='gray', vmin=0, vmax=1)

    colors = ["#FF5733", "#33FF57", "#3357FF", "#F4C724", "#A833FF", "#33FFF5", "#FF33A8"]
    for p in placements:
        color = colors[(p["piece_id"] - 1) % len(colors)]
        pos = p["pos"]
        size = p["size"]
        rect = patches.Rectangle((pos[1], pos[0]), size[0], size[1],
                                 linewidth=1.5, edgecolor="black",
                                 facecolor=color, alpha=0.8)
        ax.add_patch(rect)

    plt.show()

# ============================
# HÀM GIẢI TRIỆN (SINGLE EPISODE TEST)
# ============================
def simulate_episode(agent, max_steps=50):
    env = QLearningCuttingEnv()
    state = env.reset()
    total_reward = 0
    while not env.is_done():
        possible_actions = env.get_possible_actions()
        if not possible_actions:
            break
        action = agent.choose_action(state, possible_actions)
        next_state, r = env.step(action)
        total_reward += r
        state = next_state
    total_reward += env.leftover_penalty()
    total_reward += env.complete_bonus()
    return env, total_reward

# ============================
# MAIN
# ============================
def main():
    agent = QLearningAgent(alpha=0.1, gamma=0.9, epsilon=0.2)
    model_file = "q_table.pkl"

    # Nếu file model tồn tại, tải Q-table
    if os.path.exists(model_file):
        load_q_table(agent, model_file)
        print("Loaded Q-table. Running test episode...")
        best_env, test_reward = simulate_episode(agent, max_steps=50)
        rewards_history = [test_reward]
    else:
        print("No saved Q-table found. Training agent...")
        best_env, rewards_history = train_q_learning(agent=agent, num_episodes=500, max_steps=50)
        save_q_table(agent, model_file)

    plot_training_rewards(rewards_history)

    if best_env is not None:
        total_area, used_area, waste_area = compute_areas(best_env)
        print(f"\nTotal Area = {total_area}")
        print(f"Used Area  = {used_area}")
        print(f"Waste Area = {waste_area}")
        print_best_solution_summary(best_env)
        visualize_best_solution(best_env)

if __name__ == "__main__":
    main()
