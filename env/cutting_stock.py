import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


class CuttingStockEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 25}

    def __init__(
        self,
        render_mode=None,
        stock_list=None,
        product_list=None,
        seed=42,
    ):
        self.seed = seed
        set_seed(seed)

        # Khởi tạo danh sách stocks và products từ dữ liệu đầu vào
        self.stock_list = stock_list if stock_list else []
        self.product_list = product_list if product_list else []
        self.num_stocks = len(self.stock_list)

        self.cutted_stocks = np.zeros(self.num_stocks, dtype=int)

        # Không gian quan sát
        self.observation_space = spaces.Dict(
            {
                "stocks": spaces.Tuple(
                    [spaces.MultiDiscrete([s[0], s[1]]) for s in self.stock_list]
                ),
                "products": spaces.Sequence(
                    spaces.Dict(
                        {
                            "size": spaces.MultiDiscrete([max(p[0] for p in self.product_list),
                                                           max(p[1] for p in self.product_list)]),
                            "quantity": spaces.Discrete(100),
                        }
                    )
                ),
            }
        )

        # Không gian hành động
        self.action_space = spaces.Dict(
            {
                "stock_idx": spaces.Discrete(self.num_stocks),
                "size": spaces.MultiDiscrete([max(p[0] for p in self.product_list),
                                               max(p[1] for p in self.product_list)]),
                "position": spaces.MultiDiscrete([max(s[0] for s in self.stock_list),
                                                   max(s[1] for s in self.stock_list)]),
            }
        )

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def _get_obs(self):
        return {"stocks": self.stock_list, "products": self.product_list}

    def _get_info(self):
        filled_ratio = np.mean(self.cutted_stocks).item()
        return {"filled_ratio": filled_ratio}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        set_seed(seed)
        self.cutted_stocks.fill(0)
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        stock_idx = action["stock_idx"]
        size = action["size"]
        position = action["position"]

        if 0 <= stock_idx < self.num_stocks:
            stock_w, stock_h = self.stock_list[stock_idx]
            if position[0] + size[0] <= stock_w and position[1] + size[1] <= stock_h:
                self.cutted_stocks[stock_idx] = 1

        terminated = all(self.cutted_stocks)
        reward = 1 if terminated else 0
        return self._get_obs(), reward, terminated, False, self._get_info()

    def close(self):
        pass
