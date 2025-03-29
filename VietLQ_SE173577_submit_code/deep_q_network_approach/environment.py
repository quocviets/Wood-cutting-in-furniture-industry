import numpy as np
import matplotlib.pyplot as plt
import gym
from gym import spaces
# from scipy import ndimage

class WoodCuttingEnv(gym.Env):
    def __init__(self, big_platform_size=(100, 100), max_platforms=5):
        super(WoodCuttingEnv, self).__init__()
        
        # Initialize with the size of big wood platforms
        self.big_platform_width, self.big_platform_height = big_platform_size
        self.max_platforms = max_platforms
        
        # State: Representation of the current platform's state
        # We'll represent it as a binary grid
        self.grid_size = 100  # We'll use a 100x100 grid for simplicity
        
        # Track piece types in each position (0 = empty, 1+ = piece type index + 1)
        self.platforms = [np.zeros((self.grid_size, self.grid_size), dtype=np.int8)]
        self.piece_types = [np.zeros((self.grid_size, self.grid_size), dtype=np.int8)]

        # State space: Current platform's grid state + remaining order pieces + current platform index
        grid_space = spaces.Box(low=0, high=1, shape=(self.grid_size, self.grid_size), dtype=np.int8)
        
        # Max number of different piece types in an order
        self.max_order_types = 10
        # Order space: (width, height, quantity) for each order type
        order_space = spaces.Box(
            low=np.array([[1, 1, 0]] * self.max_order_types),
            high=np.array([[self.big_platform_width, self.big_platform_height, 100]] * self.max_order_types),
            dtype=np.int32
        )
        
        # Current platform index
        platform_index_space = spaces.Discrete(self.max_platforms + 1)  # +1 for "no platforms left"
        
        # Combine spaces
        self.observation_space = spaces.Dict({
            'grid': grid_space,
            'order': order_space,
            'platform_index': platform_index_space
        })
        
        # Action space: (x, y, piece_type, rotation)
        # x, y: position to place the piece
        # piece_type: which piece type from the order to use
        # rotation: 0 or 1 (0° or 90°)
        self.action_space = spaces.MultiDiscrete([
            self.grid_size,  # x
            self.grid_size,  # y
            self.max_order_types,  # piece_type
            2  # rotation (0 or 1)
        ])
        
        # Initialize state
        self.reset()
    
    def reset(self, order=None):
        """Reset environment with a new or provided order."""
        # Initialize the first platform grid (0 = empty, 1 = filled)
        self.platforms = [np.zeros((self.grid_size, self.grid_size), dtype=np.int8)]
        self.piece_types = [np.zeros((self.grid_size, self.grid_size), dtype=np.int8)]
        self.current_platform_idx = 0
        
        # Generate a random order if none provided
        if order is None:
            self.order = self._generate_random_order()
        else:
            self.order = order.copy()
        
        return self._get_observation()
    
    def _generate_random_order(self):
        """Generate a random cutting order."""
        num_types = np.random.randint(1, self.max_order_types + 1)
        order = []
        
        for _ in range(num_types):
            width = np.random.randint(5, min(self.big_platform_width // 2, 30) + 1)
            height = np.random.randint(5, min(self.big_platform_height // 2, 30) + 1)
            quantity = np.random.randint(1, 20)
            order.append([width, height, quantity])
        
        # Pad the order to max_order_types
        while len(order) < self.max_order_types:
            order.append([0, 0, 0])
            
        return np.array(order)
    
    def _get_observation(self):
        """Return the current observation."""
        return {
            'grid': self.platforms[self.current_platform_idx],
            'order': self.order,
            'platform_index': self.current_platform_idx
        }
    
    def _is_valid_placement(self, x, y, piece_width, piece_height, platform_idx=None):
        """Check if a piece can be placed at (x, y) with given dimensions."""
        if platform_idx is None:
            platform_idx = self.current_platform_idx
            
        if x + piece_width > self.grid_size or y + piece_height > self.grid_size:
            return False
        
        # Check if the area is empty
        if np.any(self.platforms[platform_idx][y:y+piece_height, x:x+piece_width] == 1):
            return False
        
        return True
    
    def _can_fit_anywhere(self, piece_width, piece_height, platform_idx=None):
        """
        Check if a piece can fit anywhere on the specified platform.
        If platform_idx is None, check the current platform.
        Returns (can_fit, (platform_idx, x, y)) if fit is possible.
        """
        if platform_idx is None:
            platforms_to_check = [self.current_platform_idx]
        else:
            platforms_to_check = [platform_idx]
            
        for platform_idx in platforms_to_check:
            for y in range(self.grid_size - piece_height + 1):
                for x in range(self.grid_size - piece_width + 1):
                    if self._is_valid_placement(x, y, piece_width, piece_height, platform_idx):
                        return True, (platform_idx, x, y)
        return False, None
    
    def _place_piece(self, x, y, piece_width, piece_height, piece_type, platform_idx=None):
        """Place a piece at (x, y) with given dimensions."""
        if platform_idx is None:
            platform_idx = self.current_platform_idx
            
        self.platforms[platform_idx][y:y+piece_height, x:x+piece_width] = 1
        self.piece_types[platform_idx][y:y+piece_height, x:x+piece_width] = piece_type + 1  # +1 to avoid 0

    def _can_fit_on_any_platform(self, piece_width, piece_height):
        """
        Check if a piece can fit on any existing platform.
        Returns (can_fit, (platform_idx, x, y)) if fit is possible.
        """
        # Check all existing platforms
        for platform_idx in range(len(self.platforms)):
            can_fit, position = self._can_fit_anywhere(piece_width, piece_height, platform_idx)
            if can_fit:
                return True, position
        return False, None
    
    def calculate_metrics(self):
        """
        Calculate various efficiency metrics for the current cutting solution.
        
        Returns:
            dict: Dictionary containing the following metrics:
                - waste_rate: Total unused area / Total area of all cut stock
                - fitness: Total area of all cut items / Total area of all cut stock
                - platforms_used: Number of platforms used
                - filled_area: Total area filled with cut items
                - total_stock_area: Total area of all cut stock
        """
        # Count how many platforms were actually used
        used_platforms = 0
        filled_area = 0
        
        for platform in self.platforms:
            if np.any(platform == 1):
                used_platforms += 1
                filled_area += np.sum(platform)
        
        # Calculate total stock area (area of all used platforms)
        total_stock_area = self.big_platform_width * self.big_platform_height * used_platforms
        
        # Calculate waste rate
        waste_area = total_stock_area - filled_area
        waste_rate = waste_area / total_stock_area if total_stock_area > 0 else 0
        
        # Calculate fitness
        fitness = filled_area / total_stock_area if total_stock_area > 0 else 0
        
        return {
            'waste_rate': waste_rate,
            'fitness': fitness,
            'platforms_used': used_platforms,
            'filled_area': filled_area,
            'total_stock_area': total_stock_area,
            'waste_area': waste_area
        }
    
    def step(self, action):
        """
        Take a step in the environment.
        
        Args:
            action: [x, y, piece_type, rotation]
        
        Returns:
            observation, reward, done, info
        """
        x, y, piece_type, rotation = action
        
        # Check if piece_type is valid
        if piece_type >= len(self.order) or self.order[piece_type][2] <= 0:
            # Invalid piece type or no more pieces of this type
            return self._get_observation(), -10, False, {'message': 'Invalid piece type'}
        
        # Get piece dimensions
        width, height, quantity = self.order[piece_type]
        
        # Apply rotation if needed
        if rotation == 1:
            width, height = height, width
        
        # Check if placement is valid at the chosen position
        if not self._is_valid_placement(x, y, width, height):
            # First, check if the piece can fit anywhere else on the current platform
            can_fit_current, current_pos = self._can_fit_anywhere(width, height)
            
            if can_fit_current:
                # Place at the new position on current platform
                platform_idx, x, y = current_pos
                self._place_piece(x, y, width, height, piece_type, platform_idx)
                self.order[piece_type][2] -= 1
                reward = width * height - 5  # Small penalty for repositioning on same platform
                
            else:
                # Check if the piece can fit on any existing platform
                can_fit_existing, best_pos = self._can_fit_on_any_platform(width, height)
                
                if can_fit_existing:
                    # Place on an existing platform
                    platform_idx, x, y = best_pos
                    self._place_piece(x, y, width, height, piece_type, platform_idx)
                    self.current_platform_idx = platform_idx  # Update current platform
                    self.order[piece_type][2] -= 1
                    reward = width * height - 10  # Penalty for switching platforms
                    
                else:
                    # Try creating a new platform if none of the existing ones can fit the piece
                    if self.current_platform_idx + 1 < self.max_platforms:
                        # Create new platform if needed
                        if len(self.platforms) <= self.current_platform_idx + 1:
                            self.platforms.append(np.zeros((self.grid_size, self.grid_size), dtype=np.int8))
                            self.piece_types.append(np.zeros((self.grid_size, self.grid_size), dtype=np.int8))
                        
                        # Move to the new platform
                        self.current_platform_idx += 1
                        
                        # Try to place at (0,0) on new platform
                        if self._is_valid_placement(0, 0, width, height):
                            self._place_piece(0, 0, width, height, piece_type)
                            self.order[piece_type][2] -= 1
                            reward = width * height - 50  # Penalty for creating new platform
                        else:
                            reward = -20  # Penalty for invalid placement even on new platform
                    else:
                        reward = -30  # No more platforms available
                        return self._get_observation(), reward, True, {'message': 'No more platforms'}
        else:
            # Place the piece at the original position
            self._place_piece(x, y, width, height, piece_type)
            self.order[piece_type][2] -= 1
            reward = width * height  # Reward proportional to the piece area
        
        # Check if all pieces have been placed
        done = np.all(self.order[:, 2] == 0)
        if done:
            # Calculate metrics
            metrics = self.calculate_metrics()
            
            # Add final reward based on fitness
            reward += metrics['fitness'] * 1000
            
            return self._get_observation(), reward, done, {
                'message': 'All pieces placed',
                'waste_rate': metrics['waste_rate'],
                'fitness': metrics['fitness'],
                # 'waste': metrics['waste_area'],
                # 'efficiency': metrics['fitness'],  # Keep the old 'efficiency' name for backward compatibility
                'platforms_used': metrics['platforms_used'],
                'filled_area': metrics['filled_area'],
                'total_stock_area': metrics['total_stock_area']
            }

        return self._get_observation(), reward, done, {}
    
    def render(self):
        # """Render the current state of the environment."""
        """Render the current state of the environment with different colors for each piece type."""
        piece_colors = ['white', 'red', 'blue', 'green', 'purple', 'orange', 'yellow', 'black', 'gray', 'pink', 'brown']

        fig, axs = plt.subplots(1, len(self.platforms), figsize=(5*len(self.platforms), 5))
        if len(self.platforms) == 1:
            axs = [axs]
        
        for i, platform in enumerate(self.platforms):
            ax = axs[i]
            
            # Create a colored image
            colored_image = np.zeros((self.grid_size, self.grid_size, 3))
            
            # Fill with colors based on piece types
            for y in range(self.grid_size):
                for x in range(self.grid_size):
                    piece_type = self.piece_types[i][y, x]
                    if piece_type > 0:
                        # Convert color name to RGB
                        color_name = piece_colors[piece_type]
                        color_rgb = np.array(plt.matplotlib.colors.to_rgb(color_name))
                        colored_image[y, x] = color_rgb
            
            ax.imshow(colored_image)
            ax.set_title(f'Platform {i+1}')
            ax.set_xlim(0, self.grid_size)
            ax.set_ylim(0, self.grid_size)
            ax.invert_yaxis()  # Invert y-axis to match grid coordinates
            
            ax.grid(True, color='black', linewidth=0.5, alpha=0.3)
            
            # Add piece outlines
            for type_id in range(1, self.max_order_types + 1):
                piece_mask = (self.piece_types[i] == type_id)
                if not np.any(piece_mask):
                    continue
                    
        # Display remaining order with colors
        order_text = []
        for idx, (w, h, q) in enumerate(self.order):
            if q > 0:
                color = piece_colors[idx + 1]
                order_text.append(f"<span style='color:{color}'>{w}x{h} (qty: {q})</span>")
        
        if order_text:
            from matplotlib.text import Text
            plt.figtext(0.5, 0.01, f"Remaining pieces: {', '.join(order_text)}", 
                    ha="center", fontsize=9, bbox={"facecolor":"white", "alpha":0.8})
        else:
            plt.figtext(0.5, 0.01, "All pieces placed!", ha="center", fontsize=9, 
                    bbox={"facecolor":"green", "alpha":0.5})
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
        plt.show()
        
    def close(self):
        plt.close('all')