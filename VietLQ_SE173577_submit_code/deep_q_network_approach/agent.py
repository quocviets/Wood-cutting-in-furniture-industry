import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random

class DQNNetwork(nn.Module):
    def __init__(self, input_shape, output_size):
        super(DQNNetwork, self).__init__()
        
        # CNN for processing the grid
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        
        # Calculate the size after convolutions
        def conv2d_size_out(size, kernel_size=3, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1
        
        conv_width = conv2d_size_out(conv2d_size_out(conv2d_size_out(input_shape[1], 5, 2)))
        conv_height = conv2d_size_out(conv2d_size_out(conv2d_size_out(input_shape[0], 5, 2)))
        linear_input_size = conv_width * conv_height * 64
        
        # FC for processing the order information
        self.fc_order = nn.Linear(30, 128)  # 10 order types x 3 features (width, height, quantity)
        
        # FC for processing platform index
        self.fc_platform = nn.Linear(1, 32)
        
        # Combine and process
        self.fc_combine = nn.Linear(linear_input_size + 128 + 32, 512)
        self.fc_advantage = nn.Linear(512, output_size)
        self.fc_value = nn.Linear(512, 1)
        
    def forward(self, grid, order, platform_idx):
        # Process grid
        grid = grid.unsqueeze(1)  # Add channel dimension
        grid = F.relu(self.conv1(grid))
        grid = F.relu(self.conv2(grid))
        grid = F.relu(self.conv3(grid))
        grid = grid.view(grid.size(0), -1)  # Flatten
        
        # Process order
        order = order.view(order.size(0), -1)  # Flatten
        order = F.relu(self.fc_order(order))
        
        # Process platform index
        platform_idx = platform_idx.float().view(-1, 1)
        platform = F.relu(self.fc_platform(platform_idx))
        
        # Combine
        combined = torch.cat((grid, order, platform), dim=1)
        features = F.relu(self.fc_combine(combined))
        
        # Dueling DQN architecture
        advantage = self.fc_advantage(features)
        value = self.fc_value(features)
        
        # Combine value and advantage
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values

class DQNAgent:
    def __init__(self, state_shape, action_space, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.state_shape = state_shape
        self.action_space = action_space
        self.device = device
        
        # DQN hyperparameters
        self.gamma = 0.99  # Discount factor
        self.learning_rate = 0.0001
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 64
        self.memory = deque(maxlen=10000)
        
        # Create Q networks (main and target)
        self.q_network = DQNNetwork(
            input_shape=(state_shape['grid'][0], state_shape['grid'][1]),
            output_size=action_space[0] * action_space[1] * action_space[2] * action_space[3]
        ).to(device)
        
        self.target_network = DQNNetwork(
            input_shape=(state_shape['grid'][0], state_shape['grid'][1]),
            output_size=action_space[0] * action_space[1] * action_space[2] * action_space[3]
        ).to(device)
        
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        
        # For handling actions
        self.grid_size = state_shape['grid'][0]
        self.max_order_types = state_shape['order'][0]
        self.rotations = action_space[3]
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, training=True):
        # Random exploration during training
        if training and np.random.rand() <= self.epsilon:
            # Random action but only for valid piece types
            valid_piece_types = [i for i, (_, _, qty) in enumerate(state['order']) if qty > 0]
            if not valid_piece_types:
                # No valid pieces left
                return [0, 0, 0, 0]  # Return dummy action
                
            piece_type = np.random.choice(valid_piece_types)
            x = np.random.randint(0, self.action_space[0])
            y = np.random.randint(0, self.action_space[1])
            rotation = np.random.randint(0, self.action_space[3])
            return [x, y, piece_type, rotation]
        
        # Convert state to tensors
        grid = torch.FloatTensor(state['grid']).unsqueeze(0).to(self.device)
        order = torch.FloatTensor(state['order']).unsqueeze(0).to(self.device)
        platform_idx = torch.LongTensor([state['platform_index']]).to(self.device)
        
        # Get Q values
        with torch.no_grad():
            q_values = self.q_network(grid, order, platform_idx)
            q_values = q_values.view(
                self.grid_size, 
                self.grid_size, 
                self.max_order_types, 
                self.rotations
            )
        
        # Create a mask for invalid piece types
        mask = np.ones_like(q_values.cpu().numpy()) * float('-inf')
        for i, (_, _, qty) in enumerate(state['order']):
            if qty > 0:
                mask[:, :, i, :] = 0  # Set to 0 for valid piece types
        
        # Apply the mask to Q values
        q_values_np = q_values.cpu().numpy()
        masked_q_values = q_values_np + mask
        
        # Choose best action from masked Q values
        action_flat = np.argmax(masked_q_values.flatten())
        
        # Convert flat index to multi-dimensional action
        indices = np.unravel_index(action_flat, q_values_np.shape)
        return list(indices)
    
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        
        # Sample a batch from memory
        minibatch = random.sample(self.memory, self.batch_size)
        
        # Prepare batch data
        grids = torch.FloatTensor(np.array([s['grid'] for s, _, _, _, _ in minibatch])).to(self.device)
        orders = torch.FloatTensor(np.array([s['order'] for s, _, _, _, _ in minibatch])).to(self.device)
        platform_idxs = torch.LongTensor(np.array([s['platform_index'] for s, _, _, _, _ in minibatch])).to(self.device)
        
        next_grids = torch.FloatTensor(np.array([s['grid'] for _, _, _, s, _ in minibatch])).to(self.device)
        next_orders = torch.FloatTensor(np.array([s['order'] for _, _, _, s, _ in minibatch])).to(self.device)
        next_platform_idxs = torch.LongTensor(np.array([s['platform_index'] for _, _, _, s, _ in minibatch])).to(self.device)
        
        actions = np.array([a for _, a, _, _, _ in minibatch])
        rewards = torch.FloatTensor(np.array([r for _, _, r, _, _ in minibatch])).to(self.device)
        dones = torch.FloatTensor(np.array([d for _, _, _, _, d in minibatch])).to(self.device)
        
        # Compute current Q values
        current_q_values = self.q_network(grids, orders, platform_idxs)
        
        # Convert actions to flat indices for gathering
        action_indices = np.zeros(self.batch_size, dtype=np.int64)
        for i, action in enumerate(actions):
            x, y, piece_type, rotation = action
            action_indices[i] = (x * self.grid_size * self.max_order_types * self.rotations + 
                               y * self.max_order_types * self.rotations + 
                               piece_type * self.rotations + 
                               rotation)
        
        action_indices = torch.LongTensor(action_indices).to(self.device)
        current_q = current_q_values.gather(1, action_indices.unsqueeze(1)).squeeze(1)
        
        # Compute next Q values using Double DQN approach
        with torch.no_grad():
            # Get actions from main network
            next_q_main = self.q_network(next_grids, next_orders, next_platform_idxs)
            next_actions = next_q_main.max(1)[1]
            
            # Get Q values from target network for those actions
            next_q_target = self.target_network(next_grids, next_orders, next_platform_idxs)
            next_q = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Compute loss and update
        loss = F.smooth_l1_loss(current_q, target_q)  # Using Huber loss for stability
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping to prevent exploding gradients
        nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss.item()
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save_model(self, path):
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, path)
    
    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']