import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import random

class ActorCriticNetwork(nn.Module):
    def __init__(self, input_shape, action_dims):
        super(ActorCriticNetwork, self).__init__()
        
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
        
        # Actor head (policy)
        self.actor_x = nn.Linear(512, action_dims[0])  # x position
        self.actor_y = nn.Linear(512, action_dims[1])  # y position
        self.actor_piece = nn.Linear(512, action_dims[2])  # piece type
        self.actor_rotation = nn.Linear(512, action_dims[3])  # rotation
        
        # Critic head (value function)
        self.critic = nn.Linear(512, 1)
        
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
        
        # Actor outputs (policy distributions)
        x_logits = self.actor_x(features)
        y_logits = self.actor_y(features)
        piece_logits = self.actor_piece(features)
        rotation_logits = self.actor_rotation(features)
        
        # Critic output (value function)
        value = self.critic(features)
        
        return x_logits, y_logits, piece_logits, rotation_logits, value

class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []
        self.batch_size = batch_size
        
    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)
    
    def clear_memory(self):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []
    
    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]
        
        return batches

class PPOAgent:
    def __init__(self, state_shape, action_space, device="cuda" if torch.cuda.is_available() else "cpu",
                 lr=0.0003, gamma=0.99, gae_lambda=0.95, policy_clip=0.2, 
                 batch_size=64, n_epochs=10, entropy_coefficient=0.01):
        self.state_shape = state_shape
        self.action_space = action_space
        self.device = device
        
        # PPO hyperparameters
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.entropy_coefficient = entropy_coefficient
        
        # Create actor-critic network
        self.actor_critic = ActorCriticNetwork(
            input_shape=(state_shape['grid'][0], state_shape['grid'][1]),
            action_dims=(action_space[0], action_space[1], action_space[2], action_space[3])
        ).to(device)
        
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        self.memory = PPOMemory(batch_size)
        
        # For handling actions
        self.grid_size = state_shape['grid'][0]
        self.max_order_types = state_shape['order'][0]
        self.rotations = action_space[3]
        
    def choose_action(self, state, training=True):
        # Convert state to tensors
        grid = torch.FloatTensor(state['grid']).unsqueeze(0).to(self.device)
        order = torch.FloatTensor(state['order']).unsqueeze(0).to(self.device)
        platform_idx = torch.LongTensor([state['platform_index']]).to(self.device)
        
        # Get action distributions and value from actor-critic
        with torch.no_grad():
            x_logits, y_logits, piece_logits, rotation_logits, value = self.actor_critic(
                grid, order, platform_idx
            )
        
        # Create distributions for each action component
        x_dist = Categorical(F.softmax(x_logits, dim=1))
        y_dist = Categorical(F.softmax(y_logits, dim=1))
        
        # Mask invalid piece types (pieces with quantity 0)
        piece_mask = torch.ones_like(piece_logits) * float('-inf')
        for i, (_, _, qty) in enumerate(state['order']):
            if qty > 0:
                piece_mask[0, i] = 0  # Unmask valid piece types
        
        masked_piece_logits = piece_logits + piece_mask
        piece_dist = Categorical(F.softmax(masked_piece_logits, dim=1))
        rotation_dist = Categorical(F.softmax(rotation_logits, dim=1))
        
        # Sample actions from distributions (or take most likely action during evaluation)
        if training:
            x = x_dist.sample()
            y = y_dist.sample()
            piece_type = piece_dist.sample()
            rotation = rotation_dist.sample()
        else:
            x = torch.argmax(x_dist.probs)
            y = torch.argmax(y_dist.probs)
            piece_type = torch.argmax(piece_dist.probs)
            rotation = torch.argmax(rotation_dist.probs)
        
        # Calculate log probabilities
        x_prob = x_dist.log_prob(x)
        y_prob = y_dist.log_prob(y)
        piece_prob = piece_dist.log_prob(piece_type)
        rotation_prob = rotation_dist.log_prob(rotation)
        
        # Sum the log probs to get the total action log probability
        action_log_prob = x_prob + y_prob + piece_prob + rotation_prob
        
        return [x.item(), y.item(), piece_type.item(), rotation.item()], action_log_prob.item(), value.item()
    
    def store_transition(self, state, action, action_log_prob, value, reward, done):
        self.memory.store_memory(state, action, action_log_prob, value, reward, done)
    
    def learn(self):
        for _ in range(self.n_epochs):
            # Calculate advantages and returns
            state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, done_arr = self._process_memory()
            
            # Generate mini-batches
            batches = self.memory.generate_batches()
            
            # Train on each batch
            for batch in batches:
                # Select batch components
                grids = torch.FloatTensor(state_arr['grid'][batch]).to(self.device)
                orders = torch.FloatTensor(state_arr['order'][batch]).to(self.device)
                platform_idxs = torch.LongTensor(state_arr['platform_index'][batch]).to(self.device)
                
                actions = action_arr[batch]
                old_probs = old_prob_arr[batch]
                values = vals_arr[batch]
                
                # Calculate advantages and returns for the batch
                advantages = self._calculate_advantages(
                    reward_arr[batch], values, done_arr[batch]
                )
                returns = advantages + values
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                # Forward pass
                x_logits, y_logits, piece_logits, rotation_logits, critic_value = self.actor_critic(
                    grids, orders, platform_idxs
                )
                
                # Extract action components
                x = actions[:, 0]
                y = actions[:, 1]
                piece_type = actions[:, 2]
                rotation = actions[:, 3]
                
                # Create distributions
                x_dist = Categorical(F.softmax(x_logits, dim=1))
                y_dist = Categorical(F.softmax(y_logits, dim=1))
                piece_dist = Categorical(F.softmax(piece_logits, dim=1))
                rotation_dist = Categorical(F.softmax(rotation_logits, dim=1))
                
                # Calculate new log probabilities
                x_new_probs = x_dist.log_prob(torch.LongTensor(x).to(self.device))
                y_new_probs = y_dist.log_prob(torch.LongTensor(y).to(self.device))
                piece_new_probs = piece_dist.log_prob(torch.LongTensor(piece_type).to(self.device))
                rotation_new_probs = rotation_dist.log_prob(torch.LongTensor(rotation).to(self.device))
                
                # Combine log probabilities
                new_probs = x_new_probs + y_new_probs + piece_new_probs + rotation_new_probs
                
                # Calculate probability ratio
                prob_ratio = torch.exp(new_probs - torch.FloatTensor(old_probs).to(self.device))
                
                # Calculate surrogate losses
                weighted_probs = advantages.to(self.device) * prob_ratio
                weighted_clipped_probs = advantages.to(self.device) * torch.clamp(
                    prob_ratio, 1-self.policy_clip, 1+self.policy_clip
                )
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()
                
                # Add entropy bonus for exploration
                entropy = x_dist.entropy().mean() + y_dist.entropy().mean() + \
                        piece_dist.entropy().mean() + rotation_dist.entropy().mean()
                
                # Calculate critic loss
                returns = returns.float().to(self.device)  # Ensure returns is float32
                critic_loss = F.mse_loss(critic_value.squeeze(), returns)
                
                # Calculate total loss
                total_loss = actor_loss + 0.5 * critic_loss - self.entropy_coefficient * entropy
                
                # Update network
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
        
        # Clear memory after learning
        self.memory.clear_memory()

    def _calculate_advantages(self, rewards, values, dones):
        """Calculate advantages using Generalized Advantage Estimation (GAE)."""
        advantages = np.zeros_like(rewards)
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
        
        return torch.tensor(advantages, dtype=torch.float32)  # Explicitly use float32
    
    def _process_memory(self):
        """Process memory data into arrays."""
        # States need special handling due to being dictionaries
        state_arr = {
            'grid': [],
            'order': [],
            'platform_index': []
        }
        
        for state in self.memory.states:
            state_arr['grid'].append(state['grid'])
            state_arr['order'].append(state['order'])
            state_arr['platform_index'].append(state['platform_index'])
        
        # Convert states to numpy arrays
        state_arr = {k: np.array(v) for k, v in state_arr.items()}
        
        # Convert other memory components to numpy arrays
        action_arr = np.array(self.memory.actions)
        old_prob_arr = np.array(self.memory.probs)
        vals_arr = np.array(self.memory.vals)
        reward_arr = np.array(self.memory.rewards)
        done_arr = np.array(self.memory.dones)
        
        return state_arr, action_arr, old_prob_arr, vals_arr, reward_arr, done_arr
    
    def _calculate_advantages(self, rewards, values, dones):
        """Calculate advantages using Generalized Advantage Estimation (GAE)."""
        advantages = np.zeros_like(rewards)
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
        
        return torch.FloatTensor(advantages)
    
    def save_model(self, path):
        torch.save({
            'actor_critic': self.actor_critic.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)
    
    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint['actor_critic'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])