import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

def train_ppo_agent(env, agent, episodes=1000, max_timesteps=200, update_every=20,
                save_path='models', save_every=100, render_every=100):
    """
    Train the PPO agent on the wood cutting environment.
    
    Args:
        env: The wood cutting environment
        agent: The PPO agent
        episodes: Number of episodes to train
        max_timesteps: Maximum timesteps per episode
        update_every: Update policy every N timesteps
        save_path: Directory to save models
        save_every: Save model every N episodes
        render_every: Render environment every N episodes
    """
    os.makedirs(save_path, exist_ok=True)
    
    rewards_history = []
    waste_history = []
    efficiency_history = []
    platforms_used_history = []
    
    timestep_counter = 0
    
    for episode in tqdm(range(episodes), desc="Training"):
        state = env.reset()
        done = False
        total_reward = 0
        timestep = 0
        
        while not done and timestep < max_timesteps:
            # Select action
            action, action_log_prob, value = agent.choose_action(state)
            
            # Take action in environment
            next_state, reward, done, info = env.step(action)
            
            # Store transition
            agent.store_transition(state, action, action_log_prob, value, reward, done)
            
            # Update state and counters
            state = next_state
            total_reward += reward
            timestep += 1
            timestep_counter += 1
            
            # Update policy if enough steps have been taken
            if timestep_counter % update_every == 0:
                agent.learn()
        
        # Record episode stats
        rewards_history.append(total_reward)
        
        if 'waste' in info:
            waste_history.append(info['waste'])
            efficiency_history.append(info['efficiency'])
            platforms_used_history.append(info['platforms_used'])
        
        # Render occasionally
        if episode % render_every == 0:
            print(f"\nEpisode {episode}")
            print(f"Reward: {total_reward:.2f}")
            if 'waste' in info:
                print(f"Waste: {info['waste']}")
                print(f"Efficiency: {info['efficiency']:.2f}")
                print(f"Platforms used: {info['platforms_used']}")
            env.render()
        
        # Save model periodically
        if episode % save_every == 0:
            agent.save_model(f"{save_path}/ppo_wood_cutting_ep{episode}.pth")
    
    # Save final model
    agent.save_model(f"{save_path}/ppo_wood_cutting_final.pth")
    
    # Plot training progress
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(rewards_history)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    if waste_history:
        plt.subplot(2, 2, 2)
        plt.plot(waste_history)
        plt.title('Waste')
        plt.xlabel('Completed Episode')
        plt.ylabel('Waste Area')
        
        plt.subplot(2, 2, 3)
        plt.plot(efficiency_history)
        plt.title('Efficiency')
        plt.xlabel('Completed Episode')
        plt.ylabel('Efficiency')
        
        plt.subplot(2, 2, 4)
        plt.plot(platforms_used_history)
        plt.title('Platforms Used')
        plt.xlabel('Completed Episode')
        plt.ylabel('Number of Platforms')
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/ppo_training_progress.png")
    plt.show()
    
    return rewards_history, waste_history, efficiency_history, platforms_used_history

def evaluate_ppo_agent(env, agent, episodes=10, max_steps=200, render=True):
    """
    Evaluate the trained PPO agent on new orders.
    
    Args:
        env: The wood cutting environment
        agent: The trained PPO agent
        episodes: Number of evaluation episodes
        max_steps: Maximum steps per episode
        render: Whether to render the environment
    """
    total_rewards = []
    wastes = []
    efficiencies = []
    platforms_used = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        step = 0
        
        while not done and step < max_steps:
            # Choose action (deterministic during evaluation)
            action, _, _ = agent.choose_action(state, training=False)
            
            # Take action
            next_state, reward, done, info = env.step(action)
            state = next_state
            total_reward += reward
            step += 1
        
        total_rewards.append(total_reward)
        
        if 'waste' in info:
            wastes.append(info['waste'])
            efficiencies.append(info['efficiency'])
            platforms_used.append(info['platforms_used'])
        
        if render:
            print(f"\nEvaluation Episode {episode + 1}")
            print(f"Reward: {total_reward:.2f}")
            if 'waste' in info:
                print(f"Waste: {info['waste']}")
                print(f"Efficiency: {info['efficiency']:.2f}")
                print(f"Platforms used: {info['platforms_used']}")
            env.render()
    
    print("\nEvaluation Results:")
    print(f"Average Reward: {np.mean(total_rewards):.2f}")
    if wastes:
        print(f"Average Waste: {np.mean(wastes):.2f}")
        print(f"Average Efficiency: {np.mean(efficiencies):.2f}")
        print(f"Average Platforms Used: {np.mean(platforms_used):.2f}")
    
    return total_rewards, wastes, efficiencies, platforms_used