import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

def train_agent(env, agent, episodes=1000, max_steps=200, update_target_every=10, 
                save_path='models', save_every=100, render_every=100):
    """
    Train the DQN agent on the wood cutting environment.
    
    Args:
        env: The wood cutting environment
        agent: The DQN agent
        episodes: Number of episodes to train
        max_steps: Maximum steps per episode
        update_target_every: Update target network every N episodes
        save_path: Directory to save models
        save_every: Save model every N episodes
        render_every: Render environment every N episodes
    """
    os.makedirs(save_path, exist_ok=True)
    training_history_file_path = f'{save_path}/training_history_data.npz'
    
    # Remove the old history file
    if os.path.exists(training_history_file_path):
        os.remove(training_history_file_path)

    rewards_history = []
    waste_rate_history = []
    fitness_history = []
    platforms_used_history = []
    
    for episode in tqdm(range(episodes), desc="Training"):
        state = env.reset()
        total_reward = 0
        
        for step in range(max_steps):
            # Choose action
            action = agent.act(state)
            
            # Take action
            next_state, reward, done, info = env.step(action)
            
            # Remember experience
            agent.remember(state, action, reward, next_state, done)
            
            # Update state and total reward
            state = next_state
            total_reward += reward
            
            # Train agent
            loss = agent.replay()
            
            if done:
                # Record metrics if episode is complete
                rewards_history.append(total_reward)
                
                if 'waste_rate' in info:
                    waste_rate_history.append(info['waste_rate'])
                    fitness_history.append(info['fitness'])
                    platforms_used_history.append(info['platforms_used'])
                np.savez(training_history_file_path, 
                        waste_rate_history=waste_rate_history, 
                        fitness_history=fitness_history, 
                        platforms_used_history=platforms_used_history)
                # Render occasionally
                if episode % render_every == 0:
                    print(f"\nEpisode {episode}")
                    print(f"Reward: {total_reward:.2f}")
                    if 'waste_rate' in info:
                        print(f"Waste_rate: {info['waste_rate']}")
                        print(f"Fitness: {info['fitness']:.2f}")
                        print(f"Platforms used: {info['platforms_used']}")
                    env.render()         
                break
        
        # Update target network periodically
        if episode % update_target_every == 0:
            agent.update_target_network()
        
        # Save model periodically
        if episode % save_every == 0:
            agent.save_model(f"{save_path}/dqn_wood_cutting_ep{episode}.pth")

    # Save final model
    agent.save_model(f"{save_path}/dqn_wood_cutting_final.pth")
    
    # Plot training progress
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(rewards_history)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    if waste_rate_history:
        plt.subplot(2, 2, 2)
        plt.plot(waste_rate_history)
        plt.title('Waste Rate Across Episodes')
        plt.xlabel('Completed Episode')
        plt.ylabel('Waste Rate')
        
        plt.subplot(2, 2, 3)
        plt.plot(fitness_history)
        plt.title('Fitness Across Episodes')
        plt.xlabel('Completed Episode')
        plt.ylabel('Efficiency')
        
        plt.subplot(2, 2, 4)
        plt.plot(platforms_used_history)
        plt.title('Platforms Used')
        plt.xlabel('Completed Episode')
        plt.ylabel('Number of Platforms')
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/training_progress.png")
    plt.show()
    
    return rewards_history, waste_rate_history, fitness_history, platforms_used_history

def evaluate_agent(env, agent, max_step=200, episodes=10, render=True):
    """
    Evaluate the trained agent on new orders.
    
    Args:
        env: The wood cutting environment
        agent: The trained DQN agent
        episodes: Number of evaluation episodes
        render: Whether to render the environment
    """
    total_rewards = []
    waste_rates = []
    fitness = []
    platforms_used = []
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        step = 0
        
        while not done and step < max_step:
            action = agent.act(state, training=False)
            print(action)
            next_state, reward, done, info = env.step(action)
            state = next_state
            total_reward += reward
            step += 1
        
        total_rewards.append(total_reward)
        
        if 'waste_rate' in info:
            waste_rates.append(info['waste_rate'])
            fitness.append(info['fitness'])
            platforms_used.append(info['platforms_used'])
        
        if render:
            print(f"\nEvaluation Episode {episode + 1}")
            print(f"Reward: {total_reward:.2f}")
            if 'waste_rate' in info:
                print(f"Waste_rate: {info['waste_rate']}")
                print(f"Fitness: {info['fitness']:.2f}")
                print(f"Platforms used: {info['platforms_used']}")
            env.render()
    
    print("\nEvaluation Results:")
    print(f"Average Reward: {np.mean(total_rewards):.2f}")
    if waste_rates:
        print(f"Average Waste_rate: {np.mean(waste_rates):.2f}")
        print(f"Average Fitness: {np.mean(fitness):.2f}")
        print(f"Average Platforms Used: {np.mean(platforms_used):.2f}")
    
    return total_rewards, waste_rates, fitness, platforms_used