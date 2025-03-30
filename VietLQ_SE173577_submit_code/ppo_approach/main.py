import numpy as np
import matplotlib.pyplot as plt
import argparse
from environment import WoodCuttingEnv
from agent import PPOAgent
from training import train_ppo_agent, evaluate_ppo_agent

def main():
    parser = argparse.ArgumentParser(description='Wood Cutting Optimization with PPO')
    parser.add_argument('--train', action='store_true', help='Train the agent')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate the agent')
    parser.add_argument('--model_path', type=str, default='models/ppo_wood_cutting_final.pth', 
                        help='Path to the model for evaluation')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--eval_episodes', type=int, default=5, help='Number of evaluation episodes')
    parser.add_argument('--platform_width', type=int, default=100, help='Width of big wood platform')
    parser.add_argument('--platform_height', type=int, default=100, help='Height of big wood platform')
    parser.add_argument('--max_platforms', type=int, default=5, help='Maximum number of platforms allowed')
    parser.add_argument('--render_every', type=int, default=100, help='Render every N episodes during training')
    parser.add_argument('--update_every', type=int, default=20, help='Update policy every N timesteps')
    parser.add_argument('--custom_order', action='store_true', help='Run with a custom order example')
    
    args = parser.parse_args()
    
    if args.custom_order:
        custom_order_example()
        return
    
    # Create environment
    env = WoodCuttingEnv(
        big_platform_size=(args.platform_width, args.platform_height),
        max_platforms=args.max_platforms
    )
    
    # Create PPO agent
    agent = PPOAgent(
        state_shape={
            'grid': env.observation_space['grid'].shape,
            'order': env.observation_space['order'].shape,
            'platform_index': (1,)
        },
        action_space=env.action_space.nvec
    )
    
    if args.train:
        # Train the agent
        print("Starting training with PPO...")
        train_ppo_agent(
            env=env,
            agent=agent,
            episodes=args.episodes,
            update_every=args.update_every,
            render_every=args.render_every
        )
        print("Training completed!")
    
    if args.evaluate:
        # Load the model if not training
        if not args.train:
            print(f"Loading model from {args.model_path}...")
            agent.load_model(args.model_path)
        
        # Evaluate the agent
        print("Starting evaluation...")
        evaluate_ppo_agent(
            env=env,
            agent=agent,
            episodes=args.eval_episodes
        )
        print("Evaluation completed!")

def custom_order_example():
    """Example of using a custom order with PPO agent."""
    # Create environment
    env = WoodCuttingEnv()
    
    # Create and load agent
    agent = PPOAgent(
        state_shape={
            'grid': env.observation_space['grid'].shape,
            'order': env.observation_space['order'].shape,
            'platform_index': (1,)
        },
        action_space=env.action_space.nvec
    )
    
    try:
        agent.load_model('models/ppo_wood_cutting_final.pth')
        print("Loaded trained model")
    except:
        print("No trained model found, using untrained agent")
    
    # Define a custom order
    # Format: [[width, height, quantity], ...]
    custom_order = np.array([
        [30, 20, 5],   # 5 pieces of 30x20
        [25, 15, 8],   # 8 pieces of 25x15
        [40, 10, 3],   # 3 pieces of 40x10
        [0, 0, 0],     # Padding
        [0, 0, 0],     # Padding
        [0, 0, 0],     # Padding
        [0, 0, 0],     # Padding
        [0, 0, 0],     # Padding
        [0, 0, 0],     # Padding
        [0, 0, 0]      # Padding
    ])
    
    # Reset environment with custom order
    state = env.reset(order=custom_order)
    
    # Run episode with custom order
    done = False
    total_reward = 0
    steps = 0
    max_steps = 200
    
    print("Starting custom order optimization with PPO...")
    
    while not done and steps < max_steps:
        # Choose action
        action, _, _ = agent.choose_action(state, training=False)
        
        # Take action
        next_state, reward, done, info = env.step(action)
        
        # Update state and total reward
        state = next_state
        total_reward += reward
        steps += 1
        
        if done:
            print(f"Order completed in {steps} steps")
            print(f"Total reward: {total_reward:.2f}")
            if 'waste' in info:
                print(f"Waste: {info['waste']}")
                print(f"Efficiency: {info['efficiency']:.2f}")
                print(f"Platforms used: {info['platforms_used']}")
    
    # Render final state
    env.render()
    
    if not done:
        print(f"Failed to complete order within {max_steps} steps")
    
    return total_reward

if __name__ == "__main__":
    main()