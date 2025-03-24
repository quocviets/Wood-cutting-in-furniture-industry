### Reinforcement Learning for Wood Panel Cutting
I implemented **Proximal Policy Optimization (PPO)** and **Deep-Q-Network** agent using a custom Gym environment to learn cutting strategies over time.

### PPO Approach
### PPO Architecture

-   **Actor-Critic Model:**
    -   CNN extracts spatial features from the cutting stock.
    -   MLP encodes product information.
-   **Actor Network:**
    -   Predicts product to cut and cutting position (x, y).
    -   Orientation is fixed.
-   **Critic Network:**
    -   Estimates state-value $V(s)$.
-   **Action Selection:**
    -   Invalid product selections are masked.
-   **Training:**
    -   Uses Proximal Policy Optimization (PPO).
    -   Generalized Advantage Estimation (GAE) for advantage.
Absolutely, let's condense those detailed descriptions into short, impactful summaries:


### PPO Reward Function

-   Encourages high **cutting efficiency**.
-   Penalizes **wood waste and excessive panel use**.
-   Rewards **successful piece placement**.

### PPO Limitations Observed

-   **Slow learning** compared to heuristics.
-   **Sensitivity** to hyperparameter tuning.
-   **Poor performance** on simple cutting tasks.

### DQN Approach

#### **DQN Architecture**

- **Dueling DQN Model:**
    - CNN extracts spatial features.
    - MLP encodes product info.
- **Dueling Network:**
    - Estimates state-value and action advantages.
- **Action Selection:**
    - Masks invalid actions.
    - Uses ϵ-greedy policy.
- **Training:**
    - Double DQN with Huber loss.

#### **DQN Reward Function**

- Encourages high **cutting efficiency**.
- Penalizes **wood waste and excessive panel use**.
- Rewards **successful piece placement**.

#### **DQN Limitations Observed**

- **Slow learning** compared to heuristics.
- **Sensitivity** to hyperparameters.
- **Poor performance** on simple cutting tasks.

###  Folder Structure:
```           
├── VietLQ_SE173577_submit_code    
│   ├── deep_q_network_approach                   
│   │   ├── README            # Provided detail about the concept of implementing the agent  
│   │   ├── agent.py          # Code for setup the deep-q-network agent             
│   │   ├── environment.py    # Code for setup the environment             
│   │   ├── main.py           # Main code for training the agent in cli   
│   │   ├── training.py       # Code for setup the training pipeline
│   │   ├── test_on_custom_order.ipynb     # Notebook for tesing and visualizing agent on a defined customer order
│   │   ├── training_with_visualize.py     # Notebook for training and visualizing the training progress                     
│   │   ├── result                    
│   │   │   ├── dqn_training_result.png    # Visualize the training result
│   ├── ppo_approach                  
│   │   ├── README            # Provided detail about the concept of implementing the agent              
│   │   ├── agent.py          # Code for setup the ppo agent             
│   │   ├── environment.py    # Code for setup the environment             
│   │   ├── main.py           # Main code for training the agent in cli   
│   │   ├── training.py       # Code for setup the training pipeline
│   │   ├── test_on_custom_order.ipynb     # Notebook for tesing and visualizing agent on a defined customer order
│   │   ├── training_with_visualize.py     # Notebook for training and visualizing the training progress                   
│   │   ├── result                    
│   │   │   ├── ppo_training_result.png    # Visualize the training result
```
## License

MIT License – See `LICENSE` for more details.