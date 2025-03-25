# Cutting Wood Problem in Timber Industry (2D Cutting Stock Problem)

This repository addresses the Cutting Wood Problem within the Timber Industry, focusing on optimizing wood panel cutting to minimize waste. It explores solutions using both Reinforcement Learning (PPO,DQN Q-Learning) and Heuristic algorithms, aiming to improve efficiency and reduce material costs.

## 1. Introduction and Motivation:
### 1.1 The Cutting-Stock Problem
The Cutting-Stock Problem, an NP-hard optimization challenge, focuses on efficiently cutting stock sheets into smaller items to minimize waste or stock usage while satisfying demands. Traditionally, solutions involve Integer Linear Programming (ILP), heuristics, or column generation. This project investigates a 2D variation of the problem, specifically dealing with non-rotatable rectangular items of integer dimensions, as it applies to the timber industry. We explore the potential of Reinforcement Learning (RL), a method that trains agents through trial-and-error to maximize cumulative rewards, to develop dynamic and efficient cutting patterns for this problem. RL's adaptability to complex environments makes it a promising approach for optimizing material utilization in wood cutting.

### 1.2 Motivation
This project aims to address the wood cutting problem by:

1. Developing and evaluating heuristic algorithms to provide baseline performance and practical solutions for the timber industry.
2. Exploring and implementing Reinforcement Learning (RL) techniques (PPO, Q-Learning) to create adaptive cutting strategies that surpass traditional methods in efficiency and waste reduction.
3. Conducting a comparative analysis of heuristic and RL approaches to determine the optimal solution based on specific industry requirements, such as minimizing material waste, maximizing throughput, or handling complex order specifications.
---

## 2. Slide

```
root
├── _slide
│   └── REL301m_AI17C_Group2_Project.pptx
```

## 3. Report 

```
root
├── report
|   └──Assignment-SE173577_QE170039_QE170144_QE170087_QE170011.pdf
```

## 4. Diary 

```
root
├── diary.md
```

## 5. Effort 

```
root
├── effort.md
```
## Problem Description: Wood Panel Optimization

In the timber and woodworking industries, large wood panels (stock sheets) need to be cut into smaller rectangular pieces to fulfill customer orders and manufacturing requirements. The core challenge revolves around:

- **Minimizing wood waste (trim loss):** Maximizing the usable area of each panel is critical for cost-effectiveness and sustainability.
- **Reducing the number of panels used:** Efficient layouts directly impact material costs and storage needs.
- **Ensuring practical and non-overlapping cuts:** Cutting patterns must be feasible and avoid collisions.

This project models the wood panel cutting process as a **2D Cutting Stock Problem (2D-CSP)** with specific constraints relevant to the timber industry:

- **Meeting order demands:** All required piece sizes and quantities must be produced.
- **Optimizing material usage within panel boundaries:** No cuts can extend beyond the panel's dimensions.
- **Preventing overlapping cuts:** Pieces cannot overlap during the cutting process.
- **Cuts originating from panel edges:** This constraint reflects common cutting practices in woodworking.
- **Fixed piece orientations:** Pieces cannot be rotated, mirroring real-world limitations in many cutting scenarios.
---

## 6. Approaches

### 6.1 Heuristic Algorithms for Wood Panel Cutting

Our team implemented and compared three classic heuristic algorithms specifically adapted for optimizing the cutting of large wood panels (stock sheets) into smaller rectangular pieces to fulfill customer orders and manufacturing requirements:

| Algorithm     | Description                                                                                                   | Pros                                                             | Cons                                                                       |
| ------------- | ------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- | -------------------------------------------------------------------------- |
| **First-Fit** | Places each required piece in the first available stock sheet where it fits.                                  | Fast, easy to implement; suitable for real-time decision-making  | May leave large unused gaps, leading to inefficiency in material usage     |
| **Best-Fit**  | Places each required piece in the stock sheet where it leaves the least remaining space.                      | Maximizes material utilization, reducing waste                   | Slower than First-Fit due to additional computations for optimal placement |
| **Greedy**    | Combines First-Fit and Best-Fit strategies with an optimization step for merging partially used stock sheets. | Balances performance and efficiency, reduces waste significantly | Slightly more complex to implement and manage                              |
|
---

### 6.2 Reinforcement Learning for Wood Panel Cutting
We implemented a **Proximal Policy Optimization (PPO)**, **Deep-Q-Network** and **Q-Learning** agent using a custom Gym environment to learn cutting strategies over time.

### 6.2.1 PPO Approach
### 6.2.1.1 PPO Architecture

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


### 6.2.1.2 PPO Reward Function

-   Encourages high **cutting efficiency**.
-   Penalizes **wood waste and excessive panel use**.
-   Rewards **successful piece placement**.

### 6.2.1.3 PPO Limitations Observed

-   **Slow learning** compared to heuristics.
-   **Sensitivity** to hyperparameter tuning.
-   **Poor performance** on simple cutting tasks.

### 6.2.2 DQN Approach

#### **6.2.2.1 DQN Architecture**

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

#### **6.2.2.2 DQN Reward Function**

- Encourages high **cutting efficiency**.
- Penalizes **wood waste and excessive panel use**.
- Rewards **successful piece placement**.

#### **6.2.2.3 DQN Limitations Observed**

- **Slow learning** compared to heuristics.
- **Sensitivity** to hyperparameters.
- **Poor performance** on simple cutting tasks.


## 7. Evaluation Metrics

| Metric               | Description                                               |
|----------------------|-----------------------------------------------------------|
| **Runtime (s)**       | Average Runtime for on a list of customer orders                        |
| **Waste Rate**   | Sum of unused areas in used sheets                       |
| **Fitness**       | Number of sheets used to fulfill demand                  |

---
---
## 7. Results
The application of Q-learning , Proximal Policy Optimization (PPO) , and Deep Q-Networks (DQN) in the 2D Cutting Stock Problem (CSP) demonstrates the potential of Reinforcement Learning (RL) in solving complex combinatorial optimization problems. While traditional heuristic methods such as First-Fit Decreasing (FFD) , Best-Fit , and Greedy algorithms provide fast and relatively efficient solutions, RL approaches like Q-learning, PPO, and DQN offer adaptability and the ability to learn from experience.

Q-learning focuses on learning the optimal action-value function by iteratively updating Q-values based on rewards, enabling the agent to make decisions that balance immediate rewards with long-term goals. 

Similarly, DQN extends Q-learning by using a neural network to approximate the Q-function, allowing it to handle high-dimensional state spaces while employing techniques like experience replay and target networks to stabilize training. 

On the other hand, PPO leverages policy gradient methods to directly optimize the policy, ensuring stable updates through mechanisms like clipping and incorporating entropy regularization to encourage exploration. Together, these RL approaches provide flexible frameworks for tackling the complexities of the 2D CSP, offering the potential to outperform traditional heuristics in scenarios requiring dynamic decision-making and resource optimization.


## 8. Folder Structure:
```
├── _report                 
│   ├── Assignment-SE173577_QE170039_QE170144_QE170087_QE170011.pdf 
├── _slide                   
│   ├── REL301m_AI17C_Group2_Project.pptx              
├── VietLQ_SE173577_submit_code   # PPO and Deep-Q-Learning Approach           
├── ChuongLV_QE170039_submit_code   # Q-Learning Approach          
├── HoangTH_QE170011_submit_code   # Heuristic Approach       
├── PhucNN_QE170087_submit_code   # Heuristic Approach          
├── VietNLQ_QE170144_submit_code   # Q-Learning Approach
```

## This Repo's Authors

**FPT University – Quy Nhon AI Campus**  
Faculty of Information Technology – Mini Capstone Project – Mar 2025

- Nguyễn Lê Quốc Việt – QE170144(Leader)
- Lê Quốc Việt – SE173577 
- Lê Văn Chương – QE170039  
- Trần Hữu Hoàng – QE170011 
- Nguyễn Ngọc Phúc – QE170087 

**Instructor:** Dr. Nguyen An Khuong

---

## License

MIT License – See `LICENSE` for more details.
