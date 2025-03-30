# 🪵 **Wood Cutting in Furniture Industry - Capstone Assignment (REL301M)**

## ✂️ **Group 2 - Cutting Stock Problem 2D**

### 📜 **Overview**
This repository contains the capstone project for the course **Reinforcement Learning (REL301M)**. Our project focuses on solving the **2D Cutting Stock Problem (CSP)** in the **furniture industry**, optimizing wood cutting to minimize waste and maximize efficiency using heuristic and reinforcement learning approaches.

### 🛠️ **Problem Statement**
The **2D Cutting Stock Problem** involves cutting rectangular demand pieces from larger wooden stock sheets while minimizing waste. This problem has significant applications in industries like furniture manufacturing, where efficient material utilization directly impacts costs and sustainability.

### 🔬 **Approach**
We implemented various algorithms to tackle this problem, including:
- ✅ **First Fit Algorithm**: Places each demand piece in the first available position on a stock sheet.
- 🌟 **Best Fit Algorithm**: Chooses the best possible position based on remaining space.
- 🤖 **Reinforcement Learning-based Approaches**: Uses RL techniques such as **PPO**, **DQN**, and **Q-Learning** to optimize cutting strategies over time.

---

## 📚 **Problem Description: Wood Panel Optimization**
In the timber and woodworking industries, large wood panels (stock sheets) need to be cut into smaller rectangular pieces to fulfill customer orders and manufacturing requirements. The core challenges revolve around:
- **Minimizing wood waste (trim loss):** Maximizing usable area for cost-effectiveness and sustainability.
- **Reducing the number of panels used:** Efficient layouts directly impact material costs and storage needs.
- **Ensuring practical and non-overlapping cuts:** Cutting patterns must be feasible and avoid collisions.

This project models the wood panel cutting process as a **2D Cutting Stock Problem (2D-CSP)** with specific constraints relevant to the timber industry:
- Meeting order demands: All required piece sizes and quantities must be produced.
- Optimizing material usage within panel boundaries: No cuts can extend beyond the panel's dimensions.
- Preventing overlapping cuts: Pieces cannot overlap during the cutting process.
- Cuts originating from panel edges: Reflects common cutting practices in woodworking.
- Fixed piece orientations: Pieces cannot be rotated, mirroring real-world limitations.

---

## 🛠️ **Approaches**

### **Heuristic Algorithms Approach**
Our team implemented and compared three classic heuristic algorithms specifically adapted for optimizing the cutting of large wood panels into smaller rectangular pieces:

| Algorithm     | Description                                                                                                   | Pros                                                             | Cons                                                                       |
| ------------- | ------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- | -------------------------------------------------------------------------- |
| **First-Fit** | Places each required piece in the first available stock sheet where it fits.                                  | Fast, easy to implement; suitable for real-time decision-making  | May leave large unused gaps, leading to inefficiency in material usage     |
| **Best-Fit**  | Places each required piece in the stock sheet where it leaves the least remaining space.                      | Maximizes material utilization, reducing waste                   | Slower than First-Fit due to additional computations for optimal placement |
| **Greedy**    | Combines First-Fit and Best-Fit strategies with an optimization step for merging partially used stock sheets. | Balances performance and efficiency, reduces waste significantly | Slightly more complex to implement and manage                              |

---

### **Reinforcement Learning Approach**
We implemented **Proximal Policy Optimization (PPO)**, **Deep Q-Networks (DQN)**, and **Q-Learning** agents using a custom Gym environment to learn cutting strategies over time.

#### **PPO Approach**
- **Architecture:**
  - Actor-Critic model with CNN for spatial features and MLP for product information.
  - Masks invalid product selections and predicts cutting positions (x, y).
- **Reward Function:**
  - Encourages high cutting efficiency.
  - Penalizes wood waste and excessive panel use.
  - Rewards successful piece placement.
- **Limitations:**
  - Slow learning compared to heuristics.
  - Sensitive to hyperparameter tuning.
  - Poor performance on simple cutting tasks.

#### **DQN Approach**
- **Architecture:**
  - Dueling DQN model with CNN for spatial features and MLP for product information.
  - Estimates state-value and action advantages.
- **Reward Function:**
  - Similar to PPO: Encourages cutting efficiency, penalizes waste, rewards placement.
- **Limitations:**
  - Slow learning compared to heuristics.
  - Sensitive to hyperparameters.
  - Poor performance on simple cutting tasks.

#### **Q-Learning Approach**
- Q-learning focuses on learning the optimal action-value function by iteratively updating Q-values based on rewards, enabling the agent to make decisions that balance immediate rewards with long-term goals.

---

## 📊 **Evaluation Metrics**
| Metric               | Description                                               |
|----------------------|-----------------------------------------------------------|
| **Runtime (s)**       | Average runtime for processing a list of customer orders. |
| **Waste Rate**        | Sum of unused areas in used sheets.                       |
| **Fitness**           | Number of sheets used to fulfill demand.                  |

---
## 🏆 **Results**
The application of **Q-Learning**, **Proximal Policy Optimization (PPO)**, and **Deep Q-Networks (DQN)** in the 2D Cutting Stock Problem (CSP) demonstrates the potential of Reinforcement Learning (RL) in solving complex combinatorial optimization problems. While traditional heuristic methods such as **First-Fit Decreasing (FFD)**, **Best-Fit**, and **Greedy algorithms** provide fast and relatively efficient solutions, RL approaches offer adaptability and the ability to learn from experience.

---

## Slide

```
root
├── _slide
│   └── REL301m_AI17C_Group2_Project.pptx
```

## Report 

```
root
├── report
|   └──Assignment-SE173577_QE170039_QE170144_QE170087_QE170011.pdf
```

## Diary 

```
root
├── diary.md
```

## Effort 

```
root
├── effort.md
```


## 📂 **Folder Structure**
```
├── _report                 
│   └── Assignment-SE173577_QE170039_QE170144_QE170087_QE170011.pdf 
├── _slide                   
│   └── REL301m_AI17C_Group2_Project.pptx              
├── VietLQ_SE173577_submit_code   # PPO and Deep-Q-Learning Approach           
├── ChuongLV_QE170039_submit_code   # Q-Learning Approach          
├── HoangTH_QE170011_submit_code   # Heuristic Approach       
├── PhucNN_QE170087_submit_code   # Heuristic Approach          
└── VietNLQ_QE170144_submit_code   # Q-Learning Approach
```

---

## 👥 **Contributors**
- **Group 2 Members**
  - 🧑‍💻 Nguyễn Lê Quốc Việt - 📧 vietnlqqe170144@fpt.edu.vn - 🎓 Student ID: QE170144 (Leader)
  - 🧑‍💻 Lê Văn Chương - 📧 chuonglvqe170039@fpt.edu.vn - 🎓 Student ID: QE170039  
  - 🧑‍💻 Nguyễn Ngọc Phúc - 📧 phucnnqe170087@fpt.edu.vn - 🎓 Student ID: QE170087  
  - 🧑‍💻 Lê Quốc Việt - 📧 VietLQSE173577@fpt.edu.vn - 🎓 Student ID: SE173577  
  - 🧑‍💻 Trần Hữu Hoàng - 📧 HoangTHQE170011@fpt.edu.vn - 🎓 Student ID: QE170011  

**Instructor:** Dr. Nguyen An Khuong

---

## 🚀 **Future Work**
- 🚢 Implement and test advanced **Heuristic Algorithms**.
- 🤖 Develop improved **RL-based models** to enhance efficiency.
- 📊 Compare different approaches and evaluate performance under varying constraints.

---

## 📜 **License**
This project is licensed under the **MIT License** – See `LICENSE` for more details.

---

**🎓 Capstone Project for REL301M - Reinforcement Learning Course**  
**FPT University – Quy Nhon AI Campus**  
Faculty of Information Technology – Mini Capstone Project – Mar 2025
