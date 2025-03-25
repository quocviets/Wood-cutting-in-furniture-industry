## DQN Approach

### **DQN Architecture**

- **Dueling DQN Model:**
  - The architecture uses a **Convolutional Neural Network (CNN)** to extract spatial features from the cutting stock grid:
    - Three convolutional layers (`conv1`, `conv2`, `conv3`) with increasing filter sizes (16, 32, 64) and strides to downsample the grid.
    - The grid is processed into a feature map that captures occupancy patterns.
  - A **Multi-Layer Perceptron (MLP)** encodes product information:
    - The order array (width, height, quantity of each piece type) is flattened and passed through fully connected layers to produce a 128-dimensional embedding.
  - The outputs of the CNN and MLP are concatenated and passed through a shared fully connected layer (`fc_combine`) to create a unified state representation.

- **Dueling Network Structure:**
  - The network splits into two streams:
    - **State-Value Stream ($V(s)$):** Estimates the overall value of the state.
    - **Advantage Stream ($A(s, a)$):** Estimates the advantage of each action relative to the average.
  - The Q-values are computed as:
    $$Q(s, a) = V(s) + \left(A(s, a) - \frac{1}{|A|} \sum_{a'} A(s, a')\right)$$
    This structure improves learning by decoupling the state value from the action advantages.

- **Action Representation:**
  - The Q-values are predicted for all possible actions, represented as a tensor of shape grid_size, grid_size, max_order_types, $\text{rotations}$).
  - Since items are non-rotatable, the rotation dimension is fixed to $0$, simplifying the action space.

- **Action Selection:**
  - During training, invalid actions (e.g., placing pieces in occupied spaces or selecting unavailable piece types) are masked by setting their Q-values to $-\infty$ before action selection.
  - The agent uses an **$\epsilon$-greedy policy**, where it selects random actions with probability $\epsilon$ (decaying over time) and the action with the highest Q-value otherwise.

- **Training Process:**
  - The agent uses **Double DQN** to reduce overestimation bias:
    - The target Q-value is computed as:
      $$\text{Target} = r + \gamma \cdot Q_{\text{target}}(s', \arg\max_{a'} Q(s', a'))$$
    - A target network periodically updates its weights from the main network to stabilize learning.
  - The loss is computed using the **Huber loss** between predicted Q-values and target Q-values:
    $$\text{Loss} = \text{SmoothL1Loss}(Q(s, a), \text{Target})$$
  - The network is optimized using the Adam optimizer with gradient clipping to prevent exploding gradients.

---

### **DQN Reward Function**

The reward function is designed to optimize the wood-cutting process by encouraging efficient placement while penalizing inefficiencies:

- **Successful Placement:**  
  The agent receives a reward proportional to the area of the piece placed ($\text{width} \times \text{height}$) when a piece is successfully placed on the platform.

- **Repositioning Penalty:**  
  If the agent attempts to place a piece at an invalid position but the piece can still fit elsewhere on the current platform, a small penalty ($-10$) is applied.

- **New Platform Penalty:**  
  If the agent cannot place the piece on the current platform and needs to use a new platform, a larger penalty ($-50$) is applied.

- **Invalid Action Penalties:**  
  - Invalid piece type or no remaining pieces of the selected type: $-10$.
  - No platforms available: $-30$.

- **Efficiency Bonus (End of Episode):**  
  At the end of an episode, the agent receives a bonus based on the overall efficiency:
  $$\text{Efficiency Bonus} = \text{efficiency} \times 1000$$
  Where:
  $$\text{efficiency} = \frac{\text{Total Filled Area}}{\text{Total Platform Area Used}}$$

This reward structure encourages high cutting efficiency, penalizes waste, and rewards successful piece placement.

---

### **DQN Limitations Observed**

Based on the implementation and training process in the provided code, the following limitations of the DQN approach were observed:

- **Slow Learning:**  
  The DQN agent requires many episodes to learn optimal policies, especially in environments with large state and action spaces. This slow convergence makes it less suitable for real-time applications compared to heuristic methods.

- **Sensitivity to Hyperparameters:**  
  The performance of the DQN agent is highly sensitive to hyperparameter choices, such as the learning rate, discount factor ($\gamma$), exploration rate ($\epsilon$), and replay buffer size. Suboptimal settings can lead to poor learning or instability.

- **Poor Performance on Simple Tasks:**  
  On smaller datasets or simpler cutting orders, the DQN agent may underperform compared to heuristic algorithms like First-Fit or Best-Fit. This is because the neural network's capacity and generalization capabilities are better suited for complex scenarios.

- **Exploration Challenges:**  
  Despite the $\epsilon$-greedy policy, the agent may struggle to explore efficiently in environments with sparse rewards or highly constrained action spaces, leading to suboptimal policies in certain scenarios.

- **Memory and Computational Overhead:**  
  The use of a replay buffer and target network introduces additional memory and computational overhead, which can slow down training and inference, especially for large-scale problems.

### Training Outputs

- Trained DQN models are stored in google drive, please refer the README file in `models/` directory for detail about downloading the final trained model.
- Training plots saved in `result/`, tracking agent performance over episodes.

### Project Structure
```
VietLQ_SE173577_submit_code    
├── deep_q_network_approach             
│   ├── README            # Provided detail about the concept of implementing the agent     
│   ├── requirements      # Required package for running the code        
│   ├── agent.py          # Code for setup the deep-q-network agent             
│   ├── environment.py    # Code for setup the environment             
│   ├── main.py           # Main code for training the agent in cli   
│   ├── training.py       # Code for setup the training pipeline
│   ├── test_on_custom_order.ipynb     # Notebook for tesing and visualizing agent on a defined customer order
│   ├── training_with_visualize.py     # Notebook for training and visualizing the training progress                      
│   ├── result                    
│   │   ├── dqn_training_result.png    # Visualize the training result
```

### Get Started
#### **Step 1: Set Up Virtual Environment & Install Dependencies**

- Create a new virtual environment (or use an existing one).
    
- Install the required packages by running:
    
```bash
    pip install -r requirements.txt
```

#### **Step 2: Download the Trained Model**

- Download the pre-trained model from:  
    🔗 [Google Drive Link](https://drive.google.com/file/d/1jSqHK3YyNICuC3Yg2XDXSVI2-OwDkLhP/view?usp=drive_link)
    - The downloaded file will be named `dqn_wood_cutting_final.pth`.

#### **Step 3: Place the Model in the Correct Directory**

- Move the downloaded model file (`dqn_wood_cutting_final.pth`) into the `models/` folder.
    - Create the folder if it doesn’t exist.

#### **Step 4: Test on a Custom Order**

- Run the Jupyter notebook `test_on_custom_order.ipynb` to evaluate the model on a specific customer order,where you can define the wood pieces to be cut from the sheet. You can check the notebook [here](test_on_custom_order.ipynb)
    