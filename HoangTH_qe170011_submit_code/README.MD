# 🪚 2D Cutting Stock Problem - Heuristic Approach

## 👨‍🎓 **Author:** [Trần Hữu Hoàng]  
- **Student ID:** [Qe170011]  
- **Class:** [AI17C]  
- **Course:** [REL301m]  
- **Instructor:** [Nguyễn An Khương ]  

---

## 📌 **Project Overview**
This project explores the **2D Cutting Stock Problem** (CSP) using heuristic algorithms, specifically the **First-Fit**, **Best-Fit**, and **Greedy** strategies. The aim is to minimize material waste while meeting demand requirements for specific cut pieces. These heuristic approaches provide a balance between computational efficiency and solution quality, making them practical for real-world applications.

---

## 🔎 **Problem Description**
- **Objective:** Efficiently cut smaller pieces from larger stock sheets while minimizing waste.
- **Constraints:**  
  - No overlapping between cut pieces.  
  - Pieces must remain within the boundaries of the stock sheet.  
  - Fulfill all quantity demands for each piece type.  

---

## 💡 **Approach**
The solution applies a combination of heuristic algorithms to optimize the placement of pieces:

1. **First-Fit (FF):**  
   - Places each piece in the first available position of the stock sheets.  
   - Simple and fast, but may lead to higher waste.

2. **Best-Fit (BF):**  
   - Searches for the position that minimizes unused space on the stock sheets.  
   - Improves material utilization but requires more computational effort.

3. **Greedy Algorithm:**
   - Prioritizes placing larger pieces first to minimize unused space.  
   - Balances simplicity and efficiency.

---

## 📊 **Comparison**
| Metric              | First-Fit (FF)      | Best-Fit (BF)        | Greedy                |
|----------------------|----------------------|-----------------------|------------------------|
| Complexity          | Low                  | Moderate              | Low                    |
| Execution Speed     | Fast                 | Moderate              | Fast                   |
| Waste Reduction     | Moderate             | Better                | Good                   |
| Computational Cost  | Low                  | Higher                | Low                    |

---

## 🔧 **Installation & Usage**
1. **Clone the repository:**
```bash
git clone https://github.com/[YourUsername]/Cutting-Stock-Problem.git
cd Cutting-Stock-Problem
```

2. **Install dependencies:**
```bash
pip install numpy matplotlib
```

3. **Run the script:**
```bash
python main.py                   ## run 3 algorithm and evaluate
python create_animation.py       ## create gif for 3 algorithm
```

---

## 📁 **Project Structure**
```
📂 Cutting-Stock-Problem/
│──📁 data/
|   └──📄 data_1.json
|   └──📄 data_2.json
|   └──📄 data_3.json
|   └──📄 data_4.json
│──📁 policy/
|   └──📄 FirstFit.py
|   └──📄 BestFit.py
|   └──📄 Greedy.py
│──📁 utils/
|   └──📄 evaluate.py
|   └──📄 visual.py
|   └──📄 animation.py
|   └──📄 GenerateColors.py
│──📁 results_Heuristic_Algorithms/
|   └──📁 results_data_1
|   └──📁 results_data_2
|   └──📁 results_data_3
|   └──📁 results_data_4
│──📄 create_animation.py
│──📄 main.py
│──📄 README.md
```

---

## 📊 **Benchmark - Algorithm Evaluation**
This benchmark evaluates the performance of the heuristic approaches using the following metrics:
- **Waste Rate:** Ratio of unused area to total required area.
- **Fitness Score:** Ratio of required area to the total utilized stock area.
- **Runtime:** Time taken to complete the cutting process.

| Algorithm  | Waste Rate | Fitness Score | Runtime (s) |
|------------|------------|--------------|-------------|
| First-Fit  | 0.4286     | 0.7          | 0.3         |
| Best-Fit   | 0.3333     | 0.75         | 2.9         |
| Greedy     | 0.3600     | 0.72         | 1.5         |

*Note: The results are based on sample orders with predefined piece sizes and quantities.*

---

🌟 **Thank you for checking out this project!**

