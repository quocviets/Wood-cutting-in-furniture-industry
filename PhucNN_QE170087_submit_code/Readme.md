# 🪚 2D Cutting Stock Problem - Heuristic Approach

## 👨‍🎓 **Author:** Nguyễn Ngọc Phúc  
- **Student ID:** QE170087  
- **Class:** AI17C  
- **Course:** REL301m  
- **Instructor:** Nguyễn An Khương  

---

## 📌 **Project Overview**
This project explores the **2D Cutting Stock Problem** (CSP) using heuristic algorithms, specifically the **First-Fit** and **Best-Fit** strategies. The aim is to minimize material waste while meeting demand requirements for specific cut pieces. These heuristic approaches provide a balance between computational efficiency and solution quality, making them practical for real-world applications.

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
   - If FF fails, it searches for the position that minimizes unused space on the stock sheets.  
   - Improves material utilization but requires more computational effort.


---

## 📊 **Comparison**
| Metric              | First-Fit (FF)      | Best-Fit (BF)        |
|---------------------|----------------------|-----------------------|
| Complexity          | Low                  | Moderate              |
| Execution Speed     | Fast                 | Moderate              |
| Waste Reduction     | Moderate             | Better                |
| Computational Cost  | Low                  | Higher                |

---

## 🔧 **Installation & Usage**
1. **Clone the repository:**
```bash
git clone https://github.com/quocviets/Wood-cutting-in-furniture-industry/tree/submit_final/PhucNN_QE170087_submit_code
cd Best_Fit               #If you want to run the Best_Fit algorithm
cd First_Fit              #If you want to run the First_Fit algorithm
cd Combination_Heuristic  #If you want to run the Combination_Heuristic
```

2. **Install dependencies:**
```bash
pip install numpy matplotlib
```

3. **Run the script:**
```bash
python main.py
```

---

## 📁 **Project Structure**
```
📂 PhucNN_QE170087_submit_code/
│──📁 First_Fit/
|   └──📄 Figure_1.png
|   └──📄 Figure_2.png
|   └──📄 Figure_3.png
|   └──📄 Figure_4.png
|   └──📄 reward.png
|   └──📄 summary.png
|   └──📄 waste-summary.png
|   └──📄 first-fit.py
│──📁 Best_Fit/
|   └──📄 Figure_1.png
|   └──📄 Figure_2.png
|   └──📄 Figure_3.png
|   └──📄 Figure_4.png
|   └──📄 reward.png
|   └──📄 summary.png
|   └──📄 waste-summary.png
|   └──📄 best-fit.py
│──📁 Combination_Heuristic/
|   └──📄 combination.py
│──📁 Data/
|   └──📁 First_fit + Best_Fit
|       └──📄 data.json                   #Data for First_Fit and Best_Fit Implementation
|   └──📁 Combination                     
|       └──📄 combination_data.json       #Data for BenchMark Combination Heuristic algorithm
│── 📄 README.md                # Project documentation
```

---

## 📊 **Benchmark - Combination Heuristic Evaluation**
This benchmark evaluates the performance of the combination heuristic approach using the following metrics:
- **Waste Rate:** Ratio of unused area to total required area.
- **Fitness:** Ratio of required area to the total utilized stock area.
- **Runtime:** Time taken to complete the cutting process.

| Order ID | Stock Count | Waste Rate | Fitness | Runtime (s) |
|----------|-------------|------------|---------|-------------|
| Order 001 | 1           | 0.4286       | 0.7    | 0.3        |
| Order 002 | 1           | 0.4286       | 0.7    | 0.46        |
| Order 003 | 2           | 0.7391       | 0.575    | 1.689        |
| Order 004 | 2           | 0.3333       | 0.75    | 2.938        |

*Note: The results are based on sample orders with predefined piece sizes and quantities.*

---

## 📝 **Future Work**
- Implement advanced metaheuristic algorithms like **Genetic Algorithms** or **Simulated Annealing**.  
- Explore reinforcement learning approaches to enhance adaptability and efficiency.  
- Optimize for larger and more complex datasets.

---

## 🤝 **Contact**
If you have any questions or suggestions, feel free to reach out via GitHub issues or email at **your_email@example.com**.

---
🌟 **Thank you for checking out this project!**

