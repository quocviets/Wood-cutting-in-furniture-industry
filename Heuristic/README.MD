# 🪵 Cutting Stock Problem - Heuristic Approach

## 📌 Introduction
This project implements the **2D Cutting Stock Problem** using heuristic algorithms: **First Fit** and **Best Fit**. The goal is to optimize the cutting of rectangular pieces from a larger stock material while minimizing waste.

## 🛠️ Algorithms Used
- **First Fit**: Places each piece in the first available space.
- **Best Fit**: Finds the placement that minimizes wasted space.

## 📊 Results
The visualization of the cutting layout shows how the pieces are arranged within the stock material. The algorithm comparison helps determine which method results in less material waste.

### First Fit Layout
![First Fit Layout](./first_fit_layout.png)

### Best Fit Layout
![Best Fit Layout](./best_fit_layout.png)
The visualization of the cutting layout shows how the pieces are arranged within the stock material. The algorithm comparison helps determine which method results in less material waste.

## 🔢 Waste Calculation
The amount of unused material is calculated as:
```
Waste = Total Stock Area - Used Area
```
Results:
- **First Fit Waste**: XX square units
- **Best Fit Waste**: XX square units

## 📷 Visualization
Both algorithms generate graphical representations of the cutting layouts to help analyze efficiency.

## ▶️ How to Run
1. Install dependencies:  
   ```bash
   pip install numpy matplotlib
   ```
2. Run the script:  
   ```bash
   python cutting_stock_heuristic.py
   ```

## 📌 Conclusion
- **Best Fit** often results in lower waste but takes more computation time.
- **First Fit** is faster but may leave more empty space.
- The choice depends on the specific requirements of the manufacturing process.

🚀 **Optimize your material usage and reduce waste with heuristics!**

