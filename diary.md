### Reinforcement Learning - REL301m  
**Group 2**
**Instructors: Nguyễn An Khương**  

**Group Members:**  
Nguyễn Lê Quốc Việt - QE170144 (Leader)
Lê Quốc Việt - SE173577 
Lê Văn Chương - QE170039 
Trần Hữu Hoàng - QE170011 
Nguyễn Ngọc Phúc - QE170087

---

### **Week 5: Laying the Foundation**  
**Goal:** Discuss the topic and divide tasks.  
- **Achievements:**  
  - Defined the problem as an Integer Linear Programming (ILP) task, focusing on cost function goals.  
  - Split into two teams:  
    - **Reinforcement Learning (RL):** Lê Quốc Việt, Hữu Hoàng, Văn Chương.  
    - **Heuristics & ILP:** Ngọc Phúc (first-fit, best-fit, hybrid methods).  
    - Long-term roles:  
      - LaTeX: Lê Quốc Việt, Ngọc Phúc.  
      - Python implementation: Hữu Hoàng, Văn Chương, Nguyễn Lê Quốc Việt.  

---

### **Week 6: Formalizing the Problem**  
**Goal:** Refine the ILP formulation and discuss heuristics.  
- **Achievements:**  
  - Việt Lê Quốc completed the ILP formulation, introducing column generation for piece selection strategies:  
    1. Pick the largest piece.  
    2. Optimize remaining space with the best-fitting piece.  
  - Drafted sections of the LaTeX report (formulation, abstract, acknowledgments).  
  - Agreed to prioritize coding progress before refining heuristics.  

---

### **Week 7: Diving into RL and Heuristics**  
**Goal:** Explore RL networks, policies, and new heuristic ideas.  
- **Achievements:**  
  - Upgraded RL networks from linear models to convolution-based neural networks, trading runtime for accuracy and adaptability.  
  - Proposed a hybrid heuristic combining first-fit and simplex algorithms:  
    - Generate cutting arrangements using first-fit.  
    - Optimize choices with simplex.  
    - Đạt volunteered to implement this.  
  - Added visual aids (diagrams/examples) to clarify formulations in the LaTeX report.  

---

### **Week 8: Training the RL Agent**  
**Goal:** Refine RL policies, finalize report sections, and analyze training results.  
- **Achievements:**  
  - Phúc enhanced first-fit with a hybrid first-fit/best-fit approach.  
  - Introduced rewards inversely proportional to item placement height, improving packing efficiency.  
  - The RL agent showed significant improvement:  
    - Items no longer placed randomly in the middle.  
    - Reduced gaps between items.  

---

### **Week 9: Adapting to Rotatable Items**  
**Goal:** Finalize the report and adjust algorithms for rotatable items.  
- **Achievements:**  
  - Added visuals and diagrams to enhance clarity in the report.  
  - Updated algorithms to handle item rotation effectively.  

---

### **Week 10: Final Push**  
**Goal:** Benchmark algorithms, fix issues, and submit.  
- **Achievements:**  
  - Benchmarked algorithms on 4 fixed orders:  
    - **Winner:** Combination heuristic.  
    - **Runner-up:** Trained RL agent.  
  - Extended benchmarking to 46 more orders, confirming the same ranking.  
  - Finalized all components of the project, ensuring quality and completeness.  

---

**Key Takeaways:**  
- Hybrid approaches (e.g., combination heuristic) consistently outperformed standalone methods.  
- RL agents demonstrated impressive adaptability after iterative improvements. We believe that with better setups and more rigorous training and testing, this approach can handle even more complicate orders. 
- Visual aids and clear documentation were crucial for conveying complex ideas effectively.  

**Final Thoughts:**  
Through collaboration, creativity, and persistence, the team successfully tackled the cutting-stock problem, blending ILP, heuristics, and RL to deliver robust solutions.  

**Boxed Final Result:**  
$\boxed{\text{Combination Heuristic > Trained RL Agent > Other Methods}}$