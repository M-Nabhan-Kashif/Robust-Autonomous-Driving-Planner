# Robust Decision-Making for Autonomous Driving  
*CS378: Geometric Foundations of Data Science – Final Project*  
University of Texas at Austin

## 🚗 Overview
This project implements and compares five planning strategies for autonomous driving in noisy, uncertain environments using the `highway-env` simulator. We investigate how sensor noise and model uncertainty affect agent performance and safety, culminating in a novel **Policy-Conditioned Uncertainty Planner (PCU)** using Bayesian dropout.

## 🧠 Key Features
- **Model Implementations**:  
  - Random Planner (Baseline)  
  - Interval Robust Control  
  - Adaptive Planner  
  - Hierarchical Robust Planner  
  - Policy-Conditioned Uncertainty (PCU) Planner  
- **Visualizations**:  
  - Real-time metric overlays (reward, collisions)  
  - Summary dashboards with comparative performance  
  - Recorded video demos of trained policies

## 🎥 Demo Video  
Watch the final presentation here:  
👉 [Video Demo (Box)](https://utexas.app.box.com/s/jdxnumflysroja7o6w8hctycq4exvovg/file/1852638243848)

## 📁 Contents
- `Geometric_Foundations_of_Data_Science_Project.ipynb`: Full codebase with training, testing, and visualizations  
- `final-report.pdf`: Written analysis, results, and discussion  
- `videos/`: Recorded runs of each planner under sensor uncertainty

## 🛠️ Tools & Libraries
- Python, NumPy, Matplotlib  
- `highway-env`, OpenCV, Gymnasium  
- Bayesian Dropout for uncertainty modeling

## 📈 Results
The **PCU Planner** achieved up to **169% improvement** over the baseline in cumulative reward while maintaining lower collision rates under uncertainty.
