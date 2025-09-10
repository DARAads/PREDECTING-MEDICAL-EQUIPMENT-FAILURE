# 🚑 Predicting Medical Equipment Failure – Life Science Hack 2025  

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)  
[![XGBoost](https://img.shields.io/badge/ML-XGBoost-orange.svg)](https://xgboost.readthedocs.io/)  
[![Spark](https://img.shields.io/badge/Streaming-Spark-lightgrey.svg)](https://spark.apache.org/)  
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)  

---

## 📌 Overview  
This repository contains our project for **Life Science Hack 2025** built by **Team Elite Force (Team 10, RMK College of Engineering & Technology)**.  
We developed an **AI-driven predictive maintenance system** to forecast failures in medical equipment before they occur — improving hospital safety, reducing downtime, and minimizing maintenance costs.

---

## 📝 Problem Statement  
Hospitals depend on critical medical devices such as ventilators, infusion pumps, and CT scanners.  
Unplanned failures cause:
- Increased downtime (30–40% preventable)
- 25–30% higher maintenance costs compared to predictive maintenance
- Direct risks to patient safety

**Goal**: Build a real-time machine learning pipeline to predict device failures and alert technicians proactively.

---

## 💡 Solution  
- A **Machine Learning pipeline (XGBoost)** predicts failure risks using device data like:
  - Device type & name  
  - Environmental conditions (temperature, pressure, vibration)  
  - Operational features (location, climate control, runtime hours)  
- **Live streaming data** is ingested, processed, and scored in near real-time.  
- A **Gradio web dashboard** visualizes predicted failure risks to technicians for preventive action.

---

## ⚙️ Architecture  
1. **Data Collection** → Simulated medical device telemetry  
2. **Preprocessing & Feature Engineering** → Spark pipeline with HiveMQ Cloud  
3. **Model Training & Testing** → XGBoost classifier in Python  
4. **Prediction Storage** → Azure Blob Storage (CSV blobs)  
5. **Frontend Visualization** → Gradio Web Interface  
6. **Deployment** → Python backend serving model predictions  

---

## 🧠 Machine Learning Model  
- **Algorithm**: XGBoost (Extreme Gradient Boosting)  
- **Why XGBoost**:
  - High performance on structured/tabular data
  - Handles missing values & regularization

### Evaluation Metrics
- **Accuracy**  
- **Weighted F1-Score**  
- **Precision & Recall** (via classification report)

---

## 🗂️ Tech Stack  

| Component        | Technology |
|-----------------|-------------|
| Programming Language | Python |
| ML Library | XGBoost, scikit-learn |
| Data Streaming | HiveMQ Cloud |
| Data Processing | Spark Pipeline |
| Storage | Azure Blob Storage |
| Web UI | Gradio |

---

## 📊 Impact  
- Reduces unplanned downtime and costs  
- Enables proactive maintenance scheduling  
- Improves hospital operational efficiency and patient safety  

---

## 🔗 Resources  
- **Presentation Video**: [Google Drive Link](https://drive.google.com/file/d/1R1wdAMfT9Jjeyy8SMM4cKy3YMruU2L5k/view?usp=sharing)

---

## 🚀 How to Run  

1. **Clone the repository**  
   ```bash
   git clone https://github.com/<your-username>/Predicting-Medical-Equipment-Failure.git
   cd Predicting-Medical-Equipment-Failure
Install dependencies

bash
Copy code
pip install -r requirements.txt
Prepare dataset

Place SynDataset.csv in the project root folder.

Train the model

bash
Copy code
python train_model.py
Launch the Gradio web interface

bash
Copy code
python app.py
