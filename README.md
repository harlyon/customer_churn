# **🚀 Customer Churn Prediction – End-to-End Production ML System**
## - Real-Time Risk Scoring • SHAP Explainability • Optimized XGBoost Pipeline • FastAPI Deployment

## **📌 Overview**
### This project is a full production-grade customer churn prediction system, built to demonstrate mastery across:
- Data Science & ML Engineering
- MLOps & CI-ready architecture
- Model explainability (SHAP)
- Backend engineering (FastAPI)
- Interactive frontend (HTML/CSS/JS dashboard)
- Real-time inference & batch scoring

The system ingests customer attributes, predicts churn probability using a highly optimized XGBoost pipeline, exposes a scalable REST API, and visualizes predictions with an interactive dashboard including SHAP feature impact.

## **🧱 System Architecture**
1️⃣ Model Layer (Python / Scikit-learn / XGBoost)
- Preprocessing pipeline (StandardScaler + OrdinalEncoder)
- Optuna hyperparameter optimization
- Imbalance handling via dynamic scale_pos_weight
- SHAP explainability with TreeExplainer
- End-to-end training & evaluation script
- Full model pipeline serialization with joblib

## **2️⃣ API Layer (FastAPI)**
- /predict → real-time prediction
- /explain → SHAP values per prediction
- /health → readiness probe for deployment
- Deployed with CORS-enabled endpoints for frontend communication
- Zero-load warm initialization for sub-millisecond inference

## **3️⃣ Frontend Layer (HTML, CSS, JavaScript)**
- Clean, modern dark UI
- Fully visual prediction result card
- SHAP impact panel with dynamic feature importance bars
- Batch prediction via CSV upload
- Real-time refresh button for SHAP re-evaluation
- Soft animations & UX for non-technical users

## **4️⃣ Deployment-Ready Structure**
- Modular src/ folder
- Configurable model paths
- Reproducible environment (requirements.txt)
- Ready for Dockerization, CI pipelines, and cloud deployment

## **🎯 Key Features**
✔ End-to-End ML Pipeline

A complete flow from raw CSV → cleaned dataset → optimized model → serialized pipeline → API → frontend dashboard.

✔ High-Recall Churn Detection
- Prioritizes identifying churners:
- Dynamic thresholding
- Imbalance-aware training
- Recall-oriented scoring strategy

✔ Real-Time Explainability (SHAP)
- Every prediction includes:
- Top impactful features
- Direction of contribution
- Feature-level bars for intuitive understanding
- This is essential for trust, auditability, and business adoption.

✔ Production API (FastAPI)
- JSON request/response schema (Pydantic)
- Designed for low-latency scoring
- Supports batch + streaming-friendly design

✔ Interactive Web Dashboard
- A lightweight UI meant for:
- Product demos
- Business teams
- Stakeholders reviewing model behavior
- Interview environments

✔ Batch Prediction Support

Upload a CSV → get a downloadable results file.

## **📊 Model Performance**
Metric	Score
- AUC	~0.85–0.89
- Recall (Churners)	~70–80% (threshold-optimized)
- Precision	Balanced based on ROI strategy
- Training Time	~2–4 seconds on CPU
  
Business ROI Example
- Customer Lifetime Value (CLV): $1200
- Intervention Cost: $50 per customer
- Model identifies churners early → net ROI maintained at scale

## **🧪 Tech Stack**
Machine Learning
- Python
- Scikit-learn
- XGBoost
- Optuna
- SHAP

Backend
- FastAPI
- Pydantic
- Joblib

Frontend
- Vanilla HTML/CSS/JS
- Responsive components
- SHAP visualization without heavy JS libraries

## **🗂️ Project Structure**
.
├── data/
│   └── Bank_Churn.csv
├── src/
│   ├── config.py
│   ├── train_model.py
│   ├── api.py       ← FastAPI app
│   └── utils/
├── dashboards/
│   └── churn_frontend.html
├── models/
│   └── churn_pipeline.joblib
├── README.md
└── requirements.txt
