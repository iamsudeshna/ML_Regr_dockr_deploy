# ML_Regr_dockr_deploy

Developed a regression model using Scikit-learn and serialized it using pickle. I exposed it via a Flask API for real-time predictions. The application was containerized using Docker for consistency, and I set up CI/CD using GitHub Actions to automatically deploy it to Render. This created a scalable and reproducible ML inference pipeline.
# Architecture diagram
                ┌──────────────────────┐
                │   Data Sources       │
                │----------------------│
                │ Clinical Trials DB   │
                │ Historical Tenders   │
                │ Site & Geo Data      │
                │ External Signals     │
                └─────────┬────────────┘
                          │
                          ▼
                ┌──────────────────────┐
                │ Data Processing      │
                │----------------------│
                │ Cleaning             │
                │ Missing Value Handle │
                │ Feature Engineering  │
                └─────────┬────────────┘
                          │
                          ▼
                ┌──────────────────────┐
                │ Feature Store        │
                │----------------------│
                │ Curated Features     │
                │ (cost_per_patient,   │
                │ recruitment_score)   │
                └─────────┬────────────┘
                          │
          ┌───────────────┴───────────────┐
          ▼                               ▼
┌──────────────────────┐        ┌──────────────────────┐
│ Model Training       │        │ Model Registry       │
│----------------------│        │----------------------│
│ Train ML Model       │        │ Version Control      │
│ (XGBoost, RF)        │        │ Model Metadata       │
│ Evaluate (RMSE)      │        │                      │
└─────────┬────────────┘        └─────────┬────────────┘
          │                               │
          └───────────────┬───────────────┘
                          ▼
                ┌──────────────────────┐
                │ Model Serving API    │
                │----------------------│
                │ Flask / FastAPI      │
                │ /predict endpoint    │
                └─────────┬────────────┘
                          │
                          ▼
                ┌──────────────────────┐
                │ Deployment Layer     │
                │----------------------│
                │ Docker Container     │
                │ CI/CD (GitHub Actions│
                │ Cloud (Heroku/AWS)   │
                └─────────┬────────────┘
                          │
                          ▼
                ┌──────────────────────┐
                │ End Users / Systems  │
                │----------------------│
                │ Pricing Team         │
                │ Internal Tools       │
                └──────────────────────┘
