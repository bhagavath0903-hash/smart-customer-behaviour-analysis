# 📊 Smart Customer Behavior Analysis & Churn Prediction System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red?style=for-the-badge&logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-orange?style=for-the-badge&logo=scikit-learn)
![Plotly](https://img.shields.io/badge/Plotly-5.22-3d9be9?style=for-the-badge&logo=plotly)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**An end-to-end AI-powered customer intelligence platform for data-driven business decisions.**

</div>

---

## 🚀 Project Overview

The **Smart Customer Behavior Analysis & Churn Prediction System** is a professional-grade, industry-ready analytics platform built entirely in Python and Streamlit. It provides businesses with comprehensive tools to:

- **Understand** customer purchase behavior and spending patterns
- **Segment** customers into actionable groups using KMeans clustering
- **Predict** customer churn with 4 trained ML models
- **Visualize** insights through an interactive BI-style dashboard
- **Generate** automated business reports in PDF, CSV, and Excel

---

## ✨ Features

| Feature | Description |
|---|---|
| 📂 **Dataset Upload** | CSV / Excel upload with instant preview and validation |
| 🧹 **Data Cleaning** | Remove duplicates, handle missing values, encode features, scale |
| 📊 **Exploratory Data Analysis** | 20+ auto-generated charts: histograms, heatmaps, scatter plots, pair plots |
| 👥 **Customer Segmentation** | KMeans clustering with PCA 2D visualization and cluster profiles |
| 🤖 **Churn Prediction** | Train 4 models, auto-select best, ROC curve, confusion matrix, manual predict |
| 📈 **Analytics Dashboard** | Live KPI cards, distribution charts, churn heatmap, ML comparison |
| 💡 **Business Insights** | Auto-generated findings, retention strategies, growth suggestions |
| 📄 **Report Export** | PDF (fpdf2), CSV, and multi-sheet Excel with all results |
| 🌙 **Dark Mode** | Professional dark theme throughout |

---

## 🏗️ Project Structure

```
Customer-Behavior-Analysis/
│
├── app.py                          # Main entry point
├── requirements.txt                # All dependencies
├── README.md                       # This file
├── generate_data.py                # Sample data generator
│
├── .streamlit/
│   └── config.toml                 # Dark theme configuration
│
├── assets/                         # Static assets (logo, etc.)
│
├── data/
│   └── sample_customer_data.csv    # 1000 synthetic customer records
│
├── models/                         # Saved ML models (auto-created)
│   ├── churn_model.pkl
│   ├── scaler.pkl
│   └── encoder.pkl
│
├── pages/                          # Streamlit page modules
│   ├── Dashboard_Home.py           # Landing / Home page
│   ├── Data_Upload.py              # Upload & preview
│   ├── Data_Cleaning.py            # Cleaning pipeline
│   ├── EDA.py                      # Exploratory Analysis
│   ├── Segmentation.py             # KMeans clustering
│   ├── Churn_Prediction.py         # ML training & prediction
│   ├── Dashboard.py                # BI Dashboard
│   ├── Business_Insights.py        # Insights & recommendations
│   └── Report.py                   # PDF/CSV/Excel export
│
├── utils/                          # Backend utility modules
│   ├── preprocessing.py            # Data cleaning & encoding
│   ├── visualization.py            # Reusable chart functions
│   ├── segmentation.py             # KMeans & PCA utilities
│   └── prediction.py               # ML training & evaluation
│
└── reports/                        # Generated reports output directory
```

---

## 🛠️ Technologies Used

| Technology | Version | Purpose |
|---|---|---|
| Python | 3.10+ | Core language |
| Streamlit | 1.35 | Web UI framework |
| Pandas | 2.2 | Data manipulation |
| NumPy | 1.26 | Numerical operations |
| Matplotlib | 3.9 | Static charts |
| Seaborn | 0.13 | Statistical visualization |
| Plotly | 5.22 | Interactive charts |
| Scikit-Learn | 1.5 | Machine Learning |
| Joblib | 1.4 | Model persistence |
| fpdf2 | 2.7 | PDF report generation |
| Openpyxl | 3.1 | Excel export |

---

## ⚙️ Installation

### Prerequisites
- Python 3.10 or higher
- pip package manager

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/Customer-Behavior-Analysis.git
cd Customer-Behavior-Analysis

# 2. Create a virtual environment (recommended)
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

# 3. Install all dependencies
pip install -r requirements.txt

# 4. Generate sample data
python generate_data.py

# 5. Launch the application
streamlit run app.py
```

---

## 📖 Usage Guide

1. **Home Page** – Read the overview and click **Start Analysis**
2. **Upload Dataset** – Upload your CSV file or load the sample dataset
3. **Data Cleaning** – Configure and apply the cleaning pipeline
4. **EDA** – Explore distributions, correlations, and categorical breakdowns
5. **Customer Segmentation** – Select features, choose K, run KMeans clustering
6. **Churn Prediction** – Train all 4 models, view metrics, predict manually
7. **Dashboard** – View the complete BI-style analytics dashboard
8. **Business Insights** – Read auto-generated recommendations
9. **Download Report** – Export PDF, CSV, or Excel report

---

## 📊 Dataset Format

Your CSV should contain customer data. The system auto-detects column types. For churn prediction, include a **`Churn`** column with `Yes`/`No` or `1`/`0` values.

**Recommended columns:**
- `CustomerID` – Unique identifier
- `Age`, `Gender`, `City` – Demographics
- `AnnualIncome`, `MonthlyCharges`, `TotalSpend` – Financial
- `Tenure`, `Contract`, `PaymentMethod` – Subscription info
- `SpendingScore`, `Satisfaction` – Behavioral scores
- `Churn` – Target variable (Yes/No)

---

## 🤖 Machine Learning Models

| Model | Description |
|---|---|
| Logistic Regression | Baseline linear classifier |
| Decision Tree | Interpretable tree-based model |
| Random Forest | Ensemble of 200 trees |
| Gradient Boosting | Sequential boosting classifier |

The best model is selected automatically by **F1 Score** and saved to `models/churn_model.pkl`.

---

## 📸 Screenshots

*(Add screenshots of your running application here)*

---

## 🔮 Future Enhancements

- [ ] Real-time streaming data support
- [ ] Customer lifetime value prediction
- [ ] Advanced NLP sentiment analysis from reviews
- [ ] Automated email campaign integration
- [ ] REST API endpoints for model serving
- [ ] Multi-user authentication system
- [ ] Scheduled report delivery via email
- [ ] Custom dashboard builder (drag-and-drop)

---

## 📄 License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

Built as an internship / portfolio project demonstrating end-to-end data science, ML engineering, and full-stack Python development skills.

---

<div align="center">
⭐ <strong>Star this repository if you found it helpful!</strong>
</div>
