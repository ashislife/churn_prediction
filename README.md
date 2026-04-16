# 📊 Telecom Customer Churn Prediction using ANN

An end-to-end Machine Learning web application that predicts whether a telecom customer will churn or not using an Artificial Neural Network (ANN).  
The model is deployed using Streamlit for real-time interactive predictions.

---

## 🚀 Live Demo
👉 https://churnprediction-iibdp4jj6qgqcnj4mmhngt.streamlit.app/

---

## 📁 Project Structure

churn_prediction/
├── app.py                          # Streamlit web app
├── churn_ds.csv                    # Dataset
├── model.h5                        # Trained ANN model
├── scaler.pkl                      # Feature scaling object
├── features.pkl                    # Feature columns used in training
├── Telecom Customer Churn Prediction using ANN.ipynb
├── requirements.txt                # Dependencies
├── runtime.txt                     # Python version for deployment
├── README.md                       # Project documentation
├── LICENSE
├── .gitignore
└── venv/                           # Virtual environment (not pushed)

---

## 🧠 Problem Statement

Customer churn is a major issue in the telecom industry.  
This project predicts whether a customer will leave the service based on historical data.

---

## ⚙️ Tech Stack

- Python 3.10
- TensorFlow / Keras
- Scikit-learn
- Pandas
- NumPy
- Streamlit
- Joblib

---

## 🧪 Machine Learning Approach

- Data preprocessing and cleaning
- Feature scaling using StandardScaler
- Artificial Neural Network (ANN) model using TensorFlow/Keras
- Binary classification: Churn / Not Churn
- Model saved as model.h5 for inference

---

## 📊 Features Used

- Customer demographics
- Account information
- Contract type
- Payment method
- Tenure
- Service usage details

---

## 🖥️ How to Run Locally

### 1. Clone repository
git clone https://github.com/ashislife/churn_prediction
cd churn_prediction

### 2. Create virtual environment
python -m venv venv
venv\Scripts\activate   # Windows

### 3. Install dependencies
pip install -r requirements.txt

### 4. Run Streamlit app
streamlit run app.py

---

## 📦 Model Details

- Model Type: Artificial Neural Network (ANN)
- Framework: TensorFlow / Keras
- Input Scaling: StandardScaler
- Output: Binary Classification (0 = No Churn, 1 = Churn)

---

## 🎯 Future Improvements

- Add probability prediction graph
- Improve UI with dashboards
- Deploy on AWS / Render / HuggingFace
- Add explainable AI (SHAP / LIME)

---

## 👨‍💻 Author

Ashish Kumar  
GitHub: https://github.com/ashislife/churn_prediction

---

## ⭐ Support

If you like this project, please give a ⭐ to the repository.