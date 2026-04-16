# 📊 Telecom Customer Churn Prediction using ANN

## 🚀 Project Overview

This project predicts whether a telecom customer will **stay** or **leave (churn)** using an **Artificial Neural Network (ANN)** model.

Customer churn prediction helps telecom companies identify customers who are likely to leave and take preventive actions to retain them.

---

## 🎯 Objective

* Predict customer churn using Machine Learning & Deep Learning
* Build ANN model using TensorFlow/Keras
* Deploy model using Streamlit
* Host project on GitHub

---

## 🧠 Model Used

* Artificial Neural Network (ANN)
* Activation Function: ReLU & Sigmoid
* Optimizer: Adam
* Loss Function: Binary Crossentropy

---

## 📂 Project Structure

```
churn_prediction/
│
├── app.py                # Streamlit Web App
├── model.h5              # Trained ANN Model
├── scaler.pkl            # StandardScaler
├── requirements.txt      # Dependencies
├── README.md             # Project Documentation
└── Telecom_Churn.ipynb   # Model Training Notebook
```

---

## 📊 Features Used

* CreditScore
* Age
* Tenure
* Balance
* NumOfProducts
* HasCrCard
* IsActiveMember
* EstimatedSalary
* Geography (Germany/Spain)
* Gender

---

## ⚙️ Installation

### 1️⃣ Clone Repository

```
git clone https://github.com/your-username/churn_prediction.git
cd churn_prediction
```

### 2️⃣ Create Virtual Environment

```
python -m venv venv
```

Activate environment:

**Windows**

```
venv\Scripts\activate
```

**Linux/Mac**

```
source venv/bin/activate
```

### 3️⃣ Install Requirements

```
pip install -r requirements.txt
```

---

## ▶️ Run Streamlit App

```
streamlit run app.py
```

App will open in browser:

```
http://localhost:8501
```

---

## 🌐 Deployment (GitHub + Streamlit Cloud)

1. Push project to GitHub
2. Go to Streamlit Cloud
3. Connect GitHub repository
4. Select `app.py`
5. Deploy 🚀

---

## 📈 Model Performance

* Accuracy: ~85% (Approx.)
* Binary Classification Problem

---

## 🖥️ App Preview

Users enter customer details and system predicts:

✅ Customer Will Stay
❌ Customer Will Churn

---

## 🛠️ Technologies Used

* Python
* TensorFlow / Keras
* Scikit-learn
* Pandas
* NumPy
* Streamlit
* GitHub

---

## 👨‍💻 Author

**Ashish Kumar**
AIML Student
Role: Model Development & Training

---

## ⭐ Future Improvements

* Add Database Support
* Improve Model Accuracy
* Add Customer Dashboard
* Cloud Deployment

---

## 📜 License

This project is for educational and academic purposes.
