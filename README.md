# 💊 Counterfeit Medicine Risk Assessment 

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Flask](https://img.shields.io/badge/Flask-Web%20Framework-black)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Machine%20Learning-orange)
![SQLite](https://img.shields.io/badge/SQLite-Database-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## 📌 Overview

Counterfeit medicines are a major public health concern worldwide. This project is a **Machine Learning-based web application** that estimates the likelihood of a medicine being counterfeit using observable product information such as manufacturer, batch number, price, dosage, manufacturing date, expiry date, and country.

Instead of relying on manufacturer QR codes or specialized hardware, the system analyzes medicine metadata and provides:

- Estimated Counterfeit Probability
- Risk Level (Low / Medium / High)
- Observed Risk Factors
- Prediction History
- Consumer-friendly web interface

> **Note:** This project estimates counterfeit **risk** based on metadata. It does **not** laboratory-confirm or certify whether a medicine is genuine or counterfeit.

---

# ✨ Key Features

- Predict counterfeit risk using Machine Learning
- Feature Engineering for better prediction
- Compare multiple ML models
- Store prediction history using SQLite
- Clean Flask-based web interface
- Simple consumer-friendly workflow
- Easy to extend with real-world datasets

---

# 🛠️ Tech Stack

| Category | Technologies |
|-----------|--------------|
| Programming Language | Python |
| Web Framework | Flask |
| Machine Learning | Scikit-learn |
| Data Processing | Pandas, NumPy |
| Database | SQLite |
| Visualization | Matplotlib |
| Frontend | HTML, CSS |

---

# 🧠 Machine Learning Workflow

```text
Dataset
     │
     ▼
Data Cleaning
     │
     ▼
Feature Engineering
     │
     ▼
Train Multiple Models
(Logistic Regression,
Decision Tree,
Random Forest)
     │
     ▼
Model Evaluation
     │
     ▼
Best Model Selection
     │
     ▼
Risk Prediction
     │
     ▼
Flask Web Application
```

---

# 📂 Project Structure

```text
Counterfeit-Medicine-Risk-Assessment/
│
├── app.py
├── train_model.py
├── predict.py
├── feature_engineering.py
├── generate_dataset.py
├── eda.py
├── database.py
├── config.py
├── requirements.txt
├── README.md
│
├── data/
├── models/
├── reports/
├── scripts/
├── static/
└── templates/
```

---

# Installation

## Clone the Repository

```bash
git clone https://github.com/hashnen01/Counterfeit-Medicine-Risk-Assessment.git
```

## Navigate to the Project

```bash
cd Counterfeit-Medicine-Risk-Assessment
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Train the Model

```bash
python generate_dataset.py
python eda.py
python train_model.py
```

## Run the Application

```bash
python app.py
```

Open your browser and visit:

```
http://localhost:5000
```

---

#  Application Screenshots

> Screenshots will be added soon.

- Home Page
- Prediction Page
- Low Risk Prediction
- High Risk Prediction
- Prediction History
- About Page

---

# ⚠️ Limitations

- Uses a synthetic dataset for demonstration purposes.
- Estimates counterfeit **risk** rather than confirming authenticity.
- Does not perform laboratory verification or chemical analysis.
- Prediction quality depends on the quality of the training data.
- Intended for educational and research purposes.

---

# 🔮 Future Enhancements

- Support real-world pharmaceutical datasets.
- Add QR code verification as an additional input.
- Add packaging image analysis.
- Integrate manufacturer verification APIs.
- Display region-wise counterfeit trends.
- Build an admin dashboard for report management.
- Support continuous model retraining using verified reports.

---

# 👨‍💻 Author

**Hashnen Belim**

B.Tech Computer Science & Engineering  
Parul University

GitHub:
https://github.com/hashnen01

---

# ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub.