# Counterfeit Medicine Detection System

## Problem Statement

Counterfeit medicines are a serious global health problem. The WHO estimates
that 1 in 10 medical products in developing countries is substandard or
falsified. This project uses machine learning to flag potentially counterfeit
medicines based on their characteristics.

## Dataset

- **Records:** 50,000 medicine entries
- **Features:** Drug name, manufacturer, batch number, country, dates, price, dosage
- **Label:** Genuine (1) or Counterfeit (0)
- **Source:** Synthetic dataset for academic use

## ML Pipeline

```
Load CSV → Clean Data → Engineer 8 Features → Train 3 Models → Pick Best → Save counterfeit_model.pkl
```

**Models compared:**
- Logistic Regression
- Decision Tree
- Random Forest

**Features used:**

| Feature | Description |
|---|---|
| Price Ratio | Price vs average for that drug |
| Days Until Expiry | Remaining shelf life |
| Medicine Age | Days since manufacture |
| Shelf Life | Total shelf life in days |
| Batch Length | Length of batch number |
| Mfr Frequency | How often manufacturer appears |
| Country Risk | Counterfeit rate by country |
| Price Per Dose | Price per mg of dosage |

## Project Structure

```
CounterfeitMedicine/
│
├── app.py                  — Flask routes
├── train_model.py          — Model training
├── predict.py              — Prediction + reasons
├── feature_engineering.py  — Feature creation
├── database.py             — SQLite operations
├── eda.py                  — Exploratory data analysis
├── config.py               — Central path configuration
├── requirements.txt        — Dependencies
├── README.md               — Project documentation
├── .gitignore              — Git ignore rules
│
├── data/
│   └── medicine_dataset.csv — Training dataset
│
├── models/
│   └── counterfeit_model.pkl — Trained model
│
├── static/
│   └── css/
│       └── style.css       — Application styles
│
├── templates/
│   ├── base.html           — Base layout template
│   ├── home.html           — Landing page
│   ├── predict.html        — Assessment form
│   ├── result.html         — Risk report
│   ├── history.html        — Prediction log
│   └── about.html          — Project info
│
├── reports/
│   └── eda/                — Generated charts
│
└── database.db             — SQLite database
```

## How to Run

```bash
pip install -r requirements.txt
python eda.py            # Optional: generate EDA charts
python train_model.py    # Train the model
python app.py            # Start the web app
```

Open `http://localhost:5000` in your browser.

## Limitations

- Statistical estimate, not a laboratory test
- Accuracy depends on training data quality
- Cannot detect physical tampering or chemical composition
- Not a substitute for professional drug verification
- Built for educational/academic purposes only

## Future Improvements

- Add more features (packaging quality, QR code verification)
- Use a larger, real-world dataset
- Deploy on a cloud platform
- Add user authentication for the reporting system
