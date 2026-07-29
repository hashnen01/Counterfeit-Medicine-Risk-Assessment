"""
Flask application for the Counterfeit Medicine Risk Assessment System.
"""

from flask import Flask, render_template, request

from database import (
    create_table,
    save_prediction,
    get_predictions,
    search_prediction,
)

from predict import load_model, predict


app = Flask(__name__)

# Create database and load model
create_table()
model, stats = load_model()


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict", methods=["GET", "POST"])
def predict_medicine():

    if request.method == "POST":

        form_data = {
            "medicine_name": request.form.get("medicine_name", ""),
            "manufacturer": request.form.get("manufacturer", ""),
            "batch_number": request.form.get("batch_number", ""),
            "country": request.form.get("country", ""),
            "manufacture_date": request.form.get("manufacture_date", ""),
            "expiry_date": request.form.get("expiry_date", ""),
            "dosage": request.form.get("dosage", 100),
            "price": request.form.get("price", 0),
        }

        risk, confidence, reasons = predict(
            form_data,
            model,
            stats,
        )

        save_prediction(
            form_data["medicine_name"],
            form_data["manufacturer"],
            risk,
            confidence,
        )

        return render_template(
            "result.html",
            data=form_data,
            risk=risk,
            confidence=confidence,
            reasons=reasons,
        )

    manufacturers = [
        name
        for name in stats["known_manufacturers"]
        if "fake" not in name.lower()
    ]

    return render_template(
        "predict.html",
        drugs=stats["known_drugs"],
        countries=stats["known_countries"],
        manufacturers=manufacturers,
    )


@app.route("/history")
def history():

    search = request.args.get("q", "").strip()

    if search:
        predictions = search_prediction(search)
    else:
        predictions = get_predictions()

    return render_template(
        "history.html",
        predictions=predictions,
        query=search,
    )


@app.route("/about")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    app.run(debug=True)
