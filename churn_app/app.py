from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
from model_utils import (
    train_churn_model,
    predict_single_customer,
    SHAP_SUMMARY_PLOT_PATH,
    CHURN_DISTRIBUTION_PLOT_PATH,
    TENURE_CHURN_PLOT_PATH,
    MONTHLY_CHARGES_CHURN_PLOT_PATH,
    load_trained_model,
)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)


@app.route("/")
def index():
    model_trained = os.path.exists("churn_model.pkl")
    shap_available = os.path.exists(SHAP_SUMMARY_PLOT_PATH)
    charts = {
        "churn_distribution": os.path.exists(CHURN_DISTRIBUTION_PLOT_PATH),
        "tenure_churn": os.path.exists(TENURE_CHURN_PLOT_PATH),
        "monthly_charges_churn": os.path.exists(MONTHLY_CHARGES_CHURN_PLOT_PATH),
    }
    return render_template(
        "index.html",
        model_trained=model_trained,
        shap_available=shap_available,
        charts=charts,
    )


@app.route("/train", methods=["POST"])
def train():
    if "dataset" not in request.files:
        return "No file part", 400

    file = request.files["dataset"]
    if file.filename == "":
        return "No selected file", 400

    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)

    target_col = request.form.get("target_col", "Churn")

    try:
        metrics = train_churn_model(file_path, target_col=target_col)
    except Exception as e:
        model_trained = os.path.exists("churn_model.pkl")
        shap_available = os.path.exists(SHAP_SUMMARY_PLOT_PATH)
        charts = {
            "churn_distribution": os.path.exists(CHURN_DISTRIBUTION_PLOT_PATH),
            "tenure_churn": os.path.exists(TENURE_CHURN_PLOT_PATH),
            "monthly_charges_churn": os.path.exists(MONTHLY_CHARGES_CHURN_PLOT_PATH),
        }
        return render_template(
            "index.html",
            error=str(e),
            model_trained=model_trained,
            shap_available=shap_available,
            charts=charts,
        )

    model_trained = True
    shap_available = os.path.exists(SHAP_SUMMARY_PLOT_PATH)
    charts = {
        "churn_distribution": os.path.exists(CHURN_DISTRIBUTION_PLOT_PATH),
        "tenure_churn": os.path.exists(TENURE_CHURN_PLOT_PATH),
        "monthly_charges_churn": os.path.exists(MONTHLY_CHARGES_CHURN_PLOT_PATH),
    }

    return render_template(
        "index.html",
        success="Model trained successfully!",
        roc_auc=metrics["roc_auc"],
        f1=metrics["f1"],
        model_trained=model_trained,
        shap_available=shap_available,
        charts=charts,
    )


@app.route("/dashboard")
def dashboard():
    model_trained = os.path.exists("churn_model.pkl")
    return render_template(
        "dashboard.html",
        model_trained=model_trained,
        shap_available=os.path.exists(SHAP_SUMMARY_PLOT_PATH),
        charts={
            "churn_distribution": os.path.exists(CHURN_DISTRIBUTION_PLOT_PATH),
            "tenure_churn": os.path.exists(TENURE_CHURN_PLOT_PATH),
            "monthly_charges_churn": os.path.exists(MONTHLY_CHARGES_CHURN_PLOT_PATH),
        },
    )


@app.route("/predict", methods=["GET", "POST"])
def predict():
    model = load_trained_model()
    if model is None:
        return redirect(url_for("index"))

    if request.method == "GET":
        example_fields = {
            "tenure": "",
            "MonthlyCharges": "",
            "TotalCharges": "",
            "Contract": "",
            "PaymentMethod": "",
            "gender": "",
            "SeniorCitizen": "",
            "InternetService": "",
        }
        return render_template("predict.html", fields=example_fields)

    else:
        input_data = {}
        for key, value in request.form.items():
            if value.strip() == "":
                input_data[key] = None
                continue
            try:
                if "." in value:
                    input_data[key] = float(value)
                else:
                    input_data[key] = int(value)
            except ValueError:
                input_data[key] = value

        try:
            result = predict_single_customer(input_data)
        except Exception as e:
            return render_template("predict.html", fields=input_data, error=str(e))

        return render_template(
            "predict.html",
            fields=input_data,
            probability_round=round(result["probability"] * 100, 2),
            raw_probability=result["probability"],
            label=result["label"],
        )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.json
    try:
        result = predict_single_customer(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)
