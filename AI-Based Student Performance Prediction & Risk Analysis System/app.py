from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open("student_model.pkl", "rb"))

def risk(score):
    if score < 50:
        return "High Risk"
    elif score < 70:
        return "Medium Risk"
    else:
        return "Low Risk"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    features = np.array([[
        float(data["attendance"]),
        float(data["quiz"]),
        float(data["assignment"]),
        float(data["midterm"]),
        # float(data["study_hours"]),
        float(data["gpa"])
    ]])

    prediction = model.predict(features)[0]

    return jsonify({
        "final_grade": round(float(prediction), 2),
        "risk": risk(prediction)
    })

if __name__ == "__main__":
    app.run(debug=True)
