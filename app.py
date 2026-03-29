from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Load accuracy
with open("accuracy.txt", "r") as f:
    accuracy = f.read()


@app.route("/")
def home():
    return render_template("index.html", accuracy=accuracy)


@app.route("/predict", methods=["POST"])
def predict():
    Glucose = float(request.form["Glucose"])
    BloodPressure = float(request.form["BloodPressure"])
    BMI = float(request.form["BMI"])
    Age = float(request.form["Age"])

    # Build input DataFrame for model
    input_data = pd.DataFrame(
        [[Glucose, BloodPressure, BMI, Age]],
        columns=["Glucose", "BloodPressure", "BMI", "Age"],
    )

    # Raw model prediction (0 or 1)
    pred_class = model.predict(input_data)[0]  # 0 or 1

    # Convert to human-readable text
    if pred_class == 1:
        prediction_text = "Diabetes Detected"
    else:
        prediction_text = "No Diabetes Detected"

    # ---------------------------
    # 2. Build reasons based on inputs
    # ---------------------------
    reasons = []

    if Glucose >= 140:
        reasons.append(
            "Your glucose level is higher than the recommended range, which can indicate poor blood sugar control."
        )
    elif Glucose >= 126:
        reasons.append(
            "Your glucose level is in a borderline range; this can increase the chance of developing diabetes."
        )

    if BMI >= 30:
        reasons.append(
            "Your BMI is in the obesity range, which strongly increases the risk of type 2 diabetes."
        )
    elif BMI >= 25:
        reasons.append(
            "Your BMI is in the overweight range; extra body weight makes it harder for the body to use insulin properly."
        )

    if Age >= 45:
        reasons.append(
            "Age above 45 years is associated with a higher chance of type 2 diabetes."
        )

    if BloodPressure >= 90:
        reasons.append(
            "Higher blood pressure is often seen together with diabetes and other metabolic problems."
        )

    if not reasons:
        reasons.append(
            "Based on the entered values, your current numbers are closer to a healthier range, but regular checkups are still important."
        )

    reason_text = " ".join(reasons)

    # ---------------------------
    # 3. Precautions / advice text
    # ---------------------------
    if pred_class == 0:  # No diabetes detected
        advice_text = (
            "Maintain a balanced diet, stay physically active at least 30 minutes most days, "
            "keep a healthy weight, avoid smoking, and go for regular checkups to keep your risk low."
        )
    else:  # Diabetes detected
        advice_text = (
            "Please talk to a doctor for proper tests and guidance. Focus on a diet with less sugar "
            "and processed food, more vegetables, whole grains, and high-fiber foods. Try to exercise "
            "regularly, work towards a healthy weight, and monitor your blood sugar and blood pressure as advised."
        )

    # Render template once with final text
    return render_template(
        "result.html",
        prediction=prediction_text,  # "Diabetes Detected" or "No Diabetes Detected"
        accuracy=accuracy,
        reason_text=reason_text,
        advice_text=advice_text,
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)