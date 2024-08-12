from flask import Flask, render_template, request
import joblib
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route("/display", methods=['GET', 'POST'])
def uploader():
    if request.method == 'POST':
        # Load the model and scaler
        model = joblib.load("diabetes_prediction.pkl")
        sc = joblib.load("scaler.pkl")

        # Collect the form data
        pregnancies = int(request.form["pregnancies"])
        Glucose = int(request.form["Glucose"])
        BloodPressure = int(request.form["BloodPressure"])
        SkinThickness = int(request.form["SkinThickness"])
        Insulin = int(request.form["Insulin"])
        Weight = float(request.form["Weight"])
        Height = float(request.form["Height"])
        BMI = Weight / (Height ** 2)
        DiabetesPedigreeFunction = float(request.form["DiabetesPedigreeFunction"])
        Age = float(request.form["Age"])

        # Prepare the test data
        test_data = [[pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]]

        # Transform the test data with the fitted scaler
        test_data_scaled = sc.transform(test_data)

        # Make predictions
        result_num = model.predict(test_data_scaled)[0]
        probabilities = model.predict_proba(test_data_scaled)[0]

        # Prepare the result
        not_have_prob = round(probabilities[0] * 100, 2)
        have_prob = round(probabilities[1] * 100, 2)

        if result_num == 0:
            result = '"As Not Having Diabetes"'
        else:
            result = '"As Having Diabetes"'

        return render_template("display.html", idx=result, not_have_prob=not_have_prob, have_prob=have_prob)

if __name__ == "__main__":
    app.run(debug=True)
