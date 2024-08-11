from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route("/display", methods=['GET', 'POST'])
def uploader():
    if request.method == 'POST':
        model = joblib.load("diabetes_prediction.pkl")

        pregnancies = request.form["pregnancies"]
        pregnancies = int(pregnancies)

        Glucose = request.form["Glucose"]
        Glucose = int(Glucose)

        BloodPressure = request.form["BloodPressure"]
        BloodPressure = int(BloodPressure)

        SkinThickness = request.form["SkinThickness"]
        SkinThickness = int(SkinThickness)

        Insulin = request.form["Insulin"]
        Insulin = int(Insulin)

        Weight = request.form["Weight"]
        Weight = float(Weight)

        Height = request.form["Height"]
        Height = float(Height)

        BMI = Weight / (Height ** 2)

        DiabetesPedigreeFunction = request.form["DiabetesPedigreeFunction"]
        DiabetesPedigreeFunction = float(DiabetesPedigreeFunction)

        Age = request.form["Age"]
        Age = float(Age)

        test_data = [[pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]]

        result_num = model.predict(test_data)[0]
        probabilities = model.predict_proba(test_data)[0]

        not_have_prob = round(probabilities[0] * 100, 2)
        have_prob = round(probabilities[1] * 100, 2)

        if result_num == 0:
            result = '"As Not Having Diabetes"'
        else:
            result = '"As Having Diabetes"'

        return render_template("display.html", idx=result, not_have_prob=not_have_prob, have_prob=have_prob)


if __name__ == "__main__":
    app.run(debug=True)