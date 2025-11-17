from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load("models/salary_prediction_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    job_title = request.form["job_title"]
    rating = float(request.form["rating"])
    location = request.form["location"]
    python_skill = int(request.form["python"])
    aws_skill = int(request.form["aws"])

    # Build a dataframe EXACTLY like the training data
    data = pd.DataFrame([{
    "Job Title": job_title,
    "Salary Estimate": "0",
    "Job Description": "Unknown",
    "Rating": rating,
    "Company Name": "Unknown",
    "Location": location,
    "Headquarters": "Unknown",
    "Size": "Unknown",
    "Founded": 0,
    "Type of ownership": "Unknown",
    "Industry": "Unknown",
    "Sector": "Unknown",
    "Revenue": "Unknown",
    "Competitors": "Unknown",
    "hourly": 0,
    "employer_provided": 0,
    "min_salary": 0,
    "max_salary": 0,
    "avg_salary": 0,
    "company_txt": "Unknown",
    "job_state": "Unknown",
    "same_state": 0,
    "age": 0,
    "python_yn": python_skill,
    "R_yn": 0,
    "spark": 0,
    "aws": aws_skill,
    "excel": 0
}])


    prediction = model.predict(data)[0]

    return render_template("index.html", prediction=round(prediction, 2))

if __name__ == "__main__":
    app.run(debug=True)
