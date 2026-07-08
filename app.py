from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load("model/spam_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    message = request.form["message"]
    data = vectorizer.transform([message])
    prediction = model.predict(data)

    if prediction[0] == 1:
        result = "🚨 This is a Spam Email!"
    else:
        result = "✅ This is Not Spam."

    return render_template("index.html", prediction=result)

if __name__ == "__main__":
    app.run(host="127.0.0.1",
port=5000,  debug=True)
   