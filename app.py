from flask import Flask, render_template, request, redirect
import joblib
import sqlite3
from datetime import datetime
import re

app = Flask(__name__)

# Load ML model
model = joblib.load("model/spam_model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")


# Database create
def create_database():
    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS history(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT,
        result TEXT,
        probability REAL,
        keywords TEXT,
        date TEXT
    )
    """)

    conn.commit()
    conn.close()


create_database()


# Spam keywords list
spam_keywords = [
    "free",
    "win",
    "winner",
    "lottery",
    "money",
    "offer",
    "prize",
    "click here",
    "urgent",
    "claim",
    "cash",
    "discount",
    "congratulations"
]


# Home page
@app.route("/")
def home():
    return render_template("index.html")


# Prediction
@app.route("/predict", methods=["POST"])
def predict():

    email = request.form["message"]


    # Convert text to vector
    data = vectorizer.transform([email])


    # Prediction
    prediction = model.predict(data)[0]


    # Probability
    probability = model.predict_proba(data)[0]


    spam_probability = round(probability[1] * 100,2)


    if prediction == 1:
        result = "Spam Email"
    else:
        result = "Not Spam"
    from datetime import datetime

    date = datetime.now().strftime("%d-%m-%Y %H:%M")

    # Keyword detection
    found_keywords=[]

    email_lower=email.lower()

    for word in spam_keywords:
        if word in email_lower:
            found_keywords.append(word)


    if len(found_keywords)==0:
        keywords="No suspicious keywords found"
    else:
        keywords=", ".join(found_keywords)



    # Email analysis
    words=len(email.split())
    characters=len(email)

    if spam_probability > 70:
        risk="High"
    elif spam_probability > 40:
        risk="Medium"
    else:
        risk="Low"



    # Save history
    conn=sqlite3.connect("database.db")
    cursor=conn.cursor()

    cursor.execute("""
    INSERT INTO history(email,result,probability,keywords,date)
    VALUES(?,?,?,?,?)
    """,
    (
        email,
        result,
        spam_probability,
        keywords,
        datetime.now().strftime("%d-%m-%Y %H:%M")
    ))

    conn.commit()
    conn.close()



    return render_template(
        "index.html",
        prediction=result,
        probability=spam_probability,
        keywords=keywords,
        words=words,
        characters=characters,
        risk=risk
    )



# Dashboard
@app.route("/dashboard")
def dashboard():

    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM history")
    total = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM history WHERE result='Spam Email'")
    spam = cursor.fetchone()[0]

    ham = total - spam
    if total > 0:
     accuracy = round((ham / total) * 100, 2)
    else:
     accuracy = 0
    

    cursor.execute("""
        SELECT email, result, probability, date
        FROM history
        ORDER BY id DESC
        LIMIT 5
    """)

    recent = cursor.fetchall()

    conn.close()

    return render_template(
        "dashboard.html",
        total=total,
        spam=spam,
        ham=ham,
        accuracy=accuracy,
        recent=recent
        
    )



# History page
@app.route("/history")
def history():

    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM history ORDER BY id DESC")
    data = cursor.fetchall()

    conn.close()

    return render_template(
        "history.html",
        history=data
    )


@app.route("/delete_history")
def delete_history():

    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()

    cursor.execute("DELETE FROM history")

    conn.commit()
    conn.close()

    return redirect("/history")


if __name__=="__main__":
    app.run(debug=True)