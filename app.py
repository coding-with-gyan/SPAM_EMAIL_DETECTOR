from flask import Flask, render_template, request, redirect, url_for
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user
import pickle
import sqlite3
from datetime import datetime

app = Flask(__name__)
app.secret_key = "supersecretkey"

# ================= ML LOAD =================
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# ================= DATABASE =================
def init_db():
    conn = sqlite3.connect("history.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS emails
                 (message TEXT, prediction TEXT, probability REAL, time TEXT)''')
    conn.commit()
    conn.close()

init_db()

# ================= LOGIN SYSTEM =================
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

class User(UserMixin):
    def __init__(self, id):
        self.id = id

@login_manager.user_loader
def load_user(user_id):
    return User(user_id)

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        if request.form['username'] == "admin" and request.form['password'] == "1234":
            login_user(User(1))
            return redirect('/')
    return render_template("login.html")

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect('/login')

# ================= HOME =================
@app.route('/')
@login_required
def home():
    conn = sqlite3.connect("history.db")
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM emails WHERE prediction LIKE 'Spam%'")
    spam_count = c.fetchone()[0]

    c.execute("SELECT COUNT(*) FROM emails WHERE prediction LIKE 'Not Spam%'")
    ham_count = c.fetchone()[0]
    conn.close()

    return render_template("index.html", spam_count=spam_count, ham_count=ham_count)

# ================= TEXT PREDICTION =================
@app.route('/predict', methods=['POST'])
@login_required
def predict():
    email = request.form['email']
    data = vectorizer.transform([email])

    result = model.predict(data)[0]
    prob = model.predict_proba(data)[0][1] * 100
    prediction = "Spam 🚫" if result == 1 else "Not Spam ✅"

    conn = sqlite3.connect("history.db")
    c = conn.cursor()
    c.execute("INSERT INTO emails VALUES (?,?,?,?)",
              (email, prediction, prob, datetime.now()))
    conn.commit()
    conn.close()

    return render_template("index.html",
                           prediction=prediction,
                           probability=round(prob,2),
                           spam_count=0,
                           ham_count=0)

# ================= FILE UPLOAD =================
@app.route('/upload_file', methods=['POST'])
@login_required
def upload_file():
    file = request.files['file']
    content = file.read().decode("utf-8")

    data = vectorizer.transform([content])
    result = model.predict(data)[0]
    prob = model.predict_proba(data)[0][1] * 100
    prediction = "Spam 🚫" if result == 1 else "Not Spam ✅"

    return render_template("index.html",
                           prediction=prediction,
                           probability=round(prob,2),
                           spam_count=0,
                           ham_count=0)

# ================= HISTORY =================
@app.route('/history')
@login_required
def history():
    conn = sqlite3.connect("history.db")
    c = conn.cursor()
    c.execute("SELECT * FROM emails ORDER BY time DESC")
    data = c.fetchall()
    conn.close()
    return render_template("history.html", data=data)

# ================= DELETE HISTORY =================
@app.route('/delete_history', methods=['POST'])
@login_required
def delete_history():
    conn = sqlite3.connect("history.db")
    c = conn.cursor()
    c.execute("DELETE FROM emails")
    conn.commit()
    conn.close()
    return redirect(url_for('home'))

if __name__ == "__main__":
    app.run(debug=True)
