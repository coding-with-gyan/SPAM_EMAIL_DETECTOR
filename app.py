from flask import Flask, render_template, request, redirect, url_for
import pickle
import sqlite3
from datetime import datetime
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user

app = Flask(__name__)
app.secret_key = "secretkey123"

# ================= ML FILES LOAD =================
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# ================= DATABASE SETUP =================
def init_db():
    conn = sqlite3.connect("history.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS emails
                 (message TEXT, prediction TEXT, probability REAL, time TEXT)''')
    conn.commit()
    conn.close()

init_db()

# ================= LOGIN SETUP =================
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

class User(UserMixin):
    def __init__(self, id):
        self.id = id

@login_manager.user_loader
def load_user(user_id):
    return User(user_id)

# ================= LOGIN ROUTE =================
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username == "admin" and password == "1234":
            user = User(1)
            login_user(user)
            return redirect(url_for('home'))

    return render_template("login.html")

# ================= LOGOUT =================
@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

# ================= HOME PAGE =================
@app.route('/')
@login_required
def home():
    return render_template("index.html")

# ================= PREDICTION =================
@app.route('/predict', methods=['POST'])
@login_required
def predict():
    email = request.form['email']
    data = vectorizer.transform([email])

    result = model.predict(data)[0]
    prob = model.predict_proba(data)[0][1] * 100

    prediction = "Spam 🚫" if result == 1 else "Not Spam ✅"

    # Save history
    conn = sqlite3.connect("history.db")
    c = conn.cursor()
    c.execute("INSERT INTO emails VALUES (?,?,?,?)",
              (email, prediction, prob, datetime.now()))
    conn.commit()
    conn.close()

    return render_template("index.html",
                           prediction=prediction,
                           probability=round(prob, 2))

# ================= HISTORY PAGE =================
@app.route('/history')
@login_required
def history():
    conn = sqlite3.connect("history.db")
    c = conn.cursor()
    c.execute("SELECT * FROM emails ORDER BY time DESC")
    data = c.fetchall()
    conn.close()
    return render_template("history.html", data=data)

# ================= RUN APP =================
if __name__ == "__main__":
    app.run(debug=True)
