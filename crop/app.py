from flask import Flask, render_template, request
from sklearn.ensemble import RandomForestClassifier
import joblib

model1 = joblib.load("rf1.pkl")


app = Flask(__name__)


@app.route("/ho")
def ho():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        neitro = request.form["ni"]
        phosph = request.form["phs"]
        potas = request.form["pot"]
        tempa = request.form["temp"]
        humi = request.form["hum"]
        phv = request.form["ph"]
        rainfall = request.form["rai"]
        inputs = [
            float(neitro),
            float(phosph),
            float(potas),
            float(tempa),
            float(humi),
            float(phv),
            float(rainfall),
        ]
        ans = model1.predict([inputs])
        return render_template("index.html", content=ans[0])
    return render_template("index.html")


if __name__ == "__main__":
    app.run()
