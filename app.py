import random
from flask import Flask, render_template, request

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    eta = None
    explanation = None
    bus_line = ""
    stop = ""

    if request.method == "POST":
        bus_line = request.form.get("bus_line", "").strip()
        stop = request.form.get("stop", "").strip()

        # Mock ETA prediction for now.
        eta = random.randint(3, 15)
        explanation = "Traffic is high, slight delay expected."

    return render_template(
        "index.html",
        eta=eta,
        explanation=explanation,
        bus_line=bus_line,
        stop=stop,
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
