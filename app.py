from flask import Flask, render_template,request,jsonify
from flask_cors import CORS
from chat import get_response

app = Flask(__name__)
CORS(app)
@app.route("/", methods=["GET"])
def index_get():
    return render_template("base.html")

@app.route("/predict", methods=["POST"])
def predict():
    text = request.get_json(force=True).get("message")
# text = request.is_json().get("message")
# text = "hi"
    response = get_response(text)
    message = {"answer": response}
    return jsonify(message)

if __name__ == "__main__":
    app.run(debug=True)

