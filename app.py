from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # CORS 허용


@app.route("/generate", methods=["POST"])
def generate():
    data = request.json

    prompt = data.get("prompt")
    temperature = data.get("temperature", 0.7)

    # 👉 지금은 AI 대신 더미 응답
    result = f"[temp={temperature}] → {prompt} 에 대한 결과"

    return jsonify({"result": result})


if __name__ == "__main__":
    app.run(port=5000, debug=True)
