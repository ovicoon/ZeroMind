from flask import Flask, request, jsonify
from flask_cors import CORS
from mistralai import Mistral

app = Flask(__name__)
CORS(app)

# 🔑 Mistral API 키
API_KEY = "YOUR_API_KEY"
client = Mistral(api_key=API_KEY)


@app.route("/generate", methods=["POST"])
def generate():
    data = request.json

    prompt = data.get("prompt")
    temperature = data.get("temperature", 0.7)

    try:
        # 👉 Mistral 호출
        response = client.chat.complete(
            model="mistral-small-latest",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )

        result = response.choices[0].message.content

        return jsonify({"result": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(port=5000, debug=True)
