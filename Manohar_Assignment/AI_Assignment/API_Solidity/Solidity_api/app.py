from flask import Flask, request, jsonify
import json
import openai  # GPT-4 API

app = Flask(__name__)

# Load contract dataset
with open("contract_prompts.json", "r") as f:
    contract_data = json.load(f)

# OpenAI API Key (Replace with your API key)
openai.api_key = "your-openai-api-key"

@app.route('/generate_contract', methods=['POST'])
def generate_contract():
    """Generates a Solidity smart contract from a text prompt."""
    try:
        data = request.get_json()
        prompt = data.get("prompt", "")

        if not prompt:
            return jsonify({"error": "Prompt is required"}), 400

        # Search for an existing contract in our dataset
        for entry in contract_data:
            if entry["prompt"].lower() == prompt.lower():
                return jsonify({"contract": entry["output"]})

        # If not found, use GPT-4 to generate Solidity code
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "system", "content": "Generate a Solidity smart contract."},
                      {"role": "user", "content": prompt}]
        )

        contract_code = response["choices"][0]["message"]["content"]

        return jsonify({"contract": contract_code})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/status', methods=['GET'])
def status():
    """Returns API health status."""
    return jsonify({"status": "Solidity Generator API is running"})

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
