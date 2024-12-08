from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from gradio_client import Client, handle_file
import random
import os
import requests

# Flask uygulamasını başlat
app = Flask(__name__)
CORS(app)

# Gradio istemcisini başlat
client = Client("Hatman/AWS-Nova-Canvas")

# Çıktıları saklamak için bir klasör
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.route("/process-image", methods=["POST"])
def process_image():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    # Resmi al ve kaydet
    image_file = request.files["image"]
    image_path = os.path.join(OUTPUT_DIR, image_file.filename)
    image_file.save(image_path)

    # Prompt ve parametreler
    prompt = "Complete the missing parts of this building to show what it would have looked like in its prime."
    negative_prompt = "Do not add any modern elements. in Turkey."

    try:
        result = client.predict(
            images=[handle_file(image_path)],
            text=prompt,
            negative_text=negative_prompt,
            similarity_strength=0.7,
            height=1024,
            width=1024,
            quality="standard",
            cfg_scale=10,
            seed=random.randint(0, 10),
            api_name="/image_variation",
        )

        # Yanıt türüne göre işleme
        if isinstance(result, bytes):
            output_path = os.path.join(OUTPUT_DIR, "output.png")
            with open(output_path, "wb") as f:
                f.write(result)
            return jsonify({"message": "Image processed successfully", "download_url": f"/download/{os.path.basename(output_path)}"})

        elif isinstance(result, str):
            output_path = os.path.join(OUTPUT_DIR, "output.txt")
            with open(output_path, "w") as f:
                f.write(result)
            return jsonify({"message": "Text result saved successfully", "download_url": f"/download/{os.path.basename(output_path)}"})

        elif isinstance(result, dict):
            if "output_url" in result:
                response = requests.get(result["output_url"])
                output_path = os.path.join(OUTPUT_DIR, "output.png")
                with open(output_path, "wb") as f:
                    f.write(response.content)
                return jsonify({"message": "Image downloaded successfully", "download_url": f"/download/{os.path.basename(output_path)}"})
            else:
                return jsonify({"error": "Unexpected JSON response structure", "content": result}), 500

        else:
            return jsonify({"error": "Unexpected API response format", "content": result}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/download/<filename>", methods=["GET"])
def download_file(filename):
    file_path = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    else:
        return jsonify({"error": "File not found"}), 404

if __name__ == "__main__":
    app.run(debug=True, port=5000)
