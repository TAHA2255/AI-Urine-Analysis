import os
import requests
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from PIL import Image
import io
import fitz  # PyMuPDF
from openai import OpenAI
import mimetypes

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


app = Flask(__name__)

# Load reference chart
REFERENCE_IMAGE_PATH = "reference.png"  # e.g., strip color chart

# Diagnosis map
DIAGNOSIS_GUIDE = """
1 and 2 = Normal.
3 and 4 = Urinary tract infection probability.
5 and 6 = Urinary tract infection probability.
7 = Normal.
8, 9, 10 =  Initial screening for a Liver issue probability.
11 and 12 = Normal.
13–15 = Initial screening for a kidney disease probability, also not drinking enough 
amounts of water.
16–18 = Normal.
19–21 = Sign for a urinary tract infection probability.
22 = Normal.
23–26 = Initial indicator for a urinary tract infection or Kidney stones.
27–28 = Normal.
29–32 =  Sign for dehydration; not drinking enough amounts of water.
33–34 = Normal.
35–37 = body is using the fat as a source of energy.
38 = Normal.
39–40 = Liver or gallbladder issues.
41–42 = Normal.
43–45 = Initial diabetes signs.
"""

def download_file_from_drive(drive_url):
    try:
        file_id = drive_url.split("id=")[-1]
        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        response = requests.get(url)
        if response.status_code == 200:
            content_type = response.headers.get('content-type')
            return response.content, content_type
        else:
            return None, None
    except Exception as e:
        print(f"Download error: {e}")
        return None, None

def convert_pdf_to_image(pdf_bytes):
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        if doc.page_count == 0:
            raise ValueError("PDF has no pages")
        page = doc.load_page(0)
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        return img
    except Exception as e:
        print(f"❌ Error converting PDF to image: {e}")
        raise
def ask_gpt4o(user_image_b64, reference_image_b64, diagnosis_guide):
    messages = [
        {
            "role": "system",
            "content": "You are a medical assistant. The user will provide you a urine test strip image and a reference strip. Compare them, match the test strip to the closest pads on the reference, and return ONE concise sentence diagnosis using this table."
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "This is the reference chart image."},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + reference_image_b64}}
            ]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"This is the user's test strip image. Based on it and the reference, respond with only one medical diagnosis using this guide:\n\n{diagnosis_guide}"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + user_image_b64}}
            ]
        }
    ]
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=100
    )


    return response.choices[0].message.content.strip()

import base64

def to_base64(image: Image.Image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def detect_file_type(file_bytes):
    if file_bytes.startswith(b"%PDF"):
        return "pdf"
    elif file_bytes.startswith(b"\xff\xd8"):  # JPEG
        return "jpg"
    elif file_bytes.startswith(b"\x89PNG"):
        return "png"
    elif file_bytes[:15].lower().startswith(b"<!doctype html") or b"<html" in file_bytes[:100].lower():
        return "html"
    else:
        return "unknown"


@app.route("/")
def home():
    return "your app is working!"

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    url = data.get("url")
    if not url:
        return jsonify({"error": "No URL provided"}), 400

    file_bytes, content_type = download_file_from_drive(url)
    if not file_bytes:
        return jsonify({"error": "Failed to download file"}), 400

    print("Downloaded content type:", content_type)

    reference_image = Image.open(REFERENCE_IMAGE_PATH)
    reference_b64 = to_base64(reference_image)

    try:
        file_type = detect_file_type(file_bytes)
        print("Detected file type:", file_type)

        if file_type == "pdf":
            print("Processing as PDF")
            user_image = convert_pdf_to_image(file_bytes)
        elif file_type in ["jpg", "png"]:
            print("Processing as image")
            user_image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        elif file_type == "html":
            raise ValueError("Google Drive returned an HTML page instead of the actual file. Check the file permissions.")
        else:
            raise ValueError("Unknown or unsupported file type.")

        user_b64 = to_base64(user_image)
        result = ask_gpt4o(user_b64, reference_b64, DIAGNOSIS_GUIDE)

        return jsonify({"result": result})

    except Exception as e:
        print("❌ Exception during processing:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
