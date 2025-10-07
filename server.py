
# server.py
# Tiny Flask server for Atlas Beta
# - Accepts POST /api/analyze with multipart/form-data file 'image'
# - Runs analyze_image from atlas_image_beta.py (heuristic + model if model.pth present)
# - Additionally runs pytesseract OCR (if installed) to include OCR-extracted text in response
# Requirements (server): pip install flask pillow pytesseract opencv-python-headless
# If using pytesseract, Tesseract must be installed on the server (system package).
from flask import Flask, request, jsonify
from atlas_image_beta import analyze_image  # file created earlier; reuse its analyze_image function
from PIL import Image
import io, base64, os, traceback

app = Flask(__name__)

# Optional: secondary OCR via pytesseract (may be more robust on server)
try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except Exception:
    PYTESSERACT_AVAILABLE = False

def ocr_server_image(image_path):
    if not PYTESSERACT_AVAILABLE:
        return None
    try:
        text = pytesseract.image_to_string(Image.open(image_path))
        return text
    except Exception as e:
        return None

@app.route('/api/analyze', methods=['POST'])
def analyze():
    f = request.files.get('image')
    if not f:
        return jsonify({'error':'no image provided'}), 400
    # save temporary
    tmp = '/tmp/atlas_upload.png'
    f.save(tmp)
    try:
        # run the python analyzer we created earlier
        res = analyze_image(tmp)
        # add server-side OCR if available
        ocr_text = ocr_server_image(tmp)
        if ocr_text:
            res.setdefault('meta', {})['server_ocr_text'] = ocr_text
        # include annotated image as base64 for frontend convenience
        annotated_path = res.get('annotated_image')
        if annotated_path and os.path.exists(annotated_path):
            with open(annotated_path, 'rb') as rfile:
                b = base64.b64encode(rfile.read()).decode('utf-8')
            res['annotated_image_base64'] = 'data:image/png;base64,' + b
        return jsonify({'signal': res.get('signal'), 'confidence': res.get('confidence'), 'reasons': res.get('reasons'), 'meta': res.get('meta', {}), 'annotated_image_base64': res.get('annotated_image_base64')})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error':'analysis_failed', 'details': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
