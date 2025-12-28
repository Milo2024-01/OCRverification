import os
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from detector.fake_detector import FakeDocumentDetector
from PIL import Image
import pytesseract
import face_recognition

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf', 'png', 'jpg', 'jpeg'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

detector = FakeDocumentDetector()  # Your rule-based fake/legit logic

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files or 'selfie' not in request.files:
        return jsonify({'error': 'File or selfie not uploaded'}), 400
    
    file = request.files['file']
    selfie = request.files['selfie']
    
    if not allowed_file(file.filename) or not allowed_file(selfie.filename):
        return jsonify({'error': 'File type not allowed'}), 400
    
    # Save uploads temporarily
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
    selfie_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(selfie.filename))
    file.save(file_path)
    selfie.save(selfie_path)

    # OCR text extraction
    text = pytesseract.image_to_string(Image.open(file_path))
    
    # Fake vs legit detection
    result = detector.analyze_document(text)
    
    # Face matching
    try:
        id_image = face_recognition.load_image_file(file_path)
        selfie_image = face_recognition.load_image_file(selfie_path)
        id_enc = face_recognition.face_encodings(id_image)
        selfie_enc = face_recognition.face_encodings(selfie_image)
        
        if id_enc and selfie_enc:
            similarity = 1 - face_recognition.face_distance([id_enc[0]], selfie_enc[0])[0]
        else:
            similarity = None
    except:
        similarity = None

    result['face_similarity'] = similarity
    
    # Cleanup
    os.remove(file_path)
    os.remove(selfie_path)
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
