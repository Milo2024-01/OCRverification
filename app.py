# app.py - Python backend only
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import tempfile
from datetime import datetime
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import pytesseract
import PyPDF2
from pdf2image import convert_from_path
import re

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf'}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Set Tesseract path
try:
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
except:
    pass  # Use system PATH if not set

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Serve the main HTML page"""
    try:
        return send_file('index.html')
    except:
        return """
        <html>
            <head><title>Document Verification System</title></head>
            <body>
                <h1>Document Verification System</h1>
                <p>Server is running but index.html not found.</p>
                <p>Make sure index.html is in the same directory as app.py</p>
            </body>
        </html>
        """

@app.route('/verify', methods=['POST'])
def verify_documents():
    """Main verification endpoint"""
    try:
        # Get form data
        id_type = request.form.get('id_type', '')
        if not id_type:
            return jsonify({'success': False, 'message': 'ID type is required'}), 400
        
        # Check required files
        if 'id_file' not in request.files or 'selfie_file' not in request.files:
            return jsonify({'success': False, 'message': 'ID and Selfie files are required'}), 400
        
        id_file = request.files['id_file']
        selfie_file = request.files['selfie_file']
        
        # Validate files
        if id_file.filename == '' or selfie_file.filename == '':
            return jsonify({'success': False, 'message': 'No selected file'}), 400
        
        if not (allowed_file(id_file.filename) and allowed_file(selfie_file.filename)):
            return jsonify({'success': False, 'message': 'File type not allowed'}), 400
        
        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save files temporarily
            id_path = os.path.join(temp_dir, secure_filename(id_file.filename))
            selfie_path = os.path.join(temp_dir, secure_filename(selfie_file.filename))
            
            id_file.save(id_path)
            selfie_file.save(selfie_path)
            
            # Verify Primary ID
            id_verification = verify_primary_id(id_path, id_type, selfie_path)
            
            # Handle optional files
            optional_files = {}
            if 'payslip_file' in request.files and request.files['payslip_file'].filename:
                payslip_file = request.files['payslip_file']
                payslip_path = os.path.join(temp_dir, secure_filename(payslip_file.filename))
                payslip_file.save(payslip_path)
                optional_files['payslip'] = process_optional_document(payslip_path, 'payslip')
            
            if 'company_id_file' in request.files and request.files['company_id_file'].filename:
                company_file = request.files['company_id_file']
                company_path = os.path.join(temp_dir, secure_filename(company_file.filename))
                company_file.save(company_path)
                optional_files['company_id'] = process_optional_document(company_path, 'company_id')
            
            # Prepare response
            response = {
                'success': True,
                'timestamp': datetime.now().isoformat(),
                'id_type': id_type,
                'id_verification': id_verification,
                'optional_files': optional_files,
                'message': 'Verification completed successfully'
            }
            
            return jsonify(response)
            
    except Exception as e:
        print(f"Verification error: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Internal server error: {str(e)}'
        }), 500

def verify_primary_id(id_path, id_type, selfie_path):
    """Verify the primary ID document"""
    verification = {
        'is_authentic': False,
        'confidence': 0,
        'analysis': {},
        'issues': [],
        'id_type': id_type,
        'verification_methods': []
    }
    
    try:
        # Convert PDF to image if needed
        if id_path.lower().endswith('.pdf'):
            try:
                images = convert_from_path(id_path)
                if images:
                    # Save first page as image
                    temp_img_path = os.path.join(tempfile.gettempdir(), 'temp_id_image.png')
                    images[0].save(temp_img_path, 'PNG')
                    id_image_path = temp_img_path
                else:
                    verification['issues'].append('Could not convert PDF to image')
                    return verification
            except:
                verification['issues'].append('PDF conversion failed')
                return verification
        else:
            id_image_path = id_path
        
        # Load images
        id_img = cv2.imread(id_image_path)
        selfie_img = cv2.imread(selfie_path)
        
        if id_img is None:
            verification['issues'].append('Could not read ID image')
            return verification
        
        if selfie_img is None:
            verification['issues'].append('Could not read selfie image')
            return verification
        
        # Perform various checks
        checks = []
        
        # 1. Image Quality Check
        quality_score = check_image_quality(id_img)
        verification['analysis']['image_quality'] = f"{quality_score}/100"
        checks.append(('Image Quality', quality_score > 50, quality_score))
        
        # 2. Face Detection in ID
        id_faces = detect_faces(id_img)
        verification['analysis']['faces_in_id'] = len(id_faces)
        checks.append(('Face in ID', len(id_faces) > 0, 100 if len(id_faces) > 0 else 0))
        
        # 3. Face Detection in Selfie
        selfie_faces = detect_faces(selfie_img)
        verification['analysis']['faces_in_selfie'] = len(selfie_faces)
        checks.append(('Face in Selfie', len(selfie_faces) > 0, 100 if len(selfie_faces) > 0 else 0))
        
        # 4. Text Extraction and Validation
        text_data = extract_and_validate_text(id_image_path, id_type)
        verification['analysis']['text_extracted'] = 'Yes' if text_data['has_text'] else 'No'
        verification['analysis']['id_pattern_match'] = text_data['pattern_match']
        checks.append(('Text Extraction', text_data['has_text'], 80 if text_data['has_text'] else 0))
        checks.append(('ID Pattern', text_data['pattern_match'], 90 if text_data['pattern_match'] else 30))
        
        # 5. Document Dimension Check
        height, width = id_img.shape[:2]
        verification['analysis']['dimensions'] = f"{width}x{height}"
        dimension_ok = width >= 300 and height >= 300
        checks.append(('Document Dimensions', dimension_ok, 100 if dimension_ok else 40))
        
        # 6. Basic Tampering Detection
        tampering_score = check_basic_tampering(id_img)
        verification['analysis']['tampering_risk'] = f"{tampering_score}%"
        checks.append(('Tampering Risk', tampering_score < 50, 100 - tampering_score))
        
        # Calculate overall confidence
        total_score = sum(score for _, _, score in checks)
        confidence = total_score / (len(checks) * 100) * 100
        verification['confidence'] = round(confidence, 1)
        
        # Determine if authentic (threshold: 60% confidence)
        verification['is_authentic'] = confidence >= 60
        
        # Add verification methods used
        verification['verification_methods'] = [
            'Image Quality Analysis',
            'Face Detection',
            'Text Extraction & Pattern Matching',
            'Basic Tampering Detection'
        ]
        
        # Log issues if confidence is low
        if confidence < 60:
            verification['issues'].append(f'Low confidence score: {confidence}%')
            if not text_data['has_text']:
                verification['issues'].append('No readable text found in ID')
            if tampering_score > 50:
                verification['issues'].append('High tampering risk detected')
        
    except Exception as e:
        verification['issues'].append(f'Verification error: {str(e)}')
        verification['confidence'] = 0
    
    return verification

def check_image_quality(image):
    """Check basic image quality"""
    try:
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Calculate Laplacian variance (sharpness)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Calculate contrast
        contrast = gray.std()
        
        # Calculate brightness
        brightness = gray.mean()
        
        # Combined quality score (0-100)
        sharpness_score = min(100, laplacian_var / 10)
        contrast_score = min(100, contrast / 2)
        brightness_score = 100 - abs(brightness - 127) / 127 * 100
        
        quality_score = (sharpness_score * 0.5 + contrast_score * 0.3 + brightness_score * 0.2)
        
        return round(quality_score)
    except:
        return 50  # Default average score

def detect_faces(image):
    """Detect faces in an image using OpenCV"""
    try:
        # Load face cascade
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        return faces
    except Exception as e:
        print(f"Face detection error: {e}")
        return []

def extract_and_validate_text(image_path, id_type):
    """Extract text from ID and validate patterns"""
    result = {
        'has_text': False,
        'pattern_match': False,
        'extracted_text': '',
        'keywords_found': []
    }
    
    try:
        # Extract text using pytesseract
        if image_path.lower().endswith('.pdf'):
            try:
                images = convert_from_path(image_path)
                if images:
                    text = pytesseract.image_to_string(images[0])
                else:
                    return result
            except:
                return result
        else:
            try:
                image = Image.open(image_path)
                
                # Enhance image for better OCR
                enhancer = ImageEnhance.Contrast(image)
                enhanced = enhancer.enhance(2.0)
                enhancer = ImageEnhance.Sharpness(enhanced)
                enhanced = enhancer.enhance(2.0)
                
                text = pytesseract.image_to_string(enhanced)
            except:
                return result
        
        result['extracted_text'] = text
        result['has_text'] = len(text.strip()) > 0
        
        # Look for ID-specific patterns
        text_lower = text.lower()
        keywords = []
        
        if id_type == 'passport':
            keywords = ['passport', 'republic', 'government', 'number', 'date', 'expiry']
            if re.search(r'\b[A-Z]{1,2}\d{6,8}\b', text.upper()):
                result['pattern_match'] = True
        
        elif id_type == 'driver_license':
            keywords = ['driver', 'license', 'licence', 'dl', 'permit', 'expires']
            if re.search(r'\b[A-Z]{1,2}\d{6,9}\b', text.upper()):
                result['pattern_match'] = True
        
        elif id_type == 'national_id':
            keywords = ['national', 'identity', 'id', 'card', 'number', 'republic']
            if re.search(r'\b\d{9,12}\b', text):
                result['pattern_match'] = True
        
        # Check for keywords
        found_keywords = [kw for kw in keywords if kw in text_lower]
        result['keywords_found'] = found_keywords
        
        # If enough keywords found, consider it a pattern match
        if len(found_keywords) >= 2 and not result['pattern_match']:
            result['pattern_match'] = True
        
    except Exception as e:
        print(f"OCR error: {str(e)}")
    
    return result

def check_basic_tampering(image):
    """Basic tampering detection"""
    try:
        score = 0
        
        # Check for unnatural edges
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        
        edge_density = np.sum(edges > 0) / edges.size
        
        if edge_density < 0.01 or edge_density > 0.5:
            score += 30
        
        # Check color consistency
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hue_std = np.std(hsv[:,:,0])
        saturation_std = np.std(hsv[:,:,1])
        
        if hue_std > 50 or saturation_std > 50:
            score += 30
        
        # Check size
        height, width = image.shape[:2]
        if height < 300 or width < 300:
            score += 20
        
        # Check aspect ratio
        aspect_ratio = width / height
        common_ratios = [1.5, 1.33, 1.78, 0.67, 0.75]
        if not any(abs(aspect_ratio - ratio) < 0.1 for ratio in common_ratios):
            score += 20
        
        return min(100, score)
        
    except:
        return 50  # Default medium risk

def process_optional_document(file_path, doc_type):
    """Process optional documents (payslip, company ID)"""
    result = {
        'type': doc_type,
        'uploaded': True,
        'file_name': os.path.basename(file_path),
        'file_size': os.path.getsize(file_path),
        'verification_notes': 'This document type is not verified for authenticity'
    }
    
    if doc_type == 'payslip':
        try:
            if file_path.lower().endswith('.pdf'):
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    result['page_count'] = len(pdf_reader.pages)
                    
                    if pdf_reader.pages:
                        text = pdf_reader.pages[0].extract_text()
                        result['has_text'] = len(text.strip()) > 0
                        if text:
                            keywords = ['salary', 'payslip', 'employee', 'period', 'net', 'gross']
                            found = [kw for kw in keywords if kw in text.lower()]
                            result['keywords_found'] = found
            else:
                result['is_image'] = True
        except:
            pass
    
    return result

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'service': 'Document Verification API'
    })

@app.route('/test', methods=['GET'])
def test_endpoint():
    """Test endpoint"""
    return jsonify({
        'message': 'Document Verification API is running',
        'endpoints': {
            'GET /': 'Frontend',
            'POST /verify': 'Verify documents',
            'GET /health': 'Health check',
            'GET /test': 'Test endpoint'
        }
    })

if __name__ == '__main__':
    print("=" * 60)
    print("Document Verification System")
    print("=" * 60)
    
    # Check for index.html
    if os.path.exists('index.html'):
        print("✓ Found index.html")
    else:
        print("✗ Missing index.html - Please create index.html file")
    
    print("\nStarting server...")
    print("Open: http://localhost:5000")
    print("Press Ctrl+C to stop\n")
    
    # Create necessary directories
    os.makedirs('uploads', exist_ok=True)
    
    app.run(debug=True, host='0.0.0.0', port=5000)