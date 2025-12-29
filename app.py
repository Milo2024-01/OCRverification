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
import json

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf', 'txt', 'doc', 'docx'}  # Added more extensions
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Set Tesseract path (update this path based on your system)
try:
    # For Windows
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
except:
    try:
        # For Linux/Mac
        pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
    except:
        pass  # Use system PATH if not set

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def allowed_file_by_type(file_obj, allowed_types):
    """Check if file type is allowed"""
    if file_obj.filename == '':
        return False
    
    file_ext = file_obj.filename.rsplit('.', 1)[1].lower() if '.' in file_obj.filename else ''
    
    # SPECIAL CASE: Always allow Company ID regardless of type
    # We'll handle this in the calling function
    
    # Check MIME type
    file_mime = file_obj.mimetype.lower()
    if file_mime in allowed_types:
        return True
    
    # Check file extension as fallback
    ext_to_mime = {
        'jpg': 'image/jpeg',
        'jpeg': 'image/jpeg',
        'png': 'image/png',
        'pdf': 'application/pdf',
        'txt': 'text/plain',
        'doc': 'application/msword',
        'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
    }
    
    if file_ext in ext_to_mime and ext_to_mime[file_ext] in allowed_types:
        return True
    
    return False

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
    """Main verification endpoint with enhanced validation"""
    try:
        # Get form data
        id_type = request.form.get('id_type', '')
        if not id_type:
            return jsonify({'success': False, 'message': 'ID type is required'}), 400
        
        # Check all required files
        required_files = ['id_file', 'selfie_file', 'payslip_file', 'company_id_file']
        missing_files = []
        
        for file_field in required_files:
            if file_field not in request.files:
                missing_files.append(file_field.replace('_file', '').replace('_', ' '))
            elif request.files[file_field].filename == '':
                missing_files.append(file_field.replace('_file', '').replace('_', ' '))
        
        if missing_files:
            return jsonify({
                'success': False, 
                'message': f'Missing required files: {", ".join(missing_files)}'
            }), 400
        
        # Get all files
        id_file = request.files['id_file']
        selfie_file = request.files['selfie_file']
        payslip_file = request.files['payslip_file']
        company_id_file = request.files['company_id_file']
        
        # Validate file types (EXCEPT FOR COMPANY ID)
        files_to_validate = [
            ('Primary ID', id_file, ['image/jpeg', 'image/png', 'image/jpg', 'application/pdf']),
            ('Selfie', selfie_file, ['image/jpeg', 'image/png', 'image/jpg']),
            ('Payslip', payslip_file, ['image/jpeg', 'image/png', 'image/jpg', 'application/pdf']),
            # Company ID - NO VALIDATION, accept any file type
        ]
        
        invalid_files = []
        for file_name, file_obj, allowed_types in files_to_validate:
            if file_name != 'Company ID' and not allowed_file_by_type(file_obj, allowed_types):
                invalid_files.append(f'{file_name} (invalid file type)')
        
        if invalid_files:
            return jsonify({
                'success': False, 
                'message': f'Invalid file types: {", ".join(invalid_files)}'
            }), 400
        
        # Check file sizes (EXCEPT FOR COMPANY ID)
        files_to_check_size = [
            ('Primary ID', id_file),
            ('Selfie', selfie_file),
            ('Payslip', payslip_file),
            # Company ID - NO SIZE LIMIT
        ]
        
        oversized_files = []
        for file_name, file_obj in files_to_check_size:
            # Get file size by seeking to end
            file_obj.seek(0, 2)  # Seek to end of file
            file_size = file_obj.tell()
            file_obj.seek(0)  # Reset file pointer
            
            if file_size > MAX_FILE_SIZE:
                oversized_files.append(f'{file_name} ({file_size/1024/1024:.1f}MB > 5MB)')
        
        if oversized_files:
            return jsonify({
                'success': False, 
                'message': f'Files exceed 5MB limit: {", ".join(oversized_files)}'
            }), 400
        
        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save files temporarily
            id_path = os.path.join(temp_dir, secure_filename(id_file.filename))
            selfie_path = os.path.join(temp_dir, secure_filename(selfie_file.filename))
            payslip_path = os.path.join(temp_dir, secure_filename(payslip_file.filename))
            company_id_path = os.path.join(temp_dir, secure_filename(company_id_file.filename))
            
            id_file.save(id_path)
            selfie_file.save(selfie_path)
            payslip_file.save(payslip_path)
            company_id_file.save(company_id_path)
            
            # Validate document contents
            validation_results = {
                'id': validate_id_document(id_path, id_type),
                'selfie': validate_selfie_document(selfie_path),
                'payslip': validate_payslip_document(payslip_path),
                'company_id': validate_company_id_document(company_id_path)  # SIMPLIFIED
            }
            
            # Check if any validation failed (except company_id)
            validation_errors = []
            for doc_type, result in validation_results.items():
                if doc_type != 'company_id' and not result['is_valid']:
                    doc_name = doc_type.replace('_', ' ').title()
                    validation_errors.append(f"{doc_name}: {result['message']}")
            
            if validation_errors:
                return jsonify({
                    'success': False,
                    'message': 'Document validation failed',
                    'validation_errors': validation_errors,
                    'validation_details': validation_results
                }), 400
            
            # Verify Primary ID
            id_verification = verify_primary_id(id_path, id_type, selfie_path)
            
            # Prepare response
            response = {
                'success': True,
                'timestamp': datetime.now().isoformat(),
                'id_type': id_type,
                'id_verification': id_verification,
                'validation_summary': {
                    'all_documents_valid': True,
                    'details': validation_results
                },
                'message': 'All documents validated and verification completed successfully'
            }
            
            return jsonify(response)
            
    except Exception as e:
        print(f"Verification error: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Internal server error: {str(e)}'
        }), 500

def validate_id_document(file_path, id_type):
    """Validate ID document content"""
    result = {
        'is_valid': False,
        'message': '',
        'keywords_found': [],
        'text_extracted': False,
        'file_type': os.path.splitext(file_path)[1].lower()
    }
    
    try:
        # Extract text from document
        text_data = extract_and_validate_text(file_path, id_type)
        
        if not text_data['has_text']:
            result['message'] = 'No readable text found in ID document'
            return result
        
        result['text_extracted'] = True
        result['keywords_found'] = text_data['keywords_found']
        
        # ID type specific validation
        if id_type == 'passport':
            required_keywords = ['passport', 'government', 'republic', 'international', 'visa']
            if any(keyword in text_data['extracted_text'].lower() for keyword in required_keywords):
                result['is_valid'] = True
                result['message'] = 'Valid passport document detected'
            else:
                result['message'] = 'Document does not appear to be a valid passport'
        
        elif id_type == 'driver_license':
            required_keywords = ['driver', 'license', 'licence', 'permit', 'dl', 'driving']
            if any(keyword in text_data['extracted_text'].lower() for keyword in required_keywords):
                result['is_valid'] = True
                result['message'] = 'Valid driver license document detected'
            else:
                result['message'] = 'Document does not appear to be a valid driver license'
        
        elif id_type == 'national_id':
            required_keywords = ['national', 'identity', 'id', 'card', 'identification', 'citizen']
            if any(keyword in text_data['extracted_text'].lower() for keyword in required_keywords):
                result['is_valid'] = True
                result['message'] = 'Valid national ID document detected'
            else:
                result['message'] = 'Document does not appear to be a valid national ID'
        
        if not result['is_valid'] and result['message'] == '':
            result['message'] = f'Document does not appear to be a valid {id_type.replace("_", " ")}'
            
    except Exception as e:
        print(f"ID validation error: {str(e)}")
        result['message'] = f'Error validating ID document: {str(e)}'
    
    return result

def validate_selfie_document(file_path):
    """Validate selfie document"""
    result = {
        'is_valid': False,
        'message': '',
        'file_type': os.path.splitext(file_path)[1].lower()
    }
    
    try:
        # Check if it's an image file
        if not file_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            result['message'] = 'Selfie must be an image file (JPG, PNG)'
            return result
        
        # Try to open and check the image
        img = cv2.imread(file_path)
        if img is None:
            result['message'] = 'Cannot read image file. File may be corrupted.'
            return result
        
        # Check image dimensions
        height, width = img.shape[:2]
        if height < 100 or width < 100:
            result['message'] = f'Image too small ({width}x{height}). Please upload a clearer image.'
            return result
        
        # Check for faces (basic validation)
        faces = detect_faces(img)
        if len(faces) == 0:
            result['message'] = 'No face detected in selfie. Please ensure your face is clearly visible.'
            return result
        
        result['is_valid'] = True
        result['message'] = f'Valid selfie detected ({len(faces)} face(s) found)'
        result['image_dimensions'] = f'{width}x{height}'
        
    except Exception as e:
        print(f"Selfie validation error: {str(e)}")
        result['message'] = f'Error validating selfie: {str(e)}'
    
    return result

def validate_payslip_document(file_path):
    """Validate payslip document content"""
    result = {
        'is_valid': False,
        'message': '',
        'keywords_found': [],
        'file_type': os.path.splitext(file_path)[1].lower(),
        'validation_score': 0
    }
    
    try:
        # Extract text from document
        text = ''
        if file_path.lower().endswith('.pdf'):
            try:
                images = convert_from_path(file_path)
                if images:
                    text = pytesseract.image_to_string(images[0])
                    result['page_count'] = len(images)
                else:
                    result['message'] = 'Could not extract text from PDF'
                    return result
            except Exception as e:
                result['message'] = f'Error processing PDF: {str(e)}'
                return result
        else:
            try:
                image = Image.open(file_path)
                enhancer = ImageEnhance.Contrast(image)
                enhanced = enhancer.enhance(2.0)
                enhancer = ImageEnhance.Sharpness(enhanced)
                enhanced = enhancer.enhance(2.0)
                text = pytesseract.image_to_string(enhanced)
            except Exception as e:
                result['message'] = f'Error processing image: {str(e)}'
                return result
        
        text_lower = text.lower()
        
        # Look for payslip keywords
        payslip_keywords = [
            'payslip', 'salary', 'employee', 'wage', 'net', 'gross',
            'period', 'payroll', 'earning', 'deduction', 'tax',
            'basic', 'allowance', 'overtime', 'bonus', 'commission'
        ]
        
        found_keywords = [kw for kw in payslip_keywords if kw in text_lower]
        result['keywords_found'] = found_keywords
        
        # Check for amount patterns
        amount_patterns = [
            r'\$\d{1,3}(?:,\d{3})*(?:\.\d{2})?',  # $1,000.00
            r'\d{1,3}(?:,\d{3})*(?:\.\d{2})?\s*(?:USD|EUR|GBP)?',  # 1,000.00
            r'\b\d+(?:\.\d{2})?\b'  # Any number with two decimals
        ]
        
        amount_matches = []
        for pattern in amount_patterns:
            matches = re.findall(pattern, text)
            amount_matches.extend(matches)
        
        result['amount_patterns_found'] = len(amount_matches)
        
        # Calculate validation score
        score = (len(found_keywords) * 5) + (len(amount_matches) * 3)
        result['validation_score'] = score
        
        # Determine validation result
        if score >= 20:  # High confidence
            result['is_valid'] = True
            result['message'] = f'Valid payslip detected ({len(found_keywords)} keywords, {len(amount_matches)} amounts)'
            result['confidence'] = 'high'
        elif score >= 10:  # Medium confidence
            result['is_valid'] = True
            result['message'] = f'Likely payslip detected ({len(found_keywords)} keywords, {len(amount_matches)} amounts)'
            result['confidence'] = 'medium'
            result['warning'] = 'Limited payslip features detected'
        elif text.strip() and len(amount_matches) > 0:  # Low confidence with amounts
            result['is_valid'] = True
            result['message'] = f'Possible payslip ({len(amount_matches)} amounts found)'
            result['confidence'] = 'low'
            result['warning'] = 'Very few payslip keywords but amounts detected'
        elif text.strip():
            result['message'] = 'Document may not be a payslip (insufficient payslip keywords or amounts)'
        else:
            result['message'] = 'No readable text found in document'
            
    except Exception as e:
        print(f"Payslip validation error: {str(e)}")
        result['message'] = f'Error validating payslip: {str(e)}'
    
    return result

def validate_company_id_document(file_path):
    """Validate company ID document content - ALWAYS ACCEPT"""
    result = {
        'is_valid': True,  # ALWAYS VALID
        'message': 'Company ID uploaded successfully',
        'keywords_found': [],
        'file_type': os.path.splitext(file_path)[1].lower(),
        'note': 'Unrestricted validation - accepts any file type'
    }
    
    try:
        # Get file info
        file_size = os.path.getsize(file_path)
        result['file_size'] = f"{file_size} bytes"
        
        # Try to extract text if it's a text-based file
        if file_path.lower().endswith(('.jpg', '.jpeg', '.png', '.pdf', '.txt')):
            try:
                if file_path.lower().endswith('.pdf'):
                    images = convert_from_path(file_path)
                    if images:
                        text = pytesseract.image_to_string(images[0])
                    else:
                        text = ''
                elif file_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image = Image.open(file_path)
                    enhancer = ImageEnhance.Contrast(image)
                    enhanced = enhancer.enhance(2.0)
                    enhancer = ImageEnhance.Sharpness(enhanced)
                    enhanced = enhancer.enhance(2.0)
                    text = pytesseract.image_to_string(enhanced)
                elif file_path.lower().endswith('.txt'):
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        text = f.read()
                else:
                    text = ''
                
                text_lower = text.lower()
                
                # Look for company ID keywords (info only)
                company_id_keywords = [
                    'company', 'corporate', 'employee', 'staff', 'personnel',
                    'id', 'identification', 'card', 'badge', 'member',
                    'employer', 'organization', 'business', 'enterprise'
                ]
                
                found_keywords = [kw for kw in company_id_keywords if kw in text_lower]
                result['keywords_found'] = found_keywords
                
                # Update message if keywords found
                if found_keywords:
                    result['message'] = f'Company ID detected ({len(found_keywords)} keywords found)'
                
            except Exception as e:
                # If extraction fails, still accept
                print(f"Company ID content extraction warning: {str(e)}")
                result['note'] = 'File accepted (content extraction skipped)'
        
        else:
            # For other file types, just accept them
            result['note'] = 'Non-text file accepted'
            
    except Exception as e:
        # Even on major error, still accept
        print(f"Company ID validation warning: {str(e)}")
        result['note'] = f'Accepted with warning: {str(e)}'
    
    return result

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
            except Exception as e:
                print(f"PDF conversion error: {e}")
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
        verification['analysis']['keywords_found'] = len(text_data['keywords_found'])
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
        
        # 7. Color Consistency Check
        color_score = check_color_consistency(id_img)
        verification['analysis']['color_consistency'] = f"{color_score}/100"
        checks.append(('Color Consistency', color_score > 60, color_score))
        
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
            'Basic Tampering Detection',
            'Color Consistency Check'
        ]
        
        # Log issues if confidence is low
        if confidence < 60:
            verification['issues'].append(f'Low confidence score: {confidence}%')
            if not text_data['has_text']:
                verification['issues'].append('No readable text found in ID')
            if tampering_score > 50:
                verification['issues'].append('High tampering risk detected')
            if quality_score < 50:
                verification['issues'].append('Poor image quality')
        
        # Clean up temporary image file
        if id_path.lower().endswith('.pdf') and os.path.exists(id_image_path):
            try:
                os.remove(id_image_path)
            except:
                pass
                
    except Exception as e:
        print(f"Verification error: {str(e)}")
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
        
        # Calculate noise level
        noise = cv2.Laplacian(gray, cv2.CV_64F).std()
        
        # Combined quality score (0-100)
        sharpness_score = min(100, laplacian_var / 10)
        contrast_score = min(100, contrast / 2)
        brightness_score = 100 - abs(brightness - 127) / 127 * 100
        noise_score = max(0, 100 - noise * 10)
        
        quality_score = (sharpness_score * 0.3 + contrast_score * 0.3 + 
                        brightness_score * 0.2 + noise_score * 0.2)
        
        return round(max(0, min(100, quality_score)))
    except Exception as e:
        print(f"Image quality check error: {e}")
        return 50

def detect_faces(image):
    """Detect faces in an image using OpenCV"""
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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
            except Exception as e:
                print(f"PDF OCR error: {e}")
                return result
        else:
            try:
                image = Image.open(image_path)
                enhancer = ImageEnhance.Contrast(image)
                enhanced = enhancer.enhance(2.0)
                enhancer = ImageEnhance.Sharpness(enhanced)
                enhanced = enhancer.enhance(2.0)
                text = pytesseract.image_to_string(enhanced)
            except Exception as e:
                print(f"Image OCR error: {e}")
                return result
        
        result['extracted_text'] = text
        result['has_text'] = len(text.strip()) > 0
        
        # Look for ID-specific patterns
        text_lower = text.lower()
        keywords = []
        
        if id_type == 'passport':
            keywords = ['passport', 'republic', 'government', 'number', 'date', 'expiry', 'international']
            if re.search(r'\b[A-Z]{1,2}\d{6,8}\b', text.upper()):
                result['pattern_match'] = True
        
        elif id_type == 'driver_license':
            keywords = ['driver', 'license', 'licence', 'dl', 'permit', 'expires', 'expiry', 'driving']
            if re.search(r'\b[A-Z]{1,2}\d{6,9}\b', text.upper()):
                result['pattern_match'] = True
        
        elif id_type == 'national_id':
            keywords = ['national', 'identity', 'id', 'card', 'number', 'republic', 'citizen']
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
        
    except Exception as e:
        print(f"Tampering check error: {e}")
        return 50

def check_color_consistency(image):
    """Check color consistency across the image"""
    try:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_std = np.std(lab[:,:,0])
        a_std = np.std(lab[:,:,1])
        b_std = np.std(lab[:,:,2])
        avg_std = (l_std + a_std + b_std) / 3
        score = max(0, 100 - avg_std * 2)
        return round(score)
    except Exception as e:
        print(f"Color consistency check error: {e}")
        return 50

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'service': 'Document Verification API',
        'version': '2.2.0',
        'features': [
            'Strict ID validation',
            'Strict selfie validation',
            'Strict payslip validation',
            'UNRESTRICTED Company ID validation',
            'Face detection',
            'OCR text extraction'
        ]
    })

@app.route('/test', methods=['GET'])
def test_endpoint():
    """Test endpoint"""
    return jsonify({
        'message': 'Document Verification API is running',
        'timestamp': datetime.now().isoformat(),
        'endpoints': {
            'GET /': 'Frontend',
            'POST /verify': 'Verify documents',
            'GET /health': 'Health check',
            'GET /test': 'Test endpoint'
        }
    })

if __name__ == '__main__':
    print("=" * 60)
    print("Document Verification System v2.2")
    print("=" * 60)
    print("Features:")
    print("✓ Strict Primary ID validation")
    print("✓ Strict Selfie validation (face detection)")
    print("✓ Strict Payslip validation (keywords & amounts)")
    print("✓ UNRESTRICTED Company ID validation (accepts ANY file)")
    print("✓ OCR text extraction")
    print("✓ Face detection")
    print("=" * 60)
    
    if os.path.exists('index.html'):
        print("✓ Found index.html")
    else:
        print("✗ Missing index.html")
    
    print("\nChecking dependencies...")
    
    dependencies = [
        ('OpenCV', 'cv2'),
        ('Tesseract OCR', 'pytesseract'),
        ('PyPDF2', 'PyPDF2'),
        ('pdf2image', 'pdf2image'),
        ('NumPy', 'numpy'),
        ('Pillow', 'PIL')
    ]
    
    all_deps_ok = True
    for dep_name, dep_module in dependencies:
        try:
            __import__(dep_module.split('.')[0])
            print(f"✓ {dep_name} installed")
        except ImportError:
            print(f"✗ {dep_name} not installed")
            all_deps_ok = False
    
    if not all_deps_ok:
        print("\nMissing dependencies. Install with:")
        print("pip install flask flask-cors opencv-python pillow pytesseract PyPDF2 pdf2image numpy")
    
    print("\nStarting server...")
    print("Open: http://localhost:5000")
    print("API Endpoints:")
    print("  GET  /                 - Frontend application")
    print("  POST /verify           - Verify all documents")
    print("  GET  /health          - Health check")
    print("  GET  /test            - Test endpoint")
    print("\nPress Ctrl+C to stop\n")
    
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)