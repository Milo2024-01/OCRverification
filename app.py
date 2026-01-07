from flask import Flask, request, jsonify, render_template, redirect, make_response, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import tempfile
from datetime import datetime, timedelta
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import pytesseract
from pdf2image import convert_from_path
import re
import json
import hashlib
import secrets
import io
import shutil
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as ReportLabImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
import difflib
import sqlite3

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
REFERENCE_BASE_FOLDER = 'references'
REPORTS_FOLDER = 'reports'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf'}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

# Supported document types
DOCUMENT_TYPES = ['drivers_license', 'national_id', 'passport']
DEFAULT_DOC_TYPE = 'drivers_license'

# Secret key for sessions (in production, use a secure random key)
app.secret_key = secrets.token_hex(32)

# ============= SESSION MANAGEMENT =============
active_sessions = {}
SESSION_TIMEOUT = timedelta(hours=1)

# Login credentials (in production, use a database with hashed passwords)
VALID_CREDENTIALS = {
    'username': 'admin',
    'password': 'jethro123'
}

def generate_session_token(username):
    """Generate a secure session token"""
    token = secrets.token_urlsafe(32)
    active_sessions[token] = {
        'username': username,
        'login_time': datetime.now(),
        'last_activity': datetime.now()
    }
    return token

def verify_session(request):
    """Verify if user has a valid session"""
    token = request.cookies.get('session_token')
    
    if token and token in active_sessions:
        session_data = active_sessions[token]
        
        # Check session timeout
        if datetime.now() - session_data['last_activity'] > SESSION_TIMEOUT:
            del active_sessions[token]
            return False
        
        # Update last activity
        session_data['last_activity'] = datetime.now()
        return True
    
    return False

def cleanup_expired_sessions():
    """Clean up expired sessions"""
    current_time = datetime.now()
    expired_tokens = []
    
    for token, session_data in active_sessions.items():
        if current_time - session_data['last_activity'] > SESSION_TIMEOUT:
            expired_tokens.append(token)
    
    for token in expired_tokens:
        del active_sessions[token]

# ============= HELPER FUNCTIONS =============
def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_reference_folder(document_type):
    """Get the reference folder for a specific document type"""
    if document_type not in DOCUMENT_TYPES:
        return os.path.join(REFERENCE_BASE_FOLDER, DEFAULT_DOC_TYPE)
    return os.path.join(REFERENCE_BASE_FOLDER, document_type)

# Create necessary directories
for folder in [UPLOAD_FOLDER, REPORTS_FOLDER, 'static', 'templates']:
    os.makedirs(folder, exist_ok=True)

# Path for SQLite DB
DB_PATH = os.path.join(REPORTS_FOLDER, 'loan_applications.db')

def init_db():
    """Initialize SQLite database and create tables if missing."""
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS loan_applications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                full_name TEXT,
                contact TEXT,
                amount REAL,
                months INTEGER,
                interest_rate REAL,
                monthly_payment REAL,
                verification TEXT
            )
        ''')
        conn.commit()
        conn.close()
    except Exception as e:
        print('Failed to initialize database:', e)

# initialize DB
init_db()

# Create reference folders for each document type
for doc_type in DOCUMENT_TYPES:
    folder_path = os.path.join(REFERENCE_BASE_FOLDER, doc_type)
    os.makedirs(folder_path, exist_ok=True)

# Set Tesseract path
try:
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
except:
    try:
        pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
    except:
        pass

# ============= GLOBAL REFERENCE STORAGE =============
reference_data = {
    'drivers_license': {'image': None, 'features': None},
    'national_id': {'image': None, 'features': None},
    'passport': {'image': None, 'features': None}
}

# ============= LOGIN & AUTHENTICATION ROUTES =============
@app.route('/')
def index():
    """Serve the main HTML page (protected)"""
    if not verify_session(request):
        return redirect('/login')
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Handle user login"""
    if request.method == 'GET':
        # Check if already logged in
        if verify_session(request):
            return redirect('/')
        return render_template('login.html')
    
    elif request.method == 'POST':
        data = request.get_json()
        username = data.get('username', '').strip()
        password = data.get('password', '')
        
        if username == VALID_CREDENTIALS['username'] and password == VALID_CREDENTIALS['password']:
            # Create session
            session_token = generate_session_token(username)
            
            response = jsonify({
                'success': True,
                'message': 'Login successful',
                'redirect': '/dashboard'
            })
            
            # Set secure cookie
            response.set_cookie(
                'session_token',
                session_token,
                httponly=True,
                secure=False,  # Set to True in production with HTTPS
                samesite='Strict',
                max_age=3600  # 1 hour
            )
            
            # Cleanup expired sessions
            cleanup_expired_sessions()
            
            return response
        else:
            return jsonify({
                'success': False,
                'message': 'Invalid credentials'
            }), 401

@app.route('/logout', methods=['POST'])
def logout():
    """Handle user logout"""
    token = request.cookies.get('session_token')
    if token and token in active_sessions:
        del active_sessions[token]

    # If client expects JSON, return JSON. If it's a browser form, redirect to login.
    is_json = request.is_json or request.headers.get('Accept', '').find('application/json') != -1
    if is_json:
        response = jsonify({'success': True, 'message': 'Logged out successfully'})
    else:
        response = redirect('/login')

    response.set_cookie('session_token', '', expires=0)
    return response

@app.route('/check-auth', methods=['GET'])
def check_auth():
    """Check if user is authenticated"""
    if verify_session(request):
        return jsonify({'authenticated': True, 'username': 'admin'})
    return jsonify({'authenticated': False}), 401

@app.route('/session-info', methods=['GET'])
def session_info():
    """Get session information"""
    if verify_session(request):
        token = request.cookies.get('session_token')
        if token in active_sessions:
            session_data = active_sessions[token]
            time_remaining = SESSION_TIMEOUT - (datetime.now() - session_data['last_activity'])
            
            return jsonify({
                'authenticated': True,
                'username': session_data['username'],
                'login_time': session_data['login_time'].isoformat(),
                'session_timeout_minutes': int(time_remaining.total_seconds() / 60)
            })
    
    return jsonify({'authenticated': False}), 401

# ============= FILE PROCESSING FUNCTIONS =============
def load_reference_license(document_type='drivers_license'):
    """Load reference for specific document type"""
    try:
        folder_path = get_reference_folder(document_type)
        reference_files = [f for f in os.listdir(folder_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.pdf'))]
        
        if not reference_files:
            print(f"No reference found for {document_type}")
            return False
        
        # Use the first reference file found
        reference_path = os.path.join(folder_path, reference_files[0])
        print(f"Loading {document_type} reference from: {reference_path}")
        
        # Load reference image
        if reference_path.lower().endswith('.pdf'):
            images = convert_from_path(reference_path)
            if images:
                temp_img_path = os.path.join(tempfile.gettempdir(), 'temp_reference.png')
                images[0].save(temp_img_path, 'PNG')
                reference_data[document_type]['image'] = cv2.imread(temp_img_path)
                # Clean up temp file
                if os.path.exists(temp_img_path):
                    os.remove(temp_img_path)
            else:
                print(f"Could not process PDF reference for {document_type}")
                return False
        else:
            reference_data[document_type]['image'] = cv2.imread(reference_path)
        
        if reference_data[document_type]['image'] is None:
            print(f"Could not read {document_type} reference image")
            return False
        
        # Extract features from reference
        reference_text = extract_text_from_file(reference_path)
        reference_data[document_type]['features'] = extract_detailed_features(
            reference_data[document_type]['image'], 
            reference_text,
            document_type
        )
        
        print(f"{document_type} reference loaded successfully. Size: {reference_data[document_type]['image'].shape}")
        return True
        
    except Exception as e:
        print(f"Error loading {document_type} reference: {e}")
        return False

def extract_text_from_file(file_path):
    """Extract text from image or PDF file"""
    try:
        if file_path.lower().endswith('.pdf'):
            images = convert_from_path(file_path)
            if images:
                text = pytesseract.image_to_string(images[0])
                return text
        else:
            image = Image.open(file_path)
            # Enhance image for better OCR
            enhancer = ImageEnhance.Contrast(image)
            enhanced = enhancer.enhance(1.5)
            text = pytesseract.image_to_string(enhanced)
            return text
    except Exception as e:
        print(f"Text extraction error: {e}")
        return ""

def extract_detailed_features(img, text, document_type='drivers_license'):
    """Extract detailed features from image and text"""
    features = {}
    
    try:
        # 1. Image features (common for all document types)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Basic image properties
        features['height'] = int(img.shape[0])
        features['width'] = int(img.shape[1])
        features['aspect_ratio'] = float(img.shape[1] / img.shape[0])
        
        # Color features
        features['color_mean_b'] = float(np.mean(img[:,:,0]))
        features['color_mean_g'] = float(np.mean(img[:,:,1]))
        features['color_mean_r'] = float(np.mean(img[:,:,2]))
        
        # Texture features
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        features['sharpness'] = float(laplacian_var) if not np.isnan(laplacian_var) else 0.0
        features['contrast'] = float(gray.std()) if not np.isnan(gray.std()) else 0.0
        features['brightness'] = float(np.mean(gray)) if not np.isnan(np.mean(gray)) else 0.0
        
        # 2. Text features
        if text:
            text_lower = text.lower()
            features['text_length'] = int(len(text))
            features['text'] = str(text)
            
            # Document type specific patterns
            if document_type == 'drivers_license':
                patterns = {
                    'license_number': r'\b[A-Z]{1,2}\d{6,9}\b',
                    'dates': r'\b\d{1,2}[-/]\d{1,2}[-/]\d{4}\b',
                    'zip_code': r'\b\d{5}(?:-\d{4})?\b',
                }
            elif document_type == 'national_id':
                patterns = {
                    'id_number': r'\b\d{8,12}\b',
                    'dates': r'\b\d{1,2}[-/]\d{1,2}[-/]\d{4}\b',
                    'national_id_keywords': r'\b(national|id|identification|republic|philippines|ph)\b',
                }
            elif document_type == 'passport':
                patterns = {
                    'passport_number': r'\b[A-Z]{1,2}\d{6,9}\b',
                    'dates': r'\b\d{1,2}[-/]\d{1,2}[-/]\d{4}\b',
                    'passport_keywords': r'\b(passport|p[.]?no|republic|philippines|ph|diplomatic|official|ordinary|type)\b',
                    'mrz_code': r'\b[A-Z0-9<]{9,50}\b',  # Machine Readable Zone patterns
                }
            else:
                patterns = {}
            
            for name, pattern in patterns.items():
                matches = re.findall(pattern, text, re.IGNORECASE)
                features[f'{name}_count'] = int(len(matches))
        
        return features
        
    except Exception as e:
        print(f"Feature extraction error: {e}")
        return features

def compare_with_reference(img, text, document_type='drivers_license'):
    """Compare uploaded document with reference document"""
    comparison_results = {
        'similarity_score': 0.0,
        'differences': [],
        'details': {}
    }
    
    try:
        ref_data = reference_data[document_type]
        if ref_data['image'] is None or ref_data['features'] is None:
            if not load_reference_license(document_type):
                comparison_results['differences'].append(f'No {document_type} reference available for comparison')
                return comparison_results
        
        # Extract features from uploaded image
        uploaded_features = extract_detailed_features(img, text, document_type)
        
        # 1. Compare image properties
        img_similarities = []
        ref_features = ref_data['features']
        
        # Size comparison (more lenient - 50% difference threshold)
        if 'height' in ref_features and 'height' in uploaded_features:
            height_diff = abs(ref_features['height'] - uploaded_features['height']) / max(ref_features['height'], 1) * 100
            if height_diff > 50:  # Increased from 30%
                comparison_results['differences'].append(f'Height difference: {height_diff:.1f}%')
            img_similarities.append(max(0.0, 100.0 - min(height_diff, 100.0)))
        
        # Aspect ratio comparison (more lenient - 30% difference threshold)
        if 'aspect_ratio' in ref_features and 'aspect_ratio' in uploaded_features:
            ref_ratio = ref_features['aspect_ratio']
            upload_ratio = uploaded_features['aspect_ratio']
            ratio_diff = abs(ref_ratio - upload_ratio) / max(ref_ratio, 0.01) * 100
            if ratio_diff > 30:  # Increased from 20%
                comparison_results['differences'].append(f'Aspect ratio difference: {ratio_diff:.1f}%')
            img_similarities.append(max(0.0, 100.0 - min(ratio_diff, 100.0)))
        
        # Color comparison (more lenient - 60% difference threshold)
        color_similarity = 50.0  # Default
        if all(k in ref_features for k in ['color_mean_b', 'color_mean_g', 'color_mean_r']) and \
           all(k in uploaded_features for k in ['color_mean_b', 'color_mean_g', 'color_mean_r']):
            
            ref_color = np.array([ref_features['color_mean_b'], 
                                  ref_features['color_mean_g'], 
                                  ref_features['color_mean_r']])
            upload_color = np.array([uploaded_features['color_mean_b'], 
                                     uploaded_features['color_mean_g'], 
                                     uploaded_features['color_mean_r']])
            
            color_diff = np.mean(np.abs(ref_color - upload_color))
            color_similarity = max(0.0, 100.0 - color_diff)
            if color_diff > 60:  # Increased from 40%
                comparison_results['differences'].append('Significant color difference detected')
        
        img_similarities.append(color_similarity)
        
        # Sharpness comparison (more lenient)
        if 'sharpness' in ref_features and 'sharpness' in uploaded_features:
            ref_sharp = ref_features['sharpness']
            upload_sharp = uploaded_features['sharpness']
            if ref_sharp > 0 and upload_sharp > 0:
                sharpness_diff = abs(ref_sharp - upload_sharp) / max(ref_sharp, upload_sharp) * 100
                if sharpness_diff > 80:  # Increased from 60%
                    comparison_results['differences'].append(f'Sharpness difference: {sharpness_diff:.1f}%')
                img_similarities.append(max(0.0, 100.0 - min(sharpness_diff, 100.0)))
        
        # 2. Compare text
        text_similarities = []
        
        if 'text' in ref_features and 'text' in uploaded_features:
            ref_text = ref_features['text'].lower()
            upload_text = uploaded_features['text'].lower()
            
            # Document type specific keywords (expanded lists)
            if document_type == 'drivers_license':
                common_keywords = ['driver', 'license', 'licence', 'state', 'expires', 'expiration',
                                  'birth', 'dob', 'date of birth', 'issued', 'height', 'weight', 'driving',
                                  'class', 'restriction', 'endorsement', 'address']
            elif document_type == 'national_id':
                common_keywords = ['national', 'id', 'identification', 'republic', 'philippines',
                                  'birth', 'dob', 'date of birth', 'address', 'sex', 'gender',
                                  'civil status', 'blood type', 'signature', 'citizen', 'phil']
            elif document_type == 'passport':
                common_keywords = ['passport', 'republic', 'philippines', 'ph', 'type', 
                                  'country code', 'surname', 'given names', 'nationality',
                                  'date of birth', 'sex', 'place of birth', 'date of issue',
                                  'authority', 'date of expiry', 'mrz', 'machine readable', 'travel']
            else:
                common_keywords = []
            
            # More flexible keyword matching
            ref_keywords = []
            upload_keywords = []
            
            for kw in common_keywords:
                if kw in ref_text:
                    ref_keywords.append(kw)
                # Check for partial matches in uploaded text
                for word in upload_text.split():
                    if kw in word or word in kw:
                        if len(kw) > 3 and len(word) > 3:
                            upload_keywords.append(kw)
                            break
            
            if ref_keywords:
                # Calculate intersection (allowing partial matches)
                intersection = set(ref_keywords) & set(upload_keywords)
                keyword_similarity = len(intersection) / max(len(ref_keywords), 1) * 100
                text_similarities.append(keyword_similarity)
                
                if keyword_similarity < 40:  # Reduced from 60%
                    comparison_results['differences'].append(f'Missing important {document_type} keywords')
            
            # Text length comparison (more lenient)
            if 'text_length' in ref_features and 'text_length' in uploaded_features:
                ref_len = ref_features['text_length']
                upload_len = uploaded_features['text_length']
                if ref_len > 0 and upload_len > 0:
                    length_similarity = min(ref_len, upload_len) / max(ref_len, upload_len) * 100
                    text_similarities.append(length_similarity)
                    if length_similarity < 40:  # Reduced from 50%
                        comparison_results['differences'].append(f'Text length significantly different')
            
            # Add simple text overlap score
            if len(ref_text) > 10 and len(upload_text) > 10:
                # Calculate word overlap
                ref_words = set(ref_text.split())
                upload_words = set(upload_text.split())
                overlap = len(ref_words & upload_words) / max(len(ref_words), 1) * 100
                text_similarities.append(min(overlap, 100.0))
        
        # 3. Calculate overall similarity score with better weighting
        if img_similarities:
            # Give more weight to color and aspect ratio
            weights = [0.2, 0.3, 0.3, 0.2]  # height, aspect, color, sharpness
            if len(weights) == len(img_similarities):
                img_score = float(np.average(img_similarities, weights=weights[:len(img_similarities)]))
            else:
                img_score = float(np.mean(img_similarities))
        else:
            img_score = 50.0
        
        if text_similarities:
            text_score = float(np.mean(text_similarities))
        else:
            text_score = 50.0
        
        # Weighted average (50% image, 50% text) - more balanced
        overall_similarity = img_score * 0.5 + text_score * 0.5
        comparison_results['similarity_score'] = float(overall_similarity)
        
        # Add detailed comparison
        comparison_results['details'] = {
            'image_similarity': f"{img_score:.1f}%",
            'text_similarity': f"{text_score:.1f}%" if text_similarities else "N/A",
            'overall_similarity': f"{overall_similarity:.1f}%",
            'document_type': document_type.replace('_', ' ').title()
        }
        
        return comparison_results
        
    except Exception as e:
        print(f"Comparison error for {document_type}: {e}")
        comparison_results['differences'].append(f'Comparison error: {str(e)}')
        return comparison_results

def analyze_with_comparison(file_path, document_type='drivers_license'):
    """Main analysis function using reference comparison"""
    
    result = {
        'is_authentic': False,
        'confidence': 0.0,
        'similarity_score': 0.0,
        'issues': [],
        'analysis': {},
        'method': 'reference_comparison',
        'comparison_details': {},
        'has_reference': False,
        'document_type': document_type
    }
    
    try:
        # Load image
        if file_path.lower().endswith('.pdf'):
            images = convert_from_path(file_path)
            if images:
                temp_img_path = os.path.join(tempfile.gettempdir(), 'temp_image.png')
                images[0].save(temp_img_path, 'PNG')
                img = cv2.imread(temp_img_path)
                # Clean up temp file
                if os.path.exists(temp_img_path):
                    os.remove(temp_img_path)
            else:
                result['issues'].append('Could not process PDF file')
                return result
        else:
            img = cv2.imread(file_path)
        
        if img is None:
            result['issues'].append('Could not read image file')
            return result
        
        # Basic validation based on document type
        text = extract_text_from_file(file_path)
        text_lower = text.lower() if text else ""
        
        # Document type specific validation with more flexible keywords
        if document_type == 'drivers_license':
            keywords = ['driver', 'license', 'licence', 'permit', 'dl', 'driving', 'licensee', 'lic', 'drivers', 'identification']
            doc_name = "driver's license"
        elif document_type == 'national_id':
            keywords = ['national', 'id', 'identification', 'republic', 'philippines', 'ph', 'filipino', 'citizen', 'card']
            doc_name = "national ID"
        elif document_type == 'passport':
            keywords = ['passport', 'republic', 'philippines', 'ph', 'type', 'country', 'code', 'travel', 'document', 'book']
            doc_name = "passport"
        else:
            keywords = []
            doc_name = "document"
        
        # More flexible keyword matching (partial matches)
        keyword_count = 0
        for kw in keywords:
            if kw in text_lower:
                keyword_count += 1
            # Also check for similar words
            elif len(kw) > 4:
                # Check for partial matches (OCR errors might miss characters)
                words_in_text = text_lower.split()
                for word in words_in_text:
                    if len(word) > 3 and difflib.SequenceMatcher(None, kw, word).ratio() > 0.7:
                        keyword_count += 0.5  # Partial match
        
        # Check if we have a reference document
        result['has_reference'] = reference_data[document_type]['image'] is not None
        
        if result['has_reference']:
            # 1. Compare with reference document
            comparison = compare_with_reference(img, text, document_type)
            result['similarity_score'] = float(comparison['similarity_score'])
            result['comparison_details'] = comparison['details']
            
            # Filter out minor differences for the issues list
            significant_issues = []
            for diff in comparison['differences']:
                # Only include significant issues
                if 'difference' in diff.lower():
                    # Extract percentage from difference message
                    import re
                    perc_match = re.search(r'(\d+\.?\d*)%', diff)
                    if perc_match:
                        perc = float(perc_match.group(1))
                        if perc > 50:  # Only show differences > 50%
                            significant_issues.append(diff)
                    else:
                        significant_issues.append(diff)
                elif 'significant' in diff.lower() or 'missing' in diff.lower():
                    significant_issues.append(diff)
            
            result['issues'].extend(significant_issues)
            
            # 2. Run anomaly detection as secondary check
            features = extract_detailed_features(img, text, document_type)
            anomaly_score = 0.0
            
            # Check image quality (more lenient)
            if 'sharpness' in features:
                if features['sharpness'] < 20:
                    anomaly_score += 30.0
                    result['issues'].append('Very low image sharpness')
                elif features['sharpness'] < 40:
                    anomaly_score += 15.0
                    result['issues'].append('Low image sharpness')
                elif features['sharpness'] > 200:
                    anomaly_score += 10.0  # Too sharp might indicate digital manipulation
            
            if 'contrast' in features:
                if features['contrast'] < 15:
                    anomaly_score += 20.0
                    result['issues'].append('Very low contrast')
                elif features['contrast'] < 25:
                    anomaly_score += 8.0
                    # Don't add to issues for minor contrast problems
            
            # 3. Calculate overall confidence with more balanced weights
            similarity_confidence = float(comparison['similarity_score'])
            anomaly_adjustment = max(0.0, 100.0 - anomaly_score)
            
            # Calculate keyword presence score (0-100)
            keyword_score = min(100.0, (keyword_count / max(len(keywords) * 0.5, 1)) * 100)
            
            # Combined confidence with more weight on similarity
            # 50% similarity, 30% keywords, 20% anomaly adjustment
            if keyword_count >= 1:  # Reduced from 2
                final_confidence = (similarity_confidence * 0.5 + 
                                  keyword_score * 0.3 + 
                                  anomaly_adjustment * 0.2)
            else:
                # If no keywords at all, be more strict
                final_confidence = similarity_confidence * 0.3 + anomaly_adjustment * 0.2
            
            result['confidence'] = float(final_confidence)
            
            # 4. Determine authenticity with more reasonable thresholds
            if keyword_count < 0.5:  # Almost no keywords
                result['is_authentic'] = False
                result['issues'].append(f'Document lacks key {doc_name} identifiers')
            elif similarity_confidence >= 60.0 and final_confidence >= 55.0:
                result['is_authentic'] = True
                if similarity_confidence < 70.0:
                    result['issues'].append(f'Acceptable similarity to {doc_name} reference')
            elif similarity_confidence >= 50.0 and final_confidence >= 50.0 and keyword_count >= 2:
                result['is_authentic'] = True
                result['issues'].append(f'Moderate similarity to {doc_name} reference')
            else:
                result['is_authentic'] = False
            
            # 5. Detailed analysis
            result['analysis'] = {
                'similarity_to_reference': f"{comparison['similarity_score']:.1f}%",
                'image_quality': f"{features.get('sharpness', 0)/10:.1f}/10" if 'sharpness' in features else "N/A",
                'text_analysis': f"{len(text)} characters, {keyword_count:.1f} {doc_name} keywords found",
                'aspect_ratio': f"{features.get('aspect_ratio', 0):.2f}" if 'aspect_ratio' in features else "N/A",
                'document_type': doc_name,
                'keyword_score': f"{keyword_score:.1f}%"
            }
            
        else:
            # No reference available, use basic validation
            if keyword_count < 1:
                result['issues'].append(f'This does not appear to be a {doc_name} document')
                result['confidence'] = 10.0
                return result
            
            features = extract_detailed_features(img, text, document_type)
            
            # Simple quality scoring
            quality_score = 0.0
            
            if 'sharpness' in features:
                if features['sharpness'] > 40:
                    quality_score += 40.0
                elif features['sharpness'] > 20:
                    quality_score += 25.0
                else:
                    quality_score += 10.0
            
            if 'contrast' in features:
                if features['contrast'] > 25:
                    quality_score += 30.0
                elif features['contrast'] > 15:
                    quality_score += 20.0
                else:
                    quality_score += 10.0
            
            if 'aspect_ratio' in features:
                aspect = features['aspect_ratio']
                # Accept wider range for different document formats
                if 1.3 <= aspect <= 2.0:
                    quality_score += 30.0
                elif 1.2 <= aspect <= 2.2:
                    quality_score += 20.0
                else:
                    quality_score += 5.0
            
            # Add text score (more weight on keywords)
            text_score = min(100.0, (keyword_count / max(len(keywords) * 0.3, 1)) * 100)
            quality_score = quality_score * 0.4 + text_score * 0.6
            
            result['confidence'] = min(100.0, quality_score)
            
            # Without reference, be more lenient
            result['is_authentic'] = result['confidence'] >= 50.0 and keyword_count >= 1
            
            result['analysis'] = {
                'image_quality': f"{features.get('sharpness', 0)/10:.1f}/10" if 'sharpness' in features else "N/A",
                'confidence_score': f"{result['confidence']:.1f}%",
                'text_analysis': f"{len(text)} characters, {keyword_count:.1f} {doc_name} keywords found",
                'method': f'Basic Validation (No {doc_name} reference)',
                'keyword_score': f"{text_score:.1f}%"
            }
        
        # Only add "No significant issues" if confidence is good AND no other issues
        if not result['issues'] and result['confidence'] > 60.0:
            result['issues'].append('No significant issues detected')
        elif result['confidence'] > 70.0 and result['is_authentic']:
            # If authentic with high confidence but has minor issues, add positive note
            if len(result['issues']) <= 2:
                result['issues'].append('Minor variations detected, but document appears authentic')
        
    except Exception as e:
        result['issues'].append(f'Analysis error: {str(e)}')
        result['confidence'] = 0.0
        print(f"Analysis error: {e}")
    
    return result

# ============= REPORT GENERATION =============
@app.route('/generate-report', methods=['POST'])
def generate_report():
    """Generate and download PDF report of verification results"""
    if not verify_session(request):
        return jsonify({'success': False, 'message': 'Authentication required'}), 401
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'message': 'No data provided'}), 400
        
        # Create PDF in memory
        buffer = io.BytesIO()
        
        # Create the PDF document
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        
        # Container for the 'Flowable' objects
        story = []
        styles = getSampleStyleSheet()
        
        # Add custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#00f3ff'),
            spaceAfter=30,
            alignment=1  # Center aligned
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#2563eb'),
            spaceAfter=12
        )
        
        normal_style = ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            fontSize=10,
            spaceAfter=6
        )
        
        # Title
        story.append(Paragraph("CYBER-ID VERIFIER v3.1 - ANALYSIS REPORT", title_style))
        story.append(Spacer(1, 20))
        
        # Report Metadata
        report_id = hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8].upper()
        story.append(Paragraph(f"Report ID: {report_id}", normal_style))
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", normal_style))
        story.append(Paragraph(f"System: Multi-Document Verification System", normal_style))
        story.append(Spacer(1, 20))
        
        # Document Information
        story.append(Paragraph("DOCUMENT INFORMATION", heading_style))
        
        doc_type = data.get('document_type', 'Unknown').replace('_', ' ').title()
        doc_info = [
            ["Document Type:", doc_type],
            ["Analysis Date:", datetime.now().strftime('%Y-%m-%d')],
            ["Analysis Time:", datetime.now().strftime('%H:%M:%S')],
            ["Reference Used:", "Yes" if data.get('has_reference') else "No"],
            ["Method:", data.get('method', 'Reference Comparison')]
        ]
        
        doc_table = Table(doc_info, colWidths=[2*inch, 3*inch])
        doc_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f0f9ff')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#1e40af')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e5e7eb'))
        ]))
        story.append(doc_table)
        story.append(Spacer(1, 20))
        
        # Verification Results
        story.append(Paragraph("VERIFICATION RESULTS", heading_style))
        
        is_authentic = data.get('is_authentic', False)
        authenticity = "✓ AUTHENTIC" if is_authentic else "✗ SUSPICIOUS"
        authenticity_color = colors.HexColor('#059669') if is_authentic else colors.HexColor('#dc2626')
        
        confidence = data.get('confidence', 0)
        similarity = data.get('similarity_score', 0)
        
        results_info = [
            ["Status:", authenticity],
            ["Confidence Score:", f"{confidence:.1f}%"],
            ["Similarity Score:", f"{similarity:.1f}%"],
            ["Overall Verdict:", "DOCUMENT AUTHENTIC" if is_authentic else "DOCUMENT SUSPICIOUS"]
        ]
        
        results_table = Table(results_info, colWidths=[2*inch, 3*inch])
        results_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f0fdf4') if is_authentic else colors.HexColor('#fef2f2')),
            ('TEXTCOLOR', (1, 0), (1, 0), authenticity_color),
            ('TEXTCOLOR', (1, 3), (1, 3), authenticity_color),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e5e7eb'))
        ]))
        story.append(results_table)
        story.append(Spacer(1, 20))
        
        # Detailed Analysis
        if 'analysis' in data and data['analysis']:
            story.append(Paragraph("DETAILED ANALYSIS", heading_style))
            
            analysis_items = []
            for key, value in data['analysis'].items():
                analysis_items.append([key.replace('_', ' ').title() + ":", str(value)])
            
            if analysis_items:
                analysis_table = Table(analysis_items, colWidths=[2*inch, 3*inch])
                analysis_table.setStyle(TableStyle([
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 9),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                    ('TOPPADDING', (0, 0), (-1, -1), 4),
                    ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#f1f5f9'))
                ]))
                story.append(analysis_table)
                story.append(Spacer(1, 20))
        
        # Issues/Anomalies
        if 'issues' in data and data['issues']:
            story.append(Paragraph("DETECTED ISSUES & ANOMALIES", heading_style))
            
            for i, issue in enumerate(data['issues'], 1):
                if issue != 'No significant issues detected':
                    story.append(Paragraph(f"{i}. {issue}", normal_style))
            
            story.append(Spacer(1, 20))
        
        # Comparison Details
        if 'comparison_details' in data and data['comparison_details']:
            story.append(Paragraph("COMPARISON METRICS", heading_style))
            
            comparison_items = []
            for key, value in data['comparison_details'].items():
                comparison_items.append([key.replace('_', ' ').title() + ":", str(value)])
            
            if comparison_items:
                comparison_table = Table(comparison_items, colWidths=[2*inch, 3*inch])
                comparison_table.setStyle(TableStyle([
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 9),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                    ('TOPPADDING', (0, 0), (-1, -1), 4),
                    ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#f1f5f9'))
                ]))
                story.append(comparison_table)
                story.append(Spacer(1, 20))
        
        # System Information
        story.append(Paragraph("SYSTEM INFORMATION", heading_style))
        system_info = [
            ["Software Version:", "CYBER-ID VERIFIER v3.1"],
            ["Report Format:", "Official Verification Document"],
            ["Generated By:", "Administrator"],
            ["Purpose:", "Document Authentication Verification"],
            ["Confidentiality:", "Level 3 - Restricted"]
        ]
        
        system_table = Table(system_info, colWidths=[2*inch, 3*inch])
        system_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f8fafc')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#e2e8f0'))
        ]))
        story.append(system_table)
        story.append(Spacer(1, 30))
        
        # Footer/Disclaimer
        disclaimer = """
        <b>DISCLAIMER:</b> This report is generated automatically by the Cyber-ID Verifier system. 
        The results are based on computer analysis and should be used as a reference only. 
        Final authentication decisions should be made by trained personnel. 
        This document is confidential and intended for authorized personnel only.
        """
        story.append(Paragraph(disclaimer, ParagraphStyle(
            'Disclaimer',
            parent=styles['Normal'],
            fontSize=8,
            textColor=colors.grey,
            alignment=1
        )))
        
        # Build PDF
        doc.build(story)
        
        # Get PDF data
        pdf_data = buffer.getvalue()
        buffer.close()
        
        # Save to reports folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        doc_type = data.get('document_type', 'document').replace('_', '')
        status = "authentic" if is_authentic else "suspicious"
        filename = f"cyberid_verification_{doc_type}_{status}_{timestamp}.pdf"
        filepath = os.path.join(REPORTS_FOLDER, filename)
        
        # Save file to reports folder
        with open(filepath, 'wb') as f:
            f.write(pdf_data)
        
        # Create response
        response = make_response(pdf_data)
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = f'attachment; filename="{filename}"'
        
        # Log the report generation
        print(f"Report generated: {filename} - {doc_type} - Authentic: {is_authentic}")

        return response

    except Exception as e:
        print(f"Report generation error: {e}")
        return jsonify({'success': False, 'message': f'Report generation failed: {str(e)}'}), 500

@app.route('/loan-application', methods=['GET', 'POST'])
def loan_application():
    """Render loan application form (GET) and handle submission (POST)."""
    if not verify_session(request):
        return redirect('/login')
    # GET: render form
    if request.method == 'GET':
        return render_template('loan_application.html')

    # POST: process form submission
    full_name = request.form.get('full_name', '').strip()
    contact = request.form.get('contact', '').strip()
    try:
        amount = float(request.form.get('amount', '0'))
    except:
        amount = 0.0
    try:
        months = int(request.form.get('months', '0'))
    except:
        months = 0
    try:
        annual_rate = float(request.form.get('interest_rate', '0'))
    except:
        annual_rate = 0.0

    # Calculate monthly payment using annuity formula
    monthly_payment = 0.0
    if months > 0:
        r = annual_rate / 100.0 / 12.0
        if r > 0:
            monthly_payment = (amount * r) / (1 - (1 + r) ** (-months))
        else:
            monthly_payment = amount / months if months else 0.0

    monthly_payment_str = f"PHP {monthly_payment:,.2f}"

    # Check for existing application by same contact or full name
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('SELECT id FROM loan_applications WHERE contact = ? OR full_name = ? LIMIT 1', (contact, full_name))
        existing = c.fetchone()
        conn.close()
        if existing:
            # Applicant already has an application — do not process
            error_msg = 'An application for this full name or contact already exists.'
            return render_template('loan_application.html', error=error_msg,
                                   full_name=full_name, contact=contact,
                                   amount=amount, months=months, interest_rate=annual_rate)
    except Exception as e:
        print('Failed to check for duplicate application:', e)

    # Persist application to reports folder (JSON) and save to SQLite
    try:
        os.makedirs(REPORTS_FOLDER, exist_ok=True)
        app_data = {
            'timestamp': datetime.now().isoformat(),
            'full_name': full_name,
            'contact': contact,
            'amount': amount,
            'months': months,
            'interest_rate': annual_rate,
            'monthly_payment': monthly_payment,
        }
        # include verification JSON if provided
        ver_json = request.form.get('verification_json')
        verification_text = None
        if ver_json:
            try:
                app_data['verification'] = json.loads(ver_json)
                verification_text = json.dumps(app_data['verification'], ensure_ascii=False)
            except:
                app_data['verification_raw'] = ver_json
                verification_text = ver_json

        # Save JSON copy (optional backup)
        try:
            filename = os.path.join(REPORTS_FOLDER, f"loan_application_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(app_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print('Failed to save loan application JSON backup:', e)

        # Save to SQLite
        try:
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute('''INSERT INTO loan_applications
                         (timestamp, full_name, contact, amount, months, interest_rate, monthly_payment, verification)
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                      (app_data['timestamp'], full_name, contact, amount, months, annual_rate, monthly_payment, verification_text))
            conn.commit()
            conn.close()
        except Exception as e:
            print('Failed to save loan application to SQLite:', e)
    except Exception as e:
        print('Failed to prepare loan application save:', e)

    return render_template('loan_confirmation.html', full_name=full_name, contact=contact,
                           amount=f"PHP {amount:,.2f}", months=months,
                           interest_rate=annual_rate, monthly_payment=monthly_payment_str)

@app.route('/list-reports', methods=['GET'])
def list_reports():
    """List all generated reports"""
    if not verify_session(request):
        return jsonify({'success': False, 'message': 'Authentication required'}), 401
    
    try:
        reports = []
        for filename in os.listdir(REPORTS_FOLDER):
            if filename.endswith('.pdf'):
                filepath = os.path.join(REPORTS_FOLDER, filename)
                file_stats = os.stat(filepath)
                reports.append({
                    'filename': filename,
                    'size': file_stats.st_size,
                    'created': datetime.fromtimestamp(file_stats.st_ctime).isoformat(),
                    'path': f'/download-report/{filename}'
                })
        
        return jsonify({
            'success': True,
            'reports': sorted(reports, key=lambda x: x['created'], reverse=True),
            'count': len(reports)
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/dashboard', methods=['GET'])
def dashboard():
    """Show a dashboard of loan applications (requires auth)."""
    if not verify_session(request):
        return redirect('/login')

    apps = []
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute('SELECT id,timestamp,full_name,contact,amount,months,interest_rate,monthly_payment FROM loan_applications ORDER BY id DESC')
        rows = c.fetchall()
        for r in rows:
            apps.append({
                'id': r['id'],
                'timestamp': r['timestamp'],
                'full_name': r['full_name'],
                'contact': r['contact'],
                'amount': r['amount'],
                'months': r['months'],
                'interest_rate': r['interest_rate'],
                'monthly_payment': r['monthly_payment']
            })
        conn.close()
    except Exception as e:
        print('Failed to load applications for dashboard:', e)

    return render_template('dashboard.html', applications=apps)


@app.route('/authorize-admin', methods=['POST'])
def authorize_admin():
    """Verify admin password (used before sensitive actions)."""
    if not verify_session(request):
        return jsonify({'authorized': False, 'message': 'Authentication required'}), 401

    data = {}
    if request.is_json:
        data = request.get_json()
    else:
        data = request.form

    password = data.get('password', '')
    if password == VALID_CREDENTIALS.get('password'):
        return jsonify({'authorized': True})
    return jsonify({'authorized': False, 'message': 'Invalid password'}), 403


@app.route('/delete-application/<int:app_id>', methods=['POST'])
def delete_application(app_id):
    """Delete a loan application by id (requires admin password)."""
    if not verify_session(request):
        return jsonify({'success': False, 'message': 'Authentication required'}), 401

    data = request.get_json() if request.is_json else request.form
    password = data.get('password', '')
    if password != VALID_CREDENTIALS.get('password'):
        return jsonify({'success': False, 'message': 'Invalid admin password'}), 403

    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('DELETE FROM loan_applications WHERE id = ?', (app_id,))
        conn.commit()
        deleted = c.rowcount
        conn.close()
        if deleted:
            return jsonify({'success': True, 'deleted': deleted})
        else:
            return jsonify({'success': False, 'message': 'Not found'}), 404
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/update-application/<int:app_id>', methods=['POST'])
def update_application(app_id):
    """Update loan application fields (requires admin password)."""
    if not verify_session(request):
        return jsonify({'success': False, 'message': 'Authentication required'}), 401

    data = request.get_json() if request.is_json else request.form
    password = data.get('password', '')
    if password != VALID_CREDENTIALS.get('password'):
        return jsonify({'success': False, 'message': 'Invalid admin password'}), 403

    # Allowed fields to update
    full_name = data.get('full_name')
    contact = data.get('contact')
    try:
        amount = float(data.get('amount')) if data.get('amount') is not None else None
    except:
        amount = None
    try:
        months = int(data.get('months')) if data.get('months') is not None else None
    except:
        months = None
    try:
        interest_rate = float(data.get('interest_rate')) if data.get('interest_rate') is not None else None
    except:
        interest_rate = None

    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        # Build dynamic update
        fields = []
        values = []
        if full_name is not None:
            fields.append('full_name = ?')
            values.append(full_name)
        if contact is not None:
            fields.append('contact = ?')
            values.append(contact)
        if amount is not None:
            fields.append('amount = ?')
            values.append(amount)
        if months is not None:
            fields.append('months = ?')
            values.append(months)
        if interest_rate is not None:
            fields.append('interest_rate = ?')
            values.append(interest_rate)

        if not fields:
            conn.close()
            return jsonify({'success': False, 'message': 'No fields to update'}), 400

        values.append(app_id)
        sql = f"UPDATE loan_applications SET {', '.join(fields)} WHERE id = ?"
        c.execute(sql, tuple(values))
        conn.commit()
        updated = c.rowcount
        conn.close()
        if updated:
            return jsonify({'success': True, 'updated': updated})
        else:
            return jsonify({'success': False, 'message': 'Not found or no change'}), 404
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/download-report/<filename>')
def download_report(filename):
    """Download a specific report"""
    if not verify_session(request):
        return jsonify({'success': False, 'message': 'Authentication required'}), 401
    
    try:
        # Security check to prevent directory traversal
        if '..' in filename or filename.startswith('/'):
            return jsonify({'success': False, 'message': 'Invalid filename'}), 400
        
        filepath = os.path.join(REPORTS_FOLDER, filename)
        
        if not os.path.exists(filepath):
            return jsonify({'success': False, 'message': 'Report not found'}), 404
        
        return send_file(
            filepath,
            as_attachment=True,
            download_name=filename,
            mimetype='application/pdf'
        )
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/cleanup-reports', methods=['POST'])
def cleanup_reports():
    """Clean up reports older than 30 days"""
    if not verify_session(request):
        return jsonify({'success': False, 'message': 'Authentication required'}), 401
    
    try:
        cutoff_date = datetime.now() - timedelta(days=30)
        deleted_count = 0
        
        for filename in os.listdir(REPORTS_FOLDER):
            if filename.endswith('.pdf'):
                filepath = os.path.join(REPORTS_FOLDER, filename)
                file_mtime = datetime.fromtimestamp(os.path.getmtime(filepath))
                
                if file_mtime < cutoff_date:
                    os.remove(filepath)
                    deleted_count += 1
        
        return jsonify({
            'success': True,
            'message': f'Cleaned up {deleted_count} old reports',
            'deleted_count': deleted_count
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

# ============= PROTECTED APPLICATION ROUTES =============
@app.route('/upload-reference', methods=['POST'])
def upload_reference():
    """Upload a reference document for specific document type"""
    if not verify_session(request):
        return jsonify({'success': False, 'message': 'Authentication required'}), 401
    
    try:
        document_type = request.form.get('document_type', DEFAULT_DOC_TYPE)
        
        if document_type not in DOCUMENT_TYPES:
            return jsonify({'success': False, 'message': 'Invalid document type'}), 400
        
        if 'reference_file' not in request.files:
            return jsonify({'success': False, 'message': 'No file uploaded'}), 400
        
        reference_file = request.files['reference_file']
        
        if reference_file.filename == '':
            return jsonify({'success': False, 'message': 'No file selected'}), 400
        
        if not allowed_file(reference_file.filename):
            return jsonify({'success': False, 'message': 'Invalid file type. Please upload JPG, PNG, or PDF.'}), 400
        
        # Check file size
        reference_file.seek(0, 2)
        file_size = reference_file.tell()
        reference_file.seek(0)
        
        if file_size > MAX_FILE_SIZE:
            return jsonify({'success': False, 'message': f'File too large ({file_size/1024/1024:.1f}MB > 5MB)'}), 400
        
        # Get the folder for this document type
        folder_path = get_reference_folder(document_type)
        
        # Clear existing reference files in this folder
        for f in os.listdir(folder_path):
            file_path = os.path.join(folder_path, f)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")
        
        # Save reference file
        filename = secure_filename(reference_file.filename)
        filepath = os.path.join(folder_path, filename)
        reference_file.save(filepath)
        
        # Try to load the reference
        reference_data[document_type]['image'] = None
        reference_data[document_type]['features'] = None
        
        success = load_reference_license(document_type)
        
        if success:
            return jsonify({
                'success': True,
                'message': f'{document_type.replace("_", " ").title()} reference uploaded and loaded successfully',
                'filename': filename,
                'size': file_size,
                'has_reference': True,
                'document_type': document_type
            })
        else:
            # Delete the file if it couldn't be loaded
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'success': False, 'message': f'Could not process {document_type} reference. Please check file format.'}), 400
        
    except Exception as e:
        print(f"Reference upload error: {str(e)}")
        return jsonify({'success': False, 'message': f'Internal server error: {str(e)}'}), 500

@app.route('/check-reference', methods=['GET'])
def check_reference():
    """Check if reference exists for a specific document type"""
    if not verify_session(request):
        return jsonify({'success': False, 'message': 'Authentication required'}), 401
    
    try:
        document_type = request.args.get('type', DEFAULT_DOC_TYPE)
        
        if document_type not in DOCUMENT_TYPES:
            return jsonify({'success': False, 'message': 'Invalid document type'}), 400
        
        folder_path = get_reference_folder(document_type)
        reference_files = [f for f in os.listdir(folder_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.pdf'))]
        
        has_reference_file = len(reference_files) > 0
        
        # Always try to load reference if file exists
        if has_reference_file:
            reference_loaded = load_reference_license(document_type)
        else:
            reference_loaded = False
        
        return jsonify({
            'success': True,
            'has_reference': reference_loaded,
            'document_type': document_type,
            'reference_file': reference_files[0] if reference_files else None,
            'reference_loaded': reference_loaded
        })
        
    except Exception as e:
        print(f"Error checking reference: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/verify', methods=['POST'])
def verify_license():
    """Verify document using reference comparison"""
    if not verify_session(request):
        return jsonify({'success': False, 'message': 'Authentication required'}), 401
    
    try:
        # Get document type from form or use default
        document_type = request.form.get('document_type', DEFAULT_DOC_TYPE)
        
        if document_type not in DOCUMENT_TYPES:
            return jsonify({'success': False, 'message': 'Invalid document type'}), 400
        
        if 'license_file' not in request.files:
            return jsonify({'success': False, 'message': 'No file uploaded'}), 400
        
        license_file = request.files['license_file']
        
        if license_file.filename == '':
            return jsonify({'success': False, 'message': 'No file selected'}), 400
        
        if not allowed_file(license_file.filename):
            return jsonify({'success': False, 'message': 'Invalid file type. Please upload JPG, PNG, or PDF.'}), 400
        
        # Check file size
        license_file.seek(0, 2)
        file_size = license_file.tell()
        license_file.seek(0)
        
        if file_size > MAX_FILE_SIZE:
            return jsonify({'success': False, 'message': f'File too large ({file_size/1024/1024:.1f}MB > 5MB)'}), 400
        
        # Save and process file
        with tempfile.TemporaryDirectory() as temp_dir:
            license_path = os.path.join(temp_dir, secure_filename(license_file.filename))
            license_file.save(license_path)
            
            # Analyze using reference comparison
            result = analyze_with_comparison(license_path, document_type)
            
            # Add metadata
            result['success'] = True
            result['timestamp'] = datetime.now().isoformat()
            
            # Provide helpful message based on result
            doc_name = document_type.replace('_', ' ').title()
            
            if result['has_reference']:
                if result['is_authentic']:
                    if result['confidence'] > 85:
                        result['message'] = f'High confidence - {doc_name} closely matches reference'
                    elif result['confidence'] > 75:
                        result['message'] = f'Good confidence - {doc_name} similar to reference'
                    elif result['confidence'] > 65:
                        result['message'] = f'Moderate confidence - Some differences from {doc_name.lower()} reference'
                    elif result['confidence'] > 55:
                        result['message'] = f'Low confidence - Multiple differences detected but appears authentic'
                    else:
                        result['message'] = f'Very low confidence - Significant differences detected'
                else:
                    if result['confidence'] < 40:
                        result['message'] = f'High suspicion - Significant differences from {doc_name.lower()} reference'
                    elif result['confidence'] < 55:
                        result['message'] = f'Moderate suspicion - Does not match {doc_name.lower()} pattern'
                    else:
                        result['message'] = f'Suspicious - Multiple issues compared to {doc_name.lower()} reference'
            else:
                if result['is_authentic']:
                    result['message'] = f'Basic check passed - No {doc_name.lower()} reference available for comparison'
                else:
                    result['message'] = f'Failed basic check - No {doc_name.lower()} reference available'
            
            # Debug information
            print(f"\n=== VERIFICATION RESULT ===")
            print(f"Document Type: {document_type}")
            print(f"Authentic: {result['is_authentic']}")
            print(f"Confidence: {result['confidence']:.1f}%")
            print(f"Similarity: {result['similarity_score']:.1f}%")
            print(f"Has Reference: {result['has_reference']}")
            print(f"Issues: {result['issues']}")
            print("=========================\n")
            
            return jsonify(result)
            
    except Exception as e:
        print(f"Verification error: {str(e)}")
        return jsonify({'success': False, 'message': f'Internal server error: {str(e)}'}), 500

@app.route('/get-document-types', methods=['GET'])
def get_document_types():
    """Get list of available document types"""
    if not verify_session(request):
        return jsonify({'success': False, 'message': 'Authentication required'}), 401
    
    return jsonify({
        'success': True,
        'document_types': DOCUMENT_TYPES,
        'active_types': {
            'drivers_license': True,
            'national_id': True,
            'passport': True
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Enhanced health check with document types"""
    if not verify_session(request):
        return jsonify({'success': False, 'message': 'Authentication required'}), 401
    
    status = {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'service': 'Multi-ID Verification System',
        'version': '3.1.0',
        'supported_documents': DOCUMENT_TYPES,
        'references_loaded': {}
    }
    
    # Check each document type
    for doc_type in DOCUMENT_TYPES:
        folder_path = get_reference_folder(doc_type)
        files = [f for f in os.listdir(folder_path) 
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.pdf'))]
        status['references_loaded'][doc_type] = len(files) > 0
    
    return jsonify(status)

@app.route('/reset-all', methods=['POST'])
def reset_all_references():
    """Reset all references (optional endpoint)"""
    if not verify_session(request):
        return jsonify({'success': False, 'message': 'Authentication required'}), 401
    
    try:
        for doc_type in DOCUMENT_TYPES:
            folder_path = get_reference_folder(doc_type)
            # Clear folder
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")
            
            # Reset in-memory data
            reference_data[doc_type] = {'image': None, 'features': None}
        
        return jsonify({
            'success': True,
            'message': 'All references cleared',
            'cleared_types': DOCUMENT_TYPES
        })
        
    except Exception as e:
        print(f"Reset error: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

# ============= ERROR HANDLERS =============
@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'success': False, 'message': 'Page not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'success': False, 'message': 'Internal server error'}), 500

# ============= APPLICATION STARTUP =============
if __name__ == '__main__':
    print("=" * 70)
    print("MULTI-DOCUMENT VERIFICATION SYSTEM v3.1")
    print("IMPROVED DETECTION VERSION")
    print("=" * 70)
    print("Supported Document Types:")
    for doc_type in DOCUMENT_TYPES:
        print(f"  • {doc_type.replace('_', ' ').title()}")
    print("\nIMPROVEMENTS:")
    print("• Lowered similarity thresholds for better detection")
    print("• Added partial keyword matching")
    print("• More lenient image comparison")
    print("• Better OCR error handling")
    print("\nINSTRUCTIONS:")
    print("1. Place authentic documents in respective reference folders:")
    for doc_type in DOCUMENT_TYPES:
        folder = get_reference_folder(doc_type)
        print(f"   - {doc_type.replace('_', ' ').title()}: {folder}")
    print("2. Supported formats: JPG, PNG, PDF")
    print("3. Run the application")
    print("4. Open http://localhost:5000 in your browser")
    print("5. Login with credentials: admin / jethro123")
    print("6. Select document type and upload documents to verify")
    print("7. Generate PDF reports for documentation")
    print("\n" + "-" * 70)
    
    # Auto-load references on startup
    for doc_type in DOCUMENT_TYPES:
        folder_path = get_reference_folder(doc_type)
        reference_files = [f for f in os.listdir(folder_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.pdf'))]
        
        if reference_files:
            print(f"\n✓ Found {len(reference_files)} reference file(s) for {doc_type.replace('_', ' ').title()}")
            print(f"  Main reference: {reference_files[0]}")
            
            if load_reference_license(doc_type):
                print(f"  ✓ {doc_type.replace('_', ' ').title()} reference loaded successfully")
            else:
                print(f"  ✗ Could not load {doc_type.replace('_', ' ').title()} reference")
        else:
            print(f"\n⚠ No reference found for {doc_type.replace('_', ' ').title()}")
            print(f"  Add authentic document to: {folder_path}")
    
    print("\n" + "-" * 70)
    print("Starting server on http://localhost:5000")
    print("Login credentials: admin / jethro123")
    print("Press Ctrl+C to stop the server")
    print("=" * 70)
    
    app.run(debug=True, host='0.0.0.0', port=5000)