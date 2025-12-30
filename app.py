from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import tempfile
from datetime import datetime
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import pytesseract
from pdf2image import convert_from_path
import re
import json
import shutil

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# ============= CONFIGURATION =============
UPLOAD_FOLDER = 'uploads'
REFERENCE_BASE_FOLDER = 'references'  # Changed from 'reference_licenses'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf'}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

# Supported document types
DOCUMENT_TYPES = ['drivers_license', 'national_id']
DEFAULT_DOC_TYPE = 'drivers_license'

# Create necessary directories
for folder in [UPLOAD_FOLDER, 'static', 'templates']:
    os.makedirs(folder, exist_ok=True)

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
# Now we store references per document type
reference_data = {
    'drivers_license': {'image': None, 'features': None},
    'national_id': {'image': None, 'features': None}
}

# ============= HELPER FUNCTIONS =============
def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_reference_folder(document_type):
    """Get the reference folder for a specific document type"""
    if document_type not in DOCUMENT_TYPES:
        return os.path.join(REFERENCE_BASE_FOLDER, DEFAULT_DOC_TYPE)
    return os.path.join(REFERENCE_BASE_FOLDER, document_type)

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
        
        # Size comparison
        if 'height' in ref_features and 'height' in uploaded_features:
            height_diff = abs(ref_features['height'] - uploaded_features['height']) / max(ref_features['height'], 1) * 100
            if height_diff > 30:
                comparison_results['differences'].append(f'Height difference: {height_diff:.1f}%')
            img_similarities.append(max(0.0, 100.0 - height_diff))
        
        # Aspect ratio comparison
        if 'aspect_ratio' in ref_features and 'aspect_ratio' in uploaded_features:
            ref_ratio = ref_features['aspect_ratio']
            upload_ratio = uploaded_features['aspect_ratio']
            ratio_diff = abs(ref_ratio - upload_ratio) / max(ref_ratio, 0.01) * 100
            if ratio_diff > 20:
                comparison_results['differences'].append(f'Aspect ratio difference: {ratio_diff:.1f}%')
            img_similarities.append(max(0.0, 100.0 - ratio_diff))
        
        # Color comparison
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
            if color_diff > 40:
                comparison_results['differences'].append('Significant color difference detected')
        
        img_similarities.append(color_similarity)
        
        # Sharpness comparison
        if 'sharpness' in ref_features and 'sharpness' in uploaded_features:
            sharpness_diff = abs(ref_features['sharpness'] - uploaded_features['sharpness']) / max(ref_features['sharpness'], 1.0) * 100
            if sharpness_diff > 60:
                comparison_results['differences'].append(f'Sharpness difference: {sharpness_diff:.1f}%')
            img_similarities.append(max(0.0, 100.0 - sharpness_diff))
        
        # 2. Compare text
        text_similarities = []
        
        if 'text' in ref_features and 'text' in uploaded_features:
            ref_text = ref_features['text'].lower()
            upload_text = uploaded_features['text'].lower()
            
            # Document type specific keywords
            if document_type == 'drivers_license':
                common_keywords = ['driver', 'license', 'licence', 'state', 'expires', 'expiration',
                                  'birth', 'dob', 'date of birth', 'issued', 'height', 'weight', 'driving']
            elif document_type == 'national_id':
                common_keywords = ['national', 'id', 'identification', 'republic', 'philippines',
                                  'birth', 'dob', 'date of birth', 'address', 'sex', 'gender',
                                  'civil status', 'blood type', 'signature']
            else:
                common_keywords = []
            
            ref_keywords = [kw for kw in common_keywords if kw in ref_text]
            upload_keywords = [kw for kw in common_keywords if kw in upload_text]
            
            if ref_keywords:
                keyword_similarity = len(set(ref_keywords) & set(upload_keywords)) / max(len(ref_keywords), 1) * 100
                text_similarities.append(keyword_similarity)
                
                if keyword_similarity < 60:
                    comparison_results['differences'].append(f'Missing important {document_type} keywords')
            
            # Text length comparison
            if 'text_length' in ref_features and 'text_length' in uploaded_features:
                ref_len = ref_features['text_length']
                upload_len = uploaded_features['text_length']
                if ref_len > 0 and upload_len > 0:
                    length_similarity = min(ref_len, upload_len) / max(ref_len, upload_len) * 100
                    text_similarities.append(length_similarity)
                    if length_similarity < 50:
                        comparison_results['differences'].append(f'Text length significantly different')
        
        # 3. Calculate overall similarity score
        if img_similarities:
            img_score = float(np.mean(img_similarities))
        else:
            img_score = 50.0
        
        if text_similarities:
            text_score = float(np.mean(text_similarities))
        else:
            text_score = 50.0
        
        # Weighted average (60% image, 40% text)
        overall_similarity = img_score * 0.6 + text_score * 0.4
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
        
        # Document type specific validation
        if document_type == 'drivers_license':
            keywords = ['driver', 'license', 'licence', 'permit', 'dl', 'driving']
            doc_name = "driver's license"
        elif document_type == 'national_id':
            keywords = ['national', 'id', 'identification', 'republic', 'philippines']
            doc_name = "national ID"
        else:
            keywords = []
            doc_name = "document"
        
        keyword_count = sum(1 for kw in keywords if kw in text_lower)
        
        # Check if we have a reference document
        result['has_reference'] = reference_data[document_type]['image'] is not None
        
        if result['has_reference']:
            # 1. Compare with reference document
            comparison = compare_with_reference(img, text, document_type)
            result['similarity_score'] = float(comparison['similarity_score'])
            result['comparison_details'] = comparison['details']
            result['issues'].extend(comparison['differences'])
            
            # 2. Run anomaly detection as secondary check
            features = extract_detailed_features(img, text, document_type)
            anomaly_score = 0.0
            
            # Check image quality
            if 'sharpness' in features and features['sharpness'] < 30:
                anomaly_score += 25.0
                result['issues'].append('Very low image sharpness')
            elif 'sharpness' in features and features['sharpness'] < 50:
                anomaly_score += 15.0
                result['issues'].append('Low image sharpness')
            
            if 'contrast' in features and features['contrast'] < 20:
                anomaly_score += 20.0
                result['issues'].append('Very low contrast')
            elif 'contrast' in features and features['contrast'] < 35:
                anomaly_score += 10.0
                result['issues'].append('Low contrast')
            
            # 3. Calculate overall confidence
            similarity_confidence = float(comparison['similarity_score'])
            anomaly_adjustment = max(0.0, 100.0 - anomaly_score)
            
            # Combined confidence
            if keyword_count >= 2:
                final_confidence = similarity_confidence * 0.7 + anomaly_adjustment * 0.3
            else:
                final_confidence = similarity_confidence * 0.5 + anomaly_adjustment * 0.5
                
            result['confidence'] = float(final_confidence)
            
            # 4. Determine authenticity
            if keyword_count < 2:
                result['is_authentic'] = False
                result['issues'].append(f'This does not appear to be a {doc_name} document')
            elif similarity_confidence >= 70.0 and final_confidence >= 65.0:
                result['is_authentic'] = True
            elif similarity_confidence >= 60.0 and final_confidence >= 60.0:
                result['is_authentic'] = True
                result['issues'].append(f'Minor differences from {doc_name} reference')
            else:
                result['is_authentic'] = False
            
            # 5. Detailed analysis
            result['analysis'] = {
                'similarity_to_reference': f"{comparison['similarity_score']:.1f}%",
                'image_quality': f"{features.get('sharpness', 0)/10:.1f}/10" if 'sharpness' in features else "N/A",
                'text_analysis': f"{len(text)} characters, {keyword_count} {doc_name} keywords found",
                'aspect_ratio': f"{features.get('aspect_ratio', 0):.2f}" if 'aspect_ratio' in features else "N/A",
                'document_type': doc_name
            }
            
        else:
            # No reference available, use basic validation
            if keyword_count < 2:
                result['issues'].append(f'This does not appear to be a {doc_name} document')
                result['confidence'] = 10.0
                return result
            
            features = extract_detailed_features(img, text, document_type)
            
            # Simple quality scoring
            quality_score = 0.0
            
            if 'sharpness' in features and features['sharpness'] > 50:
                quality_score += 30.0
            elif 'sharpness' in features and features['sharpness'] > 20:
                quality_score += 15.0
            
            if 'contrast' in features and features['contrast'] > 30:
                quality_score += 20.0
            elif 'contrast' in features and features['contrast'] > 15:
                quality_score += 10.0
            
            if 'aspect_ratio' in features:
                aspect = features['aspect_ratio']
                if 1.4 <= aspect <= 1.8:
                    quality_score += 25.0
                elif 1.3 <= aspect <= 1.9:
                    quality_score += 15.0
            
            # Add text score
            text_score = min(100.0, keyword_count * 15.0)
            quality_score += text_score
            
            result['confidence'] = min(100.0, quality_score)
            
            # Without reference, be more conservative
            result['is_authentic'] = result['confidence'] >= 60.0 and keyword_count >= 2
            
            result['analysis'] = {
                'image_quality': f"{features.get('sharpness', 0)/10:.1f}/10" if 'sharpness' in features else "N/A",
                'confidence_score': f"{result['confidence']:.1f}%",
                'text_analysis': f"{len(text)} characters, {keyword_count} {doc_name} keywords found",
                'method': f'Basic Validation (No {doc_name} reference)'
            }
        
        if not result['issues'] and result['confidence'] > 70.0:
            result['issues'].append('No significant issues detected')
        
    except Exception as e:
        result['issues'].append(f'Analysis error: {str(e)}')
        result['confidence'] = 0.0
    
    return result

# ============= FLASK ROUTES =============
@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')

@app.route('/upload-reference', methods=['POST'])
def upload_reference():
    """Upload a reference document for specific document type"""
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
                    else:
                        result['message'] = f'Low confidence - Multiple differences detected'
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
            
            return jsonify(result)
            
    except Exception as e:
        print(f"Verification error: {str(e)}")
        return jsonify({'success': False, 'message': f'Internal server error: {str(e)}'}), 500

@app.route('/get-document-types', methods=['GET'])
def get_document_types():
    """Get list of available document types"""
    return jsonify({
        'success': True,
        'document_types': DOCUMENT_TYPES,
        'active_types': {
            'drivers_license': True,
            'national_id': True
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Enhanced health check with document types"""
    status = {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'service': 'Multi-ID Verification System',
        'version': '3.0.0',
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

if __name__ == '__main__':
    print("=" * 70)
    print("MULTI-DOCUMENT VERIFICATION SYSTEM v3.0")
    print("=" * 70)
    print("Supported Document Types:")
    for doc_type in DOCUMENT_TYPES:
        print(f"  • {doc_type.replace('_', ' ').title()}")
    print("\nINSTRUCTIONS:")
    print("1. Place authentic documents in respective reference folders:")
    for doc_type in DOCUMENT_TYPES:
        folder = get_reference_folder(doc_type)
        print(f"   - {doc_type.replace('_', ' ').title()}: {folder}")
    print("2. Supported formats: JPG, PNG, PDF")
    print("3. Run the application")
    print("4. Open http://localhost:5000 in your browser")
    print("5. Select document type and upload documents to verify")
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
    print("Press Ctrl+C to stop the server")
    print("=" * 70)
    
    app.run(debug=True, host='0.0.0.0', port=5000)