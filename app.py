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

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
REFERENCE_FOLDER = 'reference_licenses'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf'}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

# Create necessary directories
for folder in [UPLOAD_FOLDER, REFERENCE_FOLDER, 'static', 'templates']:
    os.makedirs(folder, exist_ok=True)

# Set Tesseract path
try:
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
except:
    try:
        pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
    except:
        pass

# Global variables for reference data
reference_license = None
reference_features = None

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_reference_license():
    """Load the reference driver's license for comparison"""
    global reference_license, reference_features
    
    try:
        # Check if reference exists
        reference_files = [f for f in os.listdir(REFERENCE_FOLDER) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.pdf'))]
        
        if not reference_files:
            print("No reference license found in folder.")
            return False
        
        # Use the first reference file found
        reference_path = os.path.join(REFERENCE_FOLDER, reference_files[0])
        print(f"Loading reference license from: {reference_path}")
        
        # Load reference image
        if reference_path.lower().endswith('.pdf'):
            images = convert_from_path(reference_path)
            if images:
                temp_img_path = os.path.join(tempfile.gettempdir(), 'temp_reference.png')
                images[0].save(temp_img_path, 'PNG')
                reference_license = cv2.imread(temp_img_path)
                # Clean up temp file
                if os.path.exists(temp_img_path):
                    os.remove(temp_img_path)
            else:
                print("Could not process PDF reference")
                return False
        else:
            reference_license = cv2.imread(reference_path)
        
        if reference_license is None:
            print("Could not read reference license image")
            return False
        
        # Extract features from reference
        reference_text = extract_text_from_file(reference_path)
        reference_features = extract_detailed_features(reference_license, reference_text)
        
        print(f"Reference license loaded successfully. Size: {reference_license.shape}")
        return True
        
    except Exception as e:
        print(f"Error loading reference license: {e}")
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
            enhanced = enhancer.enhance(1.5)  # Reduced from 2.0 to avoid over-enhancement
            text = pytesseract.image_to_string(enhanced)
            return text
    except Exception as e:
        print(f"Text extraction error: {e}")
        return ""

def extract_detailed_features(img, text):
    """Extract detailed features from image and text"""
    features = {}
    
    try:
        # 1. Image features
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Basic image properties
        features['height'] = int(img.shape[0])
        features['width'] = int(img.shape[1])
        features['aspect_ratio'] = float(img.shape[1] / img.shape[0])
        
        # Color features (mean and std for each channel)
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
            features['text'] = str(text)  # Ensure it's a string
            
            # Look for common patterns in driver's licenses
            patterns = {
                'license_number': r'\b[A-Z]{1,2}\d{6,9}\b',
                'dates': r'\b\d{1,2}[-/]\d{1,2}[-/]\d{4}\b',
                'zip_code': r'\b\d{5}(?:-\d{4})?\b',
            }
            
            for name, pattern in patterns.items():
                matches = re.findall(pattern, text, re.IGNORECASE)
                features[f'{name}_count'] = int(len(matches))
        
        return features
        
    except Exception as e:
        print(f"Feature extraction error: {e}")
        return features

def compare_with_reference(img, text):
    """Compare uploaded license with reference license"""
    comparison_results = {
        'similarity_score': 0.0,
        'differences': [],
        'details': {}
    }
    
    try:
        if reference_license is None or reference_features is None:
            if not load_reference_license():
                comparison_results['differences'].append('No reference license available for comparison')
                return comparison_results
        
        # Extract features from uploaded image
        uploaded_features = extract_detailed_features(img, text)
        
        # 1. Compare image properties
        img_similarities = []
        
        # Size comparison (allow some variation)
        if 'height' in reference_features and 'height' in uploaded_features:
            height_diff = abs(reference_features['height'] - uploaded_features['height']) / max(reference_features['height'], 1) * 100
            if height_diff > 30:
                comparison_results['differences'].append(f'Height difference: {height_diff:.1f}%')
            img_similarities.append(max(0.0, 100.0 - height_diff))
        
        # Aspect ratio comparison
        if 'aspect_ratio' in reference_features and 'aspect_ratio' in uploaded_features:
            ref_ratio = reference_features['aspect_ratio']
            upload_ratio = uploaded_features['aspect_ratio']
            ratio_diff = abs(ref_ratio - upload_ratio) / max(ref_ratio, 0.01) * 100
            if ratio_diff > 20:
                comparison_results['differences'].append(f'Aspect ratio difference: {ratio_diff:.1f}%')
            img_similarities.append(max(0.0, 100.0 - ratio_diff))
        
        # Color comparison
        color_similarity = 50.0  # Default
        if all(k in reference_features for k in ['color_mean_b', 'color_mean_g', 'color_mean_r']) and \
           all(k in uploaded_features for k in ['color_mean_b', 'color_mean_g', 'color_mean_r']):
            
            ref_color = np.array([reference_features['color_mean_b'], 
                                  reference_features['color_mean_g'], 
                                  reference_features['color_mean_r']])
            upload_color = np.array([uploaded_features['color_mean_b'], 
                                     uploaded_features['color_mean_g'], 
                                     uploaded_features['color_mean_r']])
            
            color_diff = np.mean(np.abs(ref_color - upload_color))
            color_similarity = max(0.0, 100.0 - color_diff)
            if color_diff > 40:
                comparison_results['differences'].append('Significant color difference detected')
        
        img_similarities.append(color_similarity)
        
        # Sharpness comparison
        if 'sharpness' in reference_features and 'sharpness' in uploaded_features:
            sharpness_diff = abs(reference_features['sharpness'] - uploaded_features['sharpness']) / max(reference_features['sharpness'], 1.0) * 100
            if sharpness_diff > 60:
                comparison_results['differences'].append(f'Sharpness difference: {sharpness_diff:.1f}%')
            img_similarities.append(max(0.0, 100.0 - sharpness_diff))
        
        # 2. Compare text
        text_similarities = []
        
        if 'text' in reference_features and 'text' in uploaded_features:
            ref_text = reference_features['text'].lower()
            upload_text = uploaded_features['text'].lower()
            
            # Common keywords check
            common_keywords = ['driver', 'license', 'licence', 'state', 'expires', 'expiration',
                              'birth', 'dob', 'date of birth', 'issued', 'height', 'weight']
            
            ref_keywords = [kw for kw in common_keywords if kw in ref_text]
            upload_keywords = [kw for kw in common_keywords if kw in upload_text]
            
            if ref_keywords:
                keyword_similarity = len(set(ref_keywords) & set(upload_keywords)) / max(len(ref_keywords), 1) * 100
                text_similarities.append(keyword_similarity)
                
                if keyword_similarity < 60:
                    comparison_results['differences'].append(f'Missing important keywords')
            
            # Text length comparison
            if 'text_length' in reference_features and 'text_length' in uploaded_features:
                ref_len = reference_features['text_length']
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
            'overall_similarity': f"{overall_similarity:.1f}%"
        }
        
        return comparison_results
        
    except Exception as e:
        print(f"Comparison error: {e}")
        comparison_results['differences'].append(f'Comparison error: {str(e)}')
        return comparison_results

def analyze_with_comparison(file_path):
    """Main analysis function using reference comparison"""
    
    result = {
        'is_authentic': False,
        'confidence': 0.0,
        'similarity_score': 0.0,
        'issues': [],
        'analysis': {},
        'method': 'reference_comparison',
        'comparison_details': {},
        'has_reference': False
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
        
        # Basic validation - is this a driver's license?
        text = extract_text_from_file(file_path)
        text_lower = text.lower() if text else ""
        
        license_keywords = ['driver', 'license', 'licence', 'permit', 'dl', 'driving']
        keyword_count = sum(1 for kw in license_keywords if kw in text_lower)
        
        # Check if we have a reference license
        result['has_reference'] = reference_license is not None
        
        if result['has_reference']:
            # 1. Compare with reference license
            comparison = compare_with_reference(img, text)
            result['similarity_score'] = float(comparison['similarity_score'])
            result['comparison_details'] = comparison['details']
            result['issues'].extend(comparison['differences'])
            
            # 2. Run anomaly detection as secondary check
            features = extract_detailed_features(img, text)
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
                result['issues'].append('This does not appear to be a driver\'s license document')
            elif similarity_confidence >= 70.0 and final_confidence >= 65.0:
                result['is_authentic'] = True
            elif similarity_confidence >= 60.0 and final_confidence >= 60.0:
                result['is_authentic'] = True
                result['issues'].append('Minor differences from reference')
            else:
                result['is_authentic'] = False
            
            # 5. Detailed analysis
            result['analysis'] = {
                'similarity_to_reference': f"{comparison['similarity_score']:.1f}%",
                'image_quality': f"{features.get('sharpness', 0)/10:.1f}/10" if 'sharpness' in features else "N/A",
                'text_analysis': f"{len(text)} characters, {keyword_count} license keywords found",
                'aspect_ratio': f"{features.get('aspect_ratio', 0):.2f}" if 'aspect_ratio' in features else "N/A"
            }
            
        else:
            # No reference available, use basic validation
            if keyword_count < 2:
                result['issues'].append('This does not appear to be a driver\'s license document')
                result['confidence'] = 10.0
                return result
            
            features = extract_detailed_features(img, text)
            
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
                'text_analysis': f"{len(text)} characters, {keyword_count} license keywords found",
                'method': 'Basic Validation (No reference)'
            }
        
        if not result['issues'] and result['confidence'] > 70.0:
            result['issues'].append('No significant issues detected')
        
    except Exception as e:
        result['issues'].append(f'Analysis error: {str(e)}')
        result['confidence'] = 0.0
    
    return result

@app.route('/')
def index():
    """Serve the main HTML page"""
    return render_template('index.html')

@app.route('/upload-reference', methods=['POST'])
def upload_reference():
    """Upload a reference/authentic driver's license via web interface"""
    try:
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
        
        # Clear existing reference files
        for f in os.listdir(REFERENCE_FOLDER):
            file_path = os.path.join(REFERENCE_FOLDER, f)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")
        
        # Save reference file
        filename = secure_filename(reference_file.filename)
        filepath = os.path.join(REFERENCE_FOLDER, filename)
        reference_file.save(filepath)
        
        # Try to load the reference
        global reference_license, reference_features
        reference_license = None
        reference_features = None
        
        success = load_reference_license()
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Reference license uploaded and loaded successfully',
                'filename': filename,
                'size': file_size,
                'has_reference': True
            })
        else:
            # Delete the file if it couldn't be loaded
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'success': False, 'message': 'Could not process reference license. Please check file format.'}), 400
        
    except Exception as e:
        print(f"Reference upload error: {str(e)}")
        return jsonify({'success': False, 'message': f'Internal server error: {str(e)}'}), 500

@app.route('/check-reference', methods=['GET'])
def check_reference():
    """Check if reference license exists and is loaded"""
    try:
        reference_files = [f for f in os.listdir(REFERENCE_FOLDER) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.pdf'))]
        
        has_reference_file = len(reference_files) > 0
        
        # Always try to load reference if file exists
        if has_reference_file:
            global reference_license, reference_features
            reference_loaded = load_reference_license()
        else:
            reference_loaded = False
        
        return jsonify({
            'success': True,
            'has_reference': reference_loaded,
            'reference_file': reference_files[0] if reference_files else None,
            'reference_loaded': reference_loaded
        })
        
    except Exception as e:
        print(f"Error checking reference: {e}")
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/verify', methods=['POST'])
def verify_license():
    """Verify driver's license using reference comparison"""
    try:
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
            result = analyze_with_comparison(license_path)
            
            # Add metadata
            result['success'] = True
            result['timestamp'] = datetime.now().isoformat()
            
            # Provide helpful message based on result
            if result['has_reference']:
                if result['is_authentic']:
                    if result['confidence'] > 85:
                        result['message'] = 'High confidence - License closely matches reference'
                    elif result['confidence'] > 75:
                        result['message'] = 'Good confidence - License similar to reference'
                    elif result['confidence'] > 65:
                        result['message'] = 'Moderate confidence - Some differences from reference'
                    else:
                        result['message'] = 'Low confidence - Multiple differences detected'
                else:
                    if result['confidence'] < 40:
                        result['message'] = 'High suspicion - Significant differences from reference'
                    elif result['confidence'] < 55:
                        result['message'] = 'Moderate suspicion - Does not match reference pattern'
                    else:
                        result['message'] = 'Suspicious - Multiple issues compared to reference'
            else:
                if result['is_authentic']:
                    result['message'] = 'Basic check passed - No reference available for comparison'
                else:
                    result['message'] = 'Failed basic check - No reference available'
            
            return jsonify(result)
            
    except Exception as e:
        print(f"Verification error: {str(e)}")
        return jsonify({'success': False, 'message': f'Internal server error: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    reference_files = [f for f in os.listdir(REFERENCE_FOLDER) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.pdf'))]
    
    global reference_license
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'service': 'Driver\'s License Verification System',
        'version': '2.0.1',
        'method': 'Reference-based Comparison',
        'has_reference_file': len(reference_files) > 0,
        'reference_loaded': reference_license is not None,
        'reference_file': reference_files[0] if reference_files else None,
        'features': [
            'Reference license comparison',
            'Image similarity analysis',
            'Text pattern matching',
            'Anomaly detection as backup'
        ]
    })

if __name__ == '__main__':
    print("=" * 70)
    print("DRIVER'S LICENSE VERIFICATION SYSTEM")
    print("=" * 70)
    print("Method: Reference-based Comparison")
    print("\nINSTRUCTIONS:")
    print("1. Place your AUTHENTIC driver's license in the 'reference_licenses' folder")
    print("2. Supported formats: JPG, PNG, PDF")
    print("3. Run the application")
    print("4. Open http://localhost:5000 in your browser")
    print("5. Upload licenses to verify against your reference")
    print("\n" + "-" * 70)
    
    # Auto-load reference license on startup
    reference_files = [f for f in os.listdir(REFERENCE_FOLDER) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.pdf'))]
    
    if reference_files:
        print(f"\n✓ Found {len(reference_files)} reference file(s)")
        print(f"  Main reference: {reference_files[0]}")
        
        if load_reference_license():
            print("✓ Reference license loaded successfully")
            print(f"✓ System ready for verification")
        else:
            print("✗ Could not load reference license")
            print("  Please check if the file is a valid image/PDF format")
    else:
        print("\n⚠ No reference license found in 'reference_licenses' folder!")
        print("  System will use basic validation only")
        print("  For better accuracy, add your authentic license to the folder")
    
    print("\n" + "-" * 70)
    print("Starting server on http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    print("=" * 70)
    
    app.run(debug=True, host='0.0.0.0', port=5000)