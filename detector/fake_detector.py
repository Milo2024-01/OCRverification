class FakeDocumentDetector:
    def __init__(self):
        pass

    def analyze_document(self, text):
        """
        Simple rule-based fake detection
        """
        result = {}
        if not text.strip():
            result['status'] = 'FAKE'
            result['reason'] = 'No text detected'
        elif 'REPUBLIC' in text.upper() or 'ID' in text.upper():
            result['status'] = 'LEGIT'
            result['reason'] = 'Keywords found'
        else:
            result['status'] = 'FAKE'
            result['reason'] = 'Missing required keywords'
        return result
