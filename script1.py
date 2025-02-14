import json
import cv2
import numpy as np
import pytesseract
from PIL import Image, ExifTags

def correct_orientation(image):
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = image._getexif()
        if exif and orientation in exif:
            orientation = exif[orientation]
            
            # Rotate image based on orientation
            if orientation == 3:
                image = image.rotate(180, expand=True)
            elif orientation == 6:
                image = image.rotate(270, expand=True)
            elif orientation == 8:
                image = image.rotate(90, expand=True)
    except (AttributeError, KeyError, IndexError):
        pass  # No EXIF data found
    return image

def preprocess_image(image_path):
    # Open image and correct orientation
    pil_image = Image.open(image_path)
    pil_image = correct_orientation(pil_image)
    
    # Convert to OpenCV format
    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    
    # Preprocessing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    # Optional: Denoising
    denoised = cv2.fastNlMeansDenoising(thresh, h=10, templateWindowSize=7, searchWindowSize=21)
    
    return denoised

def ocr_image(image_path):
    processed_image = preprocess_image(image_path)
    
    # Perform OCR with both Hindi and English
    custom_config = r'--oem 3 --psm 6 -l hin+eng'
    text = pytesseract.image_to_string(processed_image, config=custom_config)
    
    # Create JSON result
    result = {
        "text": text.strip(),
        "languages": ["Hindi", "English"],
        "orientation_corrected": True
    }
    
    return json.dumps(result, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    image_path = "path/to/your/image.jpg"
    result_json = ocr_image(image_path)
    print(result_json)