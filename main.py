from ultralytics import YOLO
import cv2
import easyocr
import os

# Create output folder
os.makedirs("output", exist_ok=True)

# Load YOLO model (use your downloaded model)
model = YOLO('license_plate_detector.pt')

# Initialize OCR
print("Loading EasyOCR... (first time takes a while)")
reader = easyocr.Reader(['en'])
print("Ready!")

def detect_plate(image_path):
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Cannot read {image_path}")
        return
    
    # Detect plates
    results = model(img)
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Get coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            
            # Crop plate
            plate_crop = img[y1:y2, x1:x2]
            
            if plate_crop.size == 0:
                continue
            
            # Convert to grayscale for better OCR
            gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
            
            # OCR
            ocr_result = reader.readtext(gray)
            
            plate_text = ""
            if ocr_result:
                plate_text = " ".join([res[1] for res in ocr_result])
                plate_text = plate_text.upper().replace(" ", "")
            
            # Draw box and text
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(img, plate_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            print(f"✅ Detected Plate: {plate_text} (Confidence: {confidence:.2f})")
    
    # Save and show result
    output_path = f"output/result_{os.path.basename(image_path)}"
    cv2.imwrite(output_path, img)
    print(f"Saved to: {output_path}")
    
    cv2.imshow("Detection Result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Test with an image
    detect_plate("images/car1.jpg")
