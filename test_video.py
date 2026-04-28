from ultralytics import YOLO
import cv2
import easyocr
import os

# Create output folder
os.makedirs("output", exist_ok=True)

# Load YOLO model
print("Loading YOLO model...")
model = YOLO('license_plate_detector.pt')

# Initialize OCR
print("Loading EasyOCR... (first time takes a while)")
reader = easyocr.Reader(['en'])
print("Ready!\n")

# ============================================
# CHANGE THIS: 0 for webcam, or "video.mp4" for video file
# ============================================
VIDEO_SOURCE = "videos/car_video.mp4"   # or use 0 for webcam
# ============================================

cap = cv2.VideoCapture(VIDEO_SOURCE)

if not cap.isOpened():
    print(f"❌ Error: Cannot open video source: {VIDEO_SOURCE}")
    exit()

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"📹 Video Info: {width}x{height} @ {fps}fps | Total frames: {total_frames}")

# Setup video writer to save output
output_path = "output/result_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Track detected plates
detected_plates = set()
frame_count = 0
ocr_skip = 5  # Run OCR every 5 frames (for speed)

print("\n🚀 Starting detection... Press 'Q' to quit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("\n✅ Video processing complete!")
        break
    
    frame_count += 1
    
    # Run YOLO detection
    results = model(frame, verbose=False)
    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            
            # Skip low confidence detections
            if conf < 0.5:
                continue
            
            # Crop plate
            plate_crop = frame[y1:y2, x1:x2]
            
            if plate_crop.size == 0:
                continue
            
            plate_text = ""
            
            # Run OCR every few frames (saves processing time)
            if frame_count % ocr_skip == 0:
                # Preprocess for better OCR
                gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                
                ocr_result = reader.readtext(gray)
                
                if ocr_result:
                    plate_text = "".join([res[1] for res in ocr_result])
                    plate_text = plate_text.upper().replace(" ", "")
                    
                    # Add to detected plates
                    if plate_text and len(plate_text) > 4:
                        if plate_text not in detected_plates:
                            detected_plates.add(plate_text)
                            print(f"🚗 New Plate Detected: {plate_text} (Confidence: {conf:.2f})")
            
            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            
            # Draw text background
            label = plate_text if plate_text else "Plate"
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w, y1), (0, 255, 0), -1)
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    
    # Show progress on frame
    progress_text = f"Frame: {frame_count}/{total_frames} | Plates Found: {len(detected_plates)}"
    cv2.putText(frame, progress_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Save frame to output video
    out.write(frame)
    
    # Show live preview
    cv2.imshow("License Plate Detection", frame)
    
    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("\n⏹️  Stopped by user")
        break

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()

# Save detected plates to file
with open("output/detected_plates.txt", "w") as f:
    f.write("Detected License Plates:\n")
    f.write("=" * 30 + "\n")
    for plate in detected_plates:
        f.write(f"{plate}\n")

print(f"\n📊 Summary:")
print(f"   Total Frames Processed: {frame_count}")
print(f"   Unique Plates Detected: {len(detected_plates)}")
print(f"   Output Video: {output_path}")
print(f"   Plates List: output/detected_plates.txt")
