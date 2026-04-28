# CarPlateDetection
A real-time Car Plate Detection system using YOLOv8, OpenCV, and EasyOCR to identify and extract license plate numbers from images and videos. It processes frames efficiently, displays results with bounding boxes, and saves detected plates with confidence scores.


markdown


# 🚗 Car Plate Detection System

[![Python](https://img.shields.io/badge/Python-3.14-blue.svg)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green.svg)](https://github.com/ultralytics/ultralytics)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-red.svg)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A real-time **Car License Plate Detection and Recognition** system built using **YOLOv8**, **OpenCV**, and **EasyOCR/Tesseract**. The system detects license plates from images and videos, extracts the plate text using OCR, and saves the results.

---

## 📸 Demo

![Demo](output/demo.gif)

**Sample Output:**

🚘 New Plate Detected: KA02MM909+ (Confidence: 0.70)
🚘 New Plate Detected: 3402HH7258 (Confidence: 0.60)
🚘 New Plate Detected: 3402HH7256" (Confidence: 0.56)
🚘 New Plate Detected: 14021H7258 (Confidence: 0.59)
🚘 New Plate Detected: 1402WH7258 (Confidence: 0.61)

✅ Video processing complete!

📊 Summary:
Total Frames Processed: 930
Unique Plates Detected: 34
Output Video: output/result_video.mp4
Plates List: output/detected_plates.txt

yaml



---

## ✨ Features

- 🎯 **Real-time Detection** - Detects license plates in images and videos
- 🔤 **OCR Integration** - Extracts plate numbers using EasyOCR
- 📹 **Video Processing** - Process entire video files with frame-by-frame analysis
- 🖼️ **Image Processing** - Single image detection support
- 💾 **Auto-Save Results** - Saves detected plates and processed video
- 📊 **Detailed Logs** - Confidence scores and detection summary
- ⚡ **Optimized Performance** - OCR runs every N frames for speed
- 🎨 **Bounding Boxes** - Visualizes detected plates with labels

---

## 🛠️ Tech Stack

| Technology | Purpose |
|-----------|---------|
| **Python 3.14** | Core programming language |
| **YOLOv8 (Ultralytics)** | License plate detection |
| **OpenCV** | Image/Video processing |
| **EasyOCR** | Text recognition from plates |
| **PyTorch** | Deep learning backend |
| **NumPy** | Numerical operations |

---

## 📁 Project Structure


CarPlateDetection/
│
├── images/                      # Test images folder
├── videos/                      # Input videos folder
├── output/                      # Output results
│   ├── result_video.mp4        # Processed video
│   └── detected_plates.txt     # List of detected plates
│
├── venv/                        # Virtual environment
├── license_plate_detector.pt    # Pre-trained YOLO model
├── main.py                      # Image detection script
├── test_video.py               # Video detection script
├── requirements.txt            # Project dependencies
├── README.md                   # Project documentation
└── LICENSE                     # MIT License

yaml



---

## 🚀 Installation

### Prerequisites

- Python 3.10 or higher
- pip (Python package manager)
- Git

### Step 1: Clone the Repository

```bash
git clone https://github.com/<your-username>/CarPlateDetection.git
cd CarPlateDetection

Step 2: Create a Virtual Environment
bash

# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate

Step 3: Install Dependencies
bash


pip install -r requirements.txt

Step 4: Download the Model
Place the license_plate_detector.pt file in the project root directory.

📥 If you don't have the model, you can download it from here or train your own using YOLOv8.

💻 Usage
🖼️ Detect Plate from an Image
bash


python main.py

Edit the image path inside main.py:

python
Run Code


image_path = "images/car.jpg"


🎥 Detect Plates from a Video
bash

python test_video.py

Edit the video path inside test_video.py:

python
Run Code

Copy code
video_path = "videos/sample.mp4"
output_path = "output/result_video.mp4"


Press Q to quit the video window anytime.

⚙️ Configuration
You can tweak these parameters in test_video.py:

python
Run Code

ocr_skip = 5              # Run OCR every 5 frames (lower = more accurate, slower)
confidence_threshold = 0.5  # Minimum detection confidence


📊 Output
After running, the following files are generated in the output/ folder:

File	Description
result_video.mp4	Annotated video with bounding boxes
detected_plates.txt	List of all unique plates detected
📋 Requirements
txt


ultralytics>=8.0.0
opencv-python>=4.8.0
easyocr>=1.7.0
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
Pillow>=10.0.0

🧠 How It Works
Frame Capture → OpenCV reads each frame from the video
YOLO Detection → YOLOv8 model detects license plate regions
Region Cropping → Detected plate area is cropped
OCR Processing → EasyOCR extracts text from plate
Result Storage → Unique plates are saved with confidence
Video Writing → Annotated frames are written to output video
🐛 Troubleshooting
Problem: ModuleNotFoundError: No module named 'ultralytics'

bash

Copy code
pip install ultralytics

Problem: CUDA not available / Slow performance

bash

# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

Problem: EasyOCR downloads models repeatedly

Models are cached in ~/.EasyOCR/ (delete to reset)
🚧 Future Improvements
 Add support for multiple countries' plate formats
 Web interface using Flask/Streamlit
 Real-time webcam detection
 Database integration for plate logging
 Vehicle make/model detection
 Speed estimation
 Mobile app integration
🤝 Contributing
Contributions are welcome! Please follow these steps:

Fork the project
Create your feature branch (git checkout -b feature/AmazingFeature)
Commit your changes (git commit -m 'Add some AmazingFeature')
Push to the branch (git push origin feature/AmazingFeature)
Open a Pull Request
📜 License
This project is licensed under the MIT License - see the LICENSE file for details.

👨‍💻 Author
Your Name

GitHub: @your-username
LinkedIn: Your LinkedIn
Email: your.email@example.com
🙏 Acknowledgements
Ultralytics YOLOv8
EasyOCR
OpenCV
Roboflow datasets community
⭐ If you found this project helpful, give it a star! ⭐

yaml



---

## 📄 requirements.txt

```txt
ultralytics>=8.0.0
opencv-python>=4.8.0
easyocr>=1.7.0
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
Pillow>=10.0.0
