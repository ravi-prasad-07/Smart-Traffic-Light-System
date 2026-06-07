````md
# 🚦 Smart Traffic Light System

An AI-powered web-based traffic management system using YOLO for real-time vehicle, pedestrian, and emergency vehicle detection. The system dynamically controls traffic signals based on traffic density to reduce congestion and improve road safety.

---

## ✨ Features

- 🚗 Real-time vehicle detection and counting
- 🚦 Dynamic traffic signal timing based on traffic density
- 🚑 Emergency vehicle priority system
- 🏛 Government priority lane clearance feature
- 🚶 Pedestrian crossing signal management
- 🌐 Web-based monitoring dashboard
- 🗄 SQLite database integration for traffic data storage
- 🖼 Processed image storage in separate folders
- ⚡ Automatically uses GPU if available, otherwise CPU

---

## 🛠 Tech Stack

- Python
- YOLOv8
- OpenCV
- Flask
- SQLite
- HTML, CSS, JavaScript

---

## 📂 Project Structure

```bash
Smart-Traffic-Light-System/
│
├── static/
│   ├── css/
│   │   └── styles.css
│   ├── js/
│   │   └── app.js
│   ├── snapshots/
│   └── uploads/
│
├── templates/
│   └── index.html
│
├── __pycache__/
│
├── .yolo/
│
├── app.py
├── emergency_detector_v2.py
├── model_optimizer.py
├── runtime_config.py
├── storage.py
├── traffic_core_optimized.py
├── vehicle_tracker.py
├── traffic.db
├── yolov8m.pt
├── requirements.txt
└── README.md
````

---

## ⚙️ How It Works

1. Video feeds are uploaded or streamed into the system.
2. YOLOv8 detects vehicles, pedestrians, and emergency vehicles.
3. Traffic density is calculated in real time.
4. Traffic signals are dynamically adjusted.
5. Emergency vehicles receive instant green signal priority.
6. Pedestrian crossing signals are managed automatically.
7. Traffic data is stored in SQLite database.
8. Processed images are saved for monitoring and analysis.
9. GPU acceleration is used when available for faster performance.

---

## 🚀 Installation

### Clone the Repository

```bash
git clone https://github.com/your-username/Smart-Traffic-Light-System.git
cd Smart-Traffic-Light-System
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Application

```bash
python app.py
```

---

## 🎯 Objective

The objective of this project is to build a smart and efficient AI-based traffic management system that:

* Reduces traffic congestion
* Improves emergency response time
* Enhances pedestrian safety
* Supports intelligent city infrastructure

---

## 👨‍💻 Team Members

* Ravi Prasad
* Sanjay Rawat
* Saloni Gupta
* Rohan Singh Rawat
* Krish Bhatt

---

## 📌 Future Improvements

* Live CCTV integration
* IoT-based smart traffic system
* Cloud traffic analytics dashboard
* Mobile application support
* Automatic accident detection

---

## 📄 License

This project is developed for educational and research purposes.

```
```
