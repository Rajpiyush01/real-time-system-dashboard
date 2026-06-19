# ⚡ Real-Time System Dashboard

A real-time system monitoring dashboard built with **Python, Flask, WebSockets, Psutil, and Chart.js**.

Monitor system performance through a modern web interface with live updates and interactive charts.

## 🚀 Features

* Real-time CPU monitoring
* RAM usage tracking
* GPU utilization monitoring
* GPU temperature and VRAM tracking
* FPS monitoring
* Multi-GPU support
* GPU diagnostics and health checks
* Live WebSocket updates
* Smart alert system
* Responsive dashboard UI

## 🛠️ Tech Stack

**Backend**

* Python
* Flask
* Flask-Sock
* Psutil
* PyNVML

**Frontend**

* HTML
* CSS
* JavaScript
* Chart.js

## ⚙️ Installation

```bash
git clone https://github.com/Rajpiyush01/real-time-system-dashboard.git
cd real-time-system-dashboard

pip install -r requirements.txt
python server.py
```

Open `index.html` in your browser.

## 🔌 API Endpoints

* `/api/stats`
* `/api/gpu/diagnostics`
* `/api/gpu/stress-test`
* `/api/gpu/benchmark`
* `/ws`

## 🔒 Note

This project is intended for local system monitoring and learning purposes.

## 👨‍💻 Author

Piyush Raj
