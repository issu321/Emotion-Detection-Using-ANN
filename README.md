# Emotion-Detection-Using-ANN

<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Space+Grotesk&weight=700&size=30&duration=3000&pause=1000&color=6366F1&center=true&vCenter=true&width=600&lines=Emotion-Detection-Using-ANN;Speech+Emotion+Recognition;Powered+by+Artificial+Neural+Networks;Real-Time+Analysis+%7C+7+Emotions;Open+Source+%7C+Flask+%7C+Python" alt="Typing Animation" />
</p>

<p align="center">
  <a href="https://github.com/issu321/Emotion-Detection-Using-ANN">
    <img src="https://img.shields.io/badge/GitHub-issu321-6366f1?style=for-the-badge&logo=github&logoColor=white" alt="GitHub" />
  </a>
  <a href="https://github.com/issu321/Emotion-Detection-Using-ANN/stargazers">
    <img src="https://img.shields.io/github/stars/issu321/Emotion-Detection-Using-ANN?style=for-the-badge&color=ec4899" alt="Stars" />
  </a>
  <a href="https://github.com/issu321/Emotion-Detection-Using-ANN/network/members">
    <img src="https://img.shields.io/github/forks/issu321/Emotion-Detection-Using-ANN?style=for-the-badge&color=06b6d4" alt="Forks" />
  </a>
  <a href="https://github.com/issu321/Emotion-Detection-Using-ANN/issues">
    <img src="https://img.shields.io/github/issues/issu321/Emotion-Detection-Using-ANN?style=for-the-badge&color=f59e0b" alt="Issues" />
  </a>
</p>

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6366f1,ec4899,06b6d4&height=200&section=header&text=Emotion-Detection-Using-ANN&fontSize=40&fontColor=fff&animation=fadeIn" alt="Wave Banner" />
</p>

---

## Features

| Feature | Description | Status |
|---------|-------------|--------|
| **ANN Classification** | 4-layer neural network with 191 input features | Active |
| **Real-Time Dashboard** | Interactive charts with live waveform visualization | Active |
| **Live Recording** | Browser-based microphone recording | Active |
| **Batch Upload** | Process multiple files simultaneously | Active |
| **Trend Analysis** | 7-day emotion tracking with line charts | Active |
| **Dark/Light Mode** | Theme toggle with persistent storage | Active |
| **Secure Auth** | Encrypted passwords & role-based access | Active |
| **Export Data** | CSV & JSON export for all history | Active |
| **Notifications** | Real-time toast & bell notifications | Active |
| **Three.js Particles** | Animated 3D background | Active |

---

## Architecture

```
Audio Upload -> Signal Preprocessing -> Feature Extraction (191 Features)
    -> ANN Model (256-128-64-32) -> Softmax Output -> 7 Emotions
    -> Dashboard Visualization -> History & Trends
```

---

## Tech Stack

- **Backend:** Python, Flask
- **Frontend:** HTML5, CSS3, JavaScript, Chart.js, Three.js, GSAP
- **Audio Processing:** Librosa, NumPy
- **Machine Learning:** Artificial Neural Network (ANN)
- **Database:** SQLite
- **UI Design:** Glass-morphism, Gradient effects, Animations

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/issu321/Emotion-Detection-Using-ANN.git
cd Emotion-Detection-Using-ANN

# 2. Create virtual environment
python -m venv venv

# 3. Activate environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run the application
python app.py

# 6. Open browser
# http://localhost:5000
```

---

## Usage

1. **Register** a new account or **Login** with existing credentials
2. **Upload audio** via drag-and-drop, file picker, or **record live**
3. View **real-time analysis** with animated probability charts
4. Check **emotion history** and **trend analysis**
5. **Export** your data to CSV or JSON
6. Toggle **settings** (dark mode, waveform, notifications)

**Admin Access:**
- Username: `admin`
- Password: `admin123`

---

## Model Performance

| Metric | Value |
|--------|-------|
| Input Features | 191 |
| Hidden Layers | 4 (256, 128, 64, 32) |
| Output Classes | 7 Emotions |
| Dropout | 0.3 |
| Optimizer | Adam |
| Activation | ReLU / Softmax |

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Home page |
| `/register` | GET/POST | User registration |
| `/login` | GET/POST | User login |
| `/logout` | GET | Logout |
| `/dashboard` | GET | Main dashboard |
| `/predict` | POST | Single audio prediction |
| `/predict_batch` | POST | Batch prediction |
| `/api/history` | GET | Get emotion history |
| `/api/stats` | GET | Get statistics |
| `/api/trend` | GET | Get trend data |
| `/api/export/<format>` | GET | Export CSV/JSON |
| `/api/model_info` | GET | Model architecture |
| `/api/live_demo` | GET | Live demo simulation |

---

## Emotion Classes

| Emotion | Icon | Description |
|---------|------|-------------|
| Neutral | :neutral_face: | Calm, balanced tone |
| Happy | :smile: | Joyful, upbeat speech |
| Sad | :cry: | Melancholic, low energy |
| Angry | :angry: | Intense, harsh tone |
| Fearful | :fearful: | Anxious, trembling voice |
| Disgusted | :nauseated_face: | Revulsion, aversion |
| Surprised | :astonished: | Astonishment, shock |

---

## Project Structure

```
Emotion-Detection-Using-ANN/
├── app.py                 # Main Flask application
├── README.md             # This file
├── INPUTGUIDE.md         # Input guide documentation
├── requirements.txt      # Python dependencies
├── templates/
│   ├── base.html         # Base template with navigation
│   ├── index.html        # Home page
│   ├── login.html        # Login page
│   ├── register.html     # Registration page
│   ├── dashboard.html    # Main analysis dashboard
│   ├── about.html        # About project page
│   ├── features.html     # Features page
│   ├── inputguide.html   # Input guide page
│   ├── model.html        # Model architecture page
│   └── admin.html        # Admin dashboard
├── static/
│   ├── css/              # Stylesheets
│   ├── js/               # JavaScript files
│   ├── images/           # Images
│   └── uploads/          # Audio uploads storage
└── users.db             # SQLite database (auto-generated)
```

---

## Contributing

Contributions are welcome! Please submit a PR.

---

## License

This project is open source and available for educational and research purposes.

---

<p align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6366f1,ec4899,06b6d4&height=150&section=footer&text=Built+with+by+@issu321&fontSize=20&fontColor=fff" alt="Footer" />
</p>

<p align="center">
  <a href="https://github.com/issu321">GitHub Profile</a> •
  <a href="https://github.com/issu321/Emotion-Detection-Using-ANN">Repository</a>
</p>
