# Intelligent-Framework-for-YouTube-Video-Quality-Assessment

## 📌 Project Overview
This project aims to evaluate the quality and credibility of YouTube videos by analyzing user comments using **Natural Language Processing (NLP)** and **Machine Learning (ML)** techniques.  
Instead of relying on manual inspection, the system automatically filters spam comments, analyzes genuine user opinions, and generates a rating and verdict indicating whether a video is worth watching.

---

## 🎯 Objectives
- Create a chrome extension that
- Automatically fetch YouTube video comments
- Detect and remove spam comments using machine learning
- Analyze sentiment of genuine comments
- Generate a rating and verdict for each video
- Compare multiple videos to recommend the best one

---

## ⚙️ Tech Stack
- **Programming Language:** Python  
- **Machine Learning:** Scikit-learn  
- **NLP Techniques:** TF-IDF Vectorization  
- **Models:** SVM
- **Backend (Planned):** Flask / FastAPI  
- **Frontend (Planned):** Chrome Extension  
- **API (Planned):** YouTube Data API v3  

---

## ✅ Work Completed So Far

### 🔹 Dataset Setup & Exploration
- Downloaded YouTube Comment Spam Dataset (Kaggle)
- Analyzed dataset structure and labels
- Identified relevant features for ML training
- Performed basic text cleaning and preprocessing

### 🔹 Spam Detection Model
- Applied TF-IDF vectorization on comment text
- Trained SVM for spam detection
- Evaluated model using accuracy, precision, recall, and confusion matrix
- Achieved ~92% accuracy on real-world YouTube spam data
- Tested the model with custom user comments

📌 **Design Decision:**  
Spam detection is performed **before sentiment analysis** to ensure only genuine user opinions influence the final rating.

---

## 🔁 Current System Workflow
1. Load and preprocess YouTube comments
2. Detect and remove spam comments using ML
3. Pass genuine comments to sentiment analysis module (next phase)
4. Aggregate results for rating and verdict generation

---

## 🚧 Upcoming Work

### 🔹 Sentiment Analysis Model
- Train a sentiment classification model (Positive / Negative / Neutral)
- Use TF-IDF and SVM
- Evaluate sentiment model performance

### 🔹 Backend Integration
- Develop backend API using Flask / FastAPI
- Integrate spam detection and sentiment analysis models
- Create endpoints for comment analysis

### 🔹 Frontend & Extension
- Build a Chrome extension UI
- Display rating, sentiment breakdown, and verdict on YouTube
- Enable multi-video comparison feature

---

## 📈 Future Enhancements
- Deep learning-based sentiment analysis (BERT)
- Multilingual comment support
- Like-weighted sentiment scoring
- Advanced spam filtering
- Visual analytics (charts and graphs)

---

## 📄 Project Status
🟢 **In Progress**  
Current Phase: **Spam Detection Module Completed**

---

## 🧠 Academic Relevance
This project demonstrates the practical application of NLP and Machine Learning in real-world content analysis.

---

## 👨‍💻 Author
Abhinav Yadav
