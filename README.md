# Federated Learning for Sentiment Analysis using Logistic Regression

A beginner-friendly Federated Learning project for sentiment analysis using the IMDb movie review dataset and Logistic Regression.

This project demonstrates the basic concept of Federated Learning (FL), where multiple clients train models locally without sharing raw data. The server combines local model parameters using Federated Averaging (FedAvg).

---

# 📌 Project Features

- Federated Learning simulation
- Logistic Regression for sentiment classification
- TF-IDF text vectorization
- IMDb movie review dataset
- FedAvg aggregation
- Privacy-preserving learning concept
- Beginner-friendly implementation
- Google Colab compatible

---

# 🧠 Project Concept

Traditional Machine Learning collects all user data into a centralized server.  
Federated Learning solves this privacy problem by training models locally on client devices.

In this project:

- Dataset is divided into 3 simulated clients
- Each client trains a Logistic Regression model locally
- The server aggregates model parameters using FedAvg
- No raw data is shared between clients

---

# 🏗️ System Architecture

```text
        Central Server
              │
     ┌────────┼────────┐
     ▼        ▼        ▼
 Client 1   Client 2   Client 3

 Local Training on Private Data

     ▲        ▲        ▲
     └────────┼────────┘
              │
       Federated Averaging
```

---

# 📂 Dataset

Dataset used:

IMDb Movie Review Dataset

- 50,000 movie reviews
- Positive and Negative sentiments

Dataset Link:

https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

For this project:
- First 5000 samples were used

---

# ⚙️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- TF-IDF Vectorizer
- Logistic Regression

---

# 📦 Installation

Install required libraries:

```bash
pip install pandas scikit-learn
```

---

# 🚀 How to Run

## Step 1: Download Dataset

Download:

```text
IMDB Dataset.csv
```

from Kaggle.

---

## Step 2: Place Dataset

Put the dataset file in the same directory as the Python script.

---

## Step 3: Run the Script

```bash
python federated_sentiment.py
```

---

# 🔄 Workflow

1. Load IMDb dataset
2. Convert text into numerical form using TF-IDF
3. Split dataset into training and testing
4. Simulate 3 federated clients
5. Train local Logistic Regression models
6. Perform Federated Averaging (FedAvg)
7. Create global model
8. Evaluate accuracy
9. Predict custom movie reviews

---

# 📊 Example Output

```text
Loading Dataset...

Training Client 1...
Training Client 2...
Training Client 3...

Federated Learning Completed

Global Model Accuracy: 0.88
```

---

# 🧪 Example Prediction

Input:

```text
This movie was absolutely fantastic
```

Output:

```text
Prediction: Positive
```

---

# 🧮 Federated Averaging (FedAvg)

The server combines parameters from all local client models using averaging.

Formula:

```math
w_global = (w_1 + w_2 + w_3) / 3
```

Where:
- `w_1`, `w_2`, `w_3` are client model parameters

---

# 🎯 Objectives

- Understand Federated Learning basics
- Implement privacy-preserving machine learning
- Learn sentiment analysis using Logistic Regression
- Simulate distributed training

---

# 📈 Future Improvements

- Increase number of federated clients
- Use deep learning models
- Implement real-time federated communication
- Apply Differential Privacy
- Use Flower framework for production-level FL

---

# 📚 Educational Purpose

This project was created as a beginner-level MSc coursework project to understand:
- Federated Learning
- Sentiment Analysis
- Logistic Regression
- FedAvg aggregation

---

# 👨‍💻 Author

Azmanul Abedin Aumi

MSc in CSE (Data Science)  
BRAC University

---

# 📜 License

This project is for educational and research purposes.
