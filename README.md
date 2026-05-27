# 🔠 Letter Recognition using Decision Tree and Streamlit

An end-to-end Machine Learning classification project that predicts alphabet letters (A–Z) using a **Decision Tree Classifier**. This project includes data preprocessing, model training, evaluation, and an interactive **Streamlit dashboard** for real-time predictions and visualizations.

---

## 📌 Features

✅ Decision Tree Classification  
✅ Data Preprocessing  
✅ Train/Test Split  
✅ Model Saving using Joblib  
✅ Streamlit Dashboard  
✅ Dataset Preview  
✅ Class Distribution Graph  
✅ Model Accuracy Display  
✅ Confusion Matrix Visualization  
✅ Feature Importance Graph  
✅ Decision Tree Visualization  
✅ Real-Time Letter Prediction  

---

## 📂 Project Structure

```text
LetterRecognitionDecisionTree/
│
├── data/
│   └── letter-recognition.csv
│
├── artifacts/
│   ├── model.pkl
│   └── label_encoder.pkl
│
├── src/
│   ├── __init__.py
│   ├── preprocess.py
│   ├── train.py
│   ├── predict.py
│   └── utils.py
│
├── app.py
├── requirements.txt
└── README.md
```

---

## 🛠️ Tech Stack

- Python
- Scikit-learn
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Joblib

---

## 📊 Dataset

**Letter Recognition Dataset**

- Multi-class classification problem
- Predicts capital letters **A–Z**
- 16 numerical features
- 26 output classes

Dataset file:

```text
letter-recognition.csv
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone <your-github-repo-link>
```

Move into project folder:

```bash
cd LetterRecognitionDecisionTree
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 Train the Model

Run:

```bash
python -m src.train
```

Expected output:

```text
Accuracy: 0.87xx
Model saved successfully!
```

Generated files:

```text
artifacts/
│
├── model.pkl
└── label_encoder.pkl
```

---

## ▶️ Run Streamlit App

Start the dashboard:

```bash
streamlit run app.py
```

Open browser:

```text
http://localhost:8501
```

---

## 📈 Dashboard Includes

### Dataset Preview
View first rows of the dataset.

### Class Distribution
Bar graph showing frequency of letters.

### Model Accuracy
Displays Decision Tree performance.

### Feature Importance
Visualizes most influential features.

### Confusion Matrix
Shows prediction performance across classes.

### Decision Tree Visualization
Displays tree structure.

### Predict Letter
Enter feature values and predict alphabet letter.

---

## 🧠 Machine Learning Workflow

1. Load Dataset  
2. Preprocess Data  
3. Encode Target Labels  
4. Split Dataset  
5. Train Decision Tree Model  
6. Evaluate Accuracy  
7. Save Model  
8. Deploy using Streamlit  

---

## 📷 Sample Output

- Accuracy Dashboard
- Feature Importance Graph
- Confusion Matrix
- Letter Prediction Interface

---

## 📌 Future Improvements

- Hyperparameter Tuning
- Cross Validation
- GridSearchCV
- Improved UI Design
- Cloud Deployment

---

## 👨‍💻 Author

Koushik Patel

---

## ⭐ Support

If you found this project useful, consider giving it a **star ⭐** on GitHub.
