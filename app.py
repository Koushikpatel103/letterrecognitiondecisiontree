import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree

from src.predict import predict_letter
from src.utils import load_object

st.set_page_config(
    page_title="Letter Recognition",
    layout="wide"
)

st.title("🔠 Letter Recognition using Decision Tree")

# Load dataset
df = pd.read_csv("data/letter-recognition.csv")

# Dataset Preview
st.subheader("Dataset Preview")
st.dataframe(df.head())

# Dataset Shape
st.subheader("Dataset Shape")
st.write(df.shape)

# Class Distribution
st.subheader("Class Distribution")

fig1, ax1 = plt.subplots()
df.iloc[:, 0].value_counts().plot(
    kind="bar",
    ax=ax1
)
st.pyplot(fig1)

# Features and Target
X = df.iloc[:, 1:]
y = df.iloc[:, 0]

# Load encoder and model
label_encoder = load_object("label_encoder.pkl")
model = load_object("model.pkl")

# Encode labels
y_encoded = label_encoder.transform(y)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42
)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

st.subheader("Model Accuracy")
st.success(f"{accuracy:.4f}")

# Feature Importance
st.subheader("Feature Importance")

fig2, ax2 = plt.subplots(figsize=(10,5))

importance = model.feature_importances_

ax2.bar(X.columns, importance)
plt.xticks(rotation=90)

st.pyplot(fig2)

# Confusion Matrix
st.subheader("Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

labels = label_encoder.classes_

fig3, ax3 = plt.subplots(figsize=(8,6))

im = ax3.imshow(cm)

ax3.set_xticks(range(len(labels)))
ax3.set_yticks(range(len(labels)))

ax3.set_xticklabels(labels, rotation=90)
ax3.set_yticklabels(labels)

ax3.set_xlabel("Predicted")
ax3.set_ylabel("Actual")

st.pyplot(fig3)

# Tree Visualization
st.subheader("Decision Tree Visualization")

fig4, ax4 = plt.subplots(figsize=(20,10))

plot_tree(
    model,
    filled=True,
    max_depth=2,
    fontsize=8
)

st.pyplot(fig4)

# Prediction UI
st.subheader("Predict Letter")

inputs = []

for col in X.columns:
    value = st.number_input(
        f"{col}",
        value=0
    )
    inputs.append(value)

if st.button("Predict"):
    result = predict_letter(inputs)
    st.success(f"Predicted Letter: {result}")