import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# App title
st.title("Heart Disease Prediction")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('heart.csv')
    df = df.drop_duplicates()
    return df

df = load_data()

# Sidebar menu
st.sidebar.header("Menu")
show_data = st.sidebar.checkbox("Show Raw Data")

if show_data:
    st.subheader("Raw Data")
    st.write(df)

# Model training
st.subheader("Model Training")

# Feature selection
features = st.multiselect(
    "Select features for model",
    df.columns.drop('target'),
    default=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs']
)

if not features:
    st.error("Please select at least one feature")
    st.stop()

X = df[features]
y = df['target']

# Train-test split
test_size = st.slider("Test size (%)", 10, 50, 30)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size/100, random_state=42
)

# Model selection
model_type = st.selectbox(
    "Select Model",
    ("Random Forest", "Logistic Regression", "SVM")
)

# Model training
if model_type == "Random Forest":
    model = RandomForestClassifier()
elif model_type == "Logistic Regression":
    model = LogisticRegression(max_iter=1000)
else:
    model = SVC(probability=True)  # Added probability=True for predict_proba

model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.success(f"Model Accuracy: {accuracy:.2%}")

# Confusion Matrix with orange-blue color scheme
st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6,5))

# Create custom orange-blue colormap
from matplotlib.colors import LinearSegmentedColormap
colors = ["#3498db", "#f39c12"]  # Blue to Orange
cmap = LinearSegmentedColormap.from_list("orange_blue", colors, N=256)

sns.heatmap(cm, annot=True, fmt='d', ax=ax, 
            cmap=cmap,
            cbar=False,
            linewidths=2,
            linecolor='navy',
            xticklabels=['Healthy', 'Disease'],
            yticklabels=['Healthy', 'Disease'],
            annot_kws={"color": "navy", "fontsize": 14, "fontweight": "bold"})

ax.set_xlabel('Predicted', fontsize=12, color='navy', fontweight='bold')
ax.set_ylabel('Actual', fontsize=12, color='navy', fontweight='bold')
ax.set_xticklabels(ax.get_xticklabels(), color='navy', fontsize=12)
ax.set_yticklabels(ax.get_yticklabels(), color='navy', fontsize=12)

# Add border
for _, spine in ax.spines.items():
    spine.set_visible(True)
    spine.set_color('navy')
    spine.set_linewidth(2)

st.pyplot(fig)

# Classification report
st.subheader("Classification Report")
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()
st.dataframe(report_df.style.background_gradient(cmap=cmap))

# Prediction form
st.subheader("Patient Details")

# Create input fields for ALL selected features
input_values = []
for feature in features:
    if feature == 'age':
        val = st.number_input("Age", min_value=20, max_value=100, value=50)
    elif feature == 'sex':
        val = st.selectbox("Sex", ["Female", "Male"])
        val = 1 if val == "Male" else 0
    elif feature == 'cp':
        val = st.selectbox("Chest Pain Type", [0, 1, 2, 3], 
                         format_func=lambda x: ['Typical angina', 'Atypical angina', 
                                             'Non-anginal pain', 'Asymptomatic'][x])
    elif feature == 'trestbps':
        val = st.number_input("Resting Blood Pressure (mm Hg)", min_value=90, max_value=200, value=120)
    elif feature == 'chol':
        val = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
    elif feature == 'fbs':
        val = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    else:
        val = st.number_input(f"Enter {feature}")
    
    input_values.append(val)

if st.button("Predict"):
    # Convert to 2D array for prediction
    input_data = np.array([input_values])
    
    try:
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0][1] if hasattr(model, "predict_proba") else None
        
        # Show result with styled output
        col1, col2 = st.columns(2)
        with col1:
            if prediction == 1:
                st.error("🚨 Prediction: Heart Disease Risk")
            else:
                st.success("✅ Prediction: Healthy Heart")
        
        with col2:
            if prediction_proba is not None:
                st.metric(label="Risk Probability", 
                          value=f"{prediction_proba:.1%}",
                          delta="High risk" if prediction_proba > 0.5 else "Low risk")
        
        # Show feature importance for Random Forest
        if model_type == "Random Forest":
            st.subheader("Feature Importance")
            importances = model.feature_importances_
            importance_df = pd.DataFrame({
                'Feature': features,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            fig2, ax2 = plt.subplots()
            sns.barplot(x='Importance', y='Feature', data=importance_df, palette=colors[::-1], ax=ax2)
            ax2.set_title('Feature Importance Scores', color='navy')
            st.pyplot(fig2)
            
    except Exception as e:
        st.error(f"❌ Prediction failed: {str(e)}")