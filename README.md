# Cardiovascular Disease (CVD) Risk Predictor 🫀

An end-to-end Machine Learning web application and clinical screening pipeline to assess cardiovascular disease risk and provide probability estimates based on patient biometrics, vitals, and lifestyle indicators.

Built with **Python**, **scikit-learn**, and **Streamlit**.

---

## 🌟 Features

- **Interactive Medical Web UI**: Clean, responsive layout with clinical reference indicators and real-time biometric metrics.
- **Dynamic Clinical Badges**:
  - **Live BMI Calculator**: Computes Body Mass Index on the fly with category tags (Underweight, Normal, Overweight, Obese).
  - **AHA/ACC Blood Pressure Staging**: Automatically categorizes blood pressure readings into Normal, Elevated, Stage 1 Hypertension, or Stage 2 Hypertension.
- **End-to-End Scikit-Learn Pipeline**:
  - Custom data cleansing and age preprocessing.
  - Automatic obesity feature engineering (`identify_obesity`).
  - Feature scaling with `StandardScaler` for continuous numerical variables.
  - Support Vector Machine (SVM / SGD) classifier.
- **Comprehensive Risk Assessment**:
  - Visual probability gauge and metric cards with clinical deltas.
  - Detailed breakdown of identified risk factors and protective indicators.
  - Pre-configured patient profiles (*Healthy* vs *High Risk*) for quick demonstration.
- **Medical Disclaimer**: Clear ethical disclaimer emphasizing educational screening use.

---

## 📂 Project Structure

```text
├── app.py                      # Main Streamlit web application
├── cardio_modelTraining.ipynb  # End-to-end model training & evaluation pipeline
├── cardio_vascular_EDA.ipynb   # Exploratory Data Analysis (EDA) on dataset
├── cardio_train.csv            # CVD dataset (70,000 patient records)
├── trained_svm_pipeline.pkl    # Serialized scikit-learn pipeline
├── requirements.txt            # Python package dependencies
└── README.md                   # Project documentation
```

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/AnujThakur-ai/CVD.git
cd CVD
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the Application
```bash
streamlit run app.py
```

The application will launch in your default browser at `http://localhost:8501`.

---

## 🩺 Clinical Input Features

The model evaluates the following biometric and lifestyle features:

| Feature | Description | Values / Units |
| :--- | :--- | :--- |
| **Age** | Patient age | Years (18 - 100) |
| **Gender** | Biological sex | Female (1) / Male (2) |
| **Height** | Standing height | Centimeters (cm) |
| **Weight** | Body weight | Kilograms (kg) |
| **ap_hi** | Systolic blood pressure | mmHg |
| **ap_lo** | Diastolic blood pressure | mmHg |
| **Cholesterol** | Total cholesterol | 1: Normal, 2: Above Normal, 3: Well Above Normal |
| **Glucose** | Fasting blood glucose | 1: Normal, 2: Above Normal, 3: Well Above Normal |
| **Smoking** | Tobacco smoking | 0: No, 1: Yes |
| **Alcohol** | Alcohol consumption | 0: No, 1: Yes |
| **Active** | Physical activity | 0: No, 1: Yes |

---

## ⚠️ Medical Disclaimer

This application is powered by a machine learning model trained on observational population data. It is intended solely for educational, research, and informational screening purposes and does **not** constitute medical advice, clinical diagnosis, or treatment recommendations. Always consult a qualified physician or cardiologist for medical advice.
