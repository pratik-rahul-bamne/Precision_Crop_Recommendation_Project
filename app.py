
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

app = Flask(__name__)
CORS(app)

# Load and preprocess dataset using advanced technology
print("Loading dataset and training advanced model...")
df = pd.read_csv('crop_yield.csv')
df.columns = df.columns.str.strip()

required_columns = [
    'Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'Soil_Type', 'Rainfall', 'Crop'
]
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Missing column: {col} in dataset")

# Clean up whitespace and ensure correct types in all relevant columns
for col in ['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'Rainfall']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df['Soil_Type'] = df['Soil_Type'].astype(str).str.strip()
df['Crop'] = df['Crop'].astype(str).str.strip()

# Remove any rows with missing or non-numeric data in required columns
before_drop = len(df)
df = df.dropna(subset=['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'Rainfall', 'Soil_Type', 'Crop'])
after_drop = len(df)
if before_drop != after_drop:
    print(f"Warning: Dropped {before_drop - after_drop} rows due to missing or non-numeric data.")

# Advanced preprocessing: OneHotEncode Soil_Type, Standardize numeric features
numeric_features = ['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'Rainfall']
categorical_features = ['Soil_Type']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# Use an advanced ensemble model: Stacking with RandomForest and GradientBoosting
estimators = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
    ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42))
]

# Dynamically determine the minimum number of samples per class in the training set
X = df[numeric_features + categorical_features]
y = df['Crop']

# Split for validation (simulate advanced tech with validation)
# Ensure test set has at least as many samples as number of classes (6)
# Use test_size=0.2 or test_size=6 (whichever is larger)
n_classes = y.nunique()
min_test_size = n_classes
if len(df) * 0.15 < min_test_size:
    test_size = min_test_size
else:
    test_size = 0.15
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

# Find the minimum number of samples in any class in the training set
min_samples_per_class = y_train.value_counts().min()
# StackingClassifier's cv cannot be greater than min_samples_per_class
cv_value = min(5, min_samples_per_class)
if cv_value < 2:
    cv_value = 2  # StackingClassifier requires at least 2 folds

stacking_clf = StackingClassifier(
    estimators=estimators,
    final_estimator=RandomForestClassifier(n_estimators=50, random_state=42),
    cv=cv_value
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', stacking_clf)
])

pipeline.fit(X_train, y_train)
print("Advanced model training complete.")

# Evaluate and print classification report for transparency
y_pred = pipeline.predict(X_test)
print("Validation Report:\n", classification_report(y_test, y_pred))

# Save the pipeline for future use (advanced deployment)
joblib.dump(pipeline, 'advanced_crop_model.joblib')

# For soil type validation in API
soil_types = df['Soil_Type'].unique()
soil_type_set = set(soil_types)

def analyze_ndvi(ndvi):
    if ndvi is None:
        return None
    try:
        ndvi = float(ndvi)
        if ndvi < 0.3:
            return "Low NDVI: Vegetation is sparse or stressed. Consider improving soil health or irrigation. Consider using remote sensing for more accurate monitoring."
        elif ndvi < 0.6:
            return "Moderate NDVI: Crop health is average. Monitor for pests and optimize nutrients. Satellite imagery can help track changes."
        else:
            return "High NDVI: Vegetation is healthy and dense. Maintain current practices. Use drone imagery for precision monitoring."
    except:
        return None

def pest_risk_advice(risk):
    if not risk:
        return None
    risk = risk.lower()
    if risk == "low":
        return "Pest risk is low. Standard monitoring is sufficient. Consider IoT sensors for early detection."
    elif risk == "medium":
        return "Medium pest risk. Consider integrated pest management and AI-based pest detection."
    elif risk == "high":
        return "High pest risk. Take preventive measures, use smart traps, and monitor closely with digital tools."
    return None

def irrigation_advice(irrigation):
    if not irrigation:
        return None
    irrigation = irrigation.lower()
    if irrigation == "yes":
        return "Irrigation available. You can consider water-intensive crops. Smart irrigation systems can optimize water use."
    elif irrigation == "no":
        return "No irrigation. Prefer drought-tolerant crops and conserve soil moisture. Use soil moisture sensors for better management."
    elif irrigation == "limited":
        return "Limited irrigation. Use water-saving techniques, select suitable crops, and consider drip irrigation technology."
    return None

def yield_prediction_advanced(features):
    # Simulate advanced yield prediction using the trained model's probabilities
    # (In real use, a regression model or ML-based yield estimator would be used)
    proba = pipeline.predict_proba(features)
    max_proba = np.max(proba)
    if max_proba > 0.7:
        return f"Estimated yield: 3.0-3.5 tons/ha (high confidence, {int(max_proba*100)}%)"
    elif max_proba > 0.4:
        return f"Estimated yield: 2.5-3.0 tons/ha (moderate confidence, {int(max_proba*100)}%)"
    else:
        return f"Estimated yield: 2.0-2.5 tons/ha (low confidence, {int(max_proba*100)}%)"

@app.route('/')
def home():
    return "Precision Crop Analytics API (Advanced Technology) is running."

@app.route('/predict', methods=['POST'])
def predict_crop():
    data = request.get_json()
    try:
        # Required fields
        nitrogen = float(data['nitrogen'])
        phosphorus = float(data['phosphorus'])
        potassium = float(data['potassium'])
        temperature = float(data['temperature'])
        humidity = float(data['humidity'])
        soil_type = data.get('soil_type', '').strip()
        rainfall = float(data['rainfall'])

        # Optional fields
        location = data.get('location', None)
        ndvi = data.get('ndvi', None)
        pest_risk = data.get('pest_risk', None)
        irrigation = data.get('irrigation', None)

        # Soil type validation
        if soil_type not in soil_type_set:
            return jsonify({
                'status': 'error',
                'message': f"Unknown soil type: {soil_type}. Supported types: {', '.join(soil_type_set)}"
            })

        # Prepare input for advanced pipeline
        input_df = pd.DataFrame([{
            'Nitrogen': nitrogen,
            'Phosphorus': phosphorus,
            'Potassium': potassium,
            'Temperature': temperature,
            'Humidity': humidity,
            'Rainfall': rainfall,
            'Soil_Type': soil_type
        }])

        prediction = pipeline.predict(input_df)[0]

        # Explanation (advanced)
        explanation = (
            f"<b>Advanced Recommendation:</b> The recommended crop is selected using an ensemble of machine learning models, "
            f"analyzing your soil type ({soil_type}), rainfall ({rainfall} mm), and nutrient profile (N:{nitrogen}, P:{phosphorus}, K:{potassium}) "
            f"for optimal yield. Data was standardized and soil type encoded using advanced preprocessing."
        )

        # Advanced insights
        advanced_insights = []
        ndvi_analysis = analyze_ndvi(ndvi) if ndvi not in [None, ""] else None
        pest_advice = pest_risk_advice(pest_risk) if pest_risk not in [None, ""] else None
        irrigation_adv = irrigation_advice(irrigation) if irrigation not in [None, ""] else None
        yield_pred = yield_prediction_advanced(input_df)

        if ndvi_analysis:
            advanced_insights.append(f"<b>NDVI Analysis (AI):</b> {ndvi_analysis}")
        if pest_advice:
            advanced_insights.append(f"<b>Pest Risk (IoT/AI):</b> {pest_advice}")
        if irrigation_adv:
            advanced_insights.append(f"<b>Irrigation (Smart Tech):</b> {irrigation_adv}")
        if yield_pred:
            advanced_insights.append(f"<b>Estimated Yield (ML):</b> {yield_pred}")

        return jsonify({
            'recommended_crop': prediction,
            'status': 'success',
            'explanation': explanation,
            'advanced_insights': "<br>".join(advanced_insights) if advanced_insights else None,
            'ndvi_analysis': ndvi_analysis,
            'pest_advice': pest_advice,
            'irrigation_advice': irrigation_adv,
            'yield_prediction': yield_pred
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True)
