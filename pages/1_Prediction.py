import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

st.title("🌧 Rainfall & Flood Prediction App")

file_path = 'rainfall_analysis_output.csv'
data = pd.read_csv(file_path)

data.columns = data.columns.str.strip()

data['rainfall'] = data['rainfall'].map({'yes': 1, 'no': 0})

# Numerical features
numerical_features = ['temparature', 'humidity', 'windspeed', 'pressure']

numerical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])
preprocessor = ColumnTransformer([
    ('num', numerical_transformer, numerical_features)
])

X = data[numerical_features]
y = data['intensity']

X_preprocessed = preprocessor.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42)

regressor = RandomForestRegressor(n_estimators=100, random_state=42)
regressor.fit(X_train, y_train)

X_flood = data[['rainfall', 'intensity', 'temparature', 'humidity', 'windspeed', 'pressure']]
y_flood = data['flood_occurence']
X_flood_train, X_flood_test, y_flood_train, y_flood_test = train_test_split(X_flood, y_flood, test_size=0.2, random_state=42)

classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_flood_train, y_flood_train)

st.sidebar.header("🌡 Enter Weather Conditions")
temperature = st.sidebar.number_input("Temperature (°C)", min_value=-10.0, max_value=50.0, value=25.0)
humidity = st.sidebar.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=60.0)
windspeed = st.sidebar.number_input("Wind Speed (km/h)", min_value=0.0, max_value=200.0, value=10.0)
pressure = st.sidebar.number_input("Pressure (hPa)", min_value=900.0, max_value=1100.0, value=1013.0)

if st.sidebar.button("Predict Rainfall & Flood Risk"):
    user_input_df = pd.DataFrame([[temperature, humidity, windspeed, pressure]], columns=numerical_features)
    user_input_preprocessed = preprocessor.transform(user_input_df)
    
    predicted_intensity = regressor.predict(user_input_preprocessed)[0]
    rainfall_occurrence = 1 if predicted_intensity > 3 else 0

    flood_input = pd.DataFrame([[rainfall_occurrence, predicted_intensity, temperature, humidity, windspeed, pressure]], 
                               columns=['rainfall', 'intensity', 'temparature', 'humidity', 'windspeed', 'pressure'])
    flood_prediction = classifier.predict(flood_input)[0]

    advisory = "⚠️ High rainfall detected. Consider drainage systems." if predicted_intensity > 5 else \
               "💧 Low rainfall detected. Consider irrigation techniques." if predicted_intensity < 2 else \
               "🌾 Moderate rainfall detected. Suitable for most crops."

    rainfall_status = "🌧 Yes" if rainfall_occurrence else "☀️ No"
    flood_status = "⚠️ High(Moderate) Risk" if flood_prediction else "✅ Low Risk"

    st.subheader("🌤 Prediction Results")
    st.write(f"**Predicted Rainfall Intensity:** {predicted_intensity:.2f}")
    st.write(f"**Will it Rain?** {rainfall_status}")
    st.write(f"**Flood Risk:** {flood_status}")
    st.write(f"**Advisory:** {advisory}")

    # Regression visualization with colors for flood occurrence
    st.subheader("📊 Rainfall Intensity vs. Temperature")
    fig, ax = plt.subplots(figsize=(8, 5))

    # Mark high rainfall
    heavy_rain = data[data['intensity'] > 5]
    moderate_rain = data[(data['intensity'] <= 5) & (data['intensity'] >= 2)]
    low_rain = data[data['intensity'] < 2]

    ax.scatter(low_rain['temparature'], low_rain['intensity'], color='blue', label='Low Rainfall', alpha=0.6)
    ax.scatter(moderate_rain['temparature'], moderate_rain['intensity'], color='green', label='Moderate Rainfall', alpha=0.6)
    ax.scatter(heavy_rain['temparature'], heavy_rain['intensity'], color='red', label='Heavy Rainfall', alpha=0.6)

    # Mark flood occurrences
    flooded = data[data['flood_occurence'] == 1]
    ax.scatter(flooded['temparature'], flooded['intensity'], edgecolors='black', facecolors='none', label='Flood Occurrence', linewidths=1.5)

    plt.xlabel("Temperature (°C)")
    plt.ylabel("Rainfall Intensity")
    plt.title("Rainfall Intensity vs. Temperature")
    plt.legend()
    st.pyplot(fig)
