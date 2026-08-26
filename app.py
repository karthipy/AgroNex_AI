from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load trained ML model
model = joblib.load("crop_yield_model.pkl")

# Load trained encoder
encoder = joblib.load("crop_yield_encoder.pkl")

# These must match the columns used during ML training
categorical_columns = [
    "Crop",
    "Soil_Type",
    "Season",
    "Water_Availability",
    "Farming_Method",
    "Crop_Disease_Risk"
]

numerical_columns = [
    "Rainfall_mm",
    "Temperature_C",
    "Humidity",
    "Fertilizer_kg",
    "Irrigation_mm",
    "NPK_Ratio",
    "Sunlight_Hours",
    "Pesticide_kg",
    "Seed_Quality_Score",
    "Soil_Moisture",
    "Previous_Yield_ton",
    "Labor_Hours",
    "Area_Acres",
    "Market_Price_per_ton"
]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    user_input = {
        "Crop": request.form["Crop"],
        "Soil_Type": request.form["Soil_Type"],
        "Season": request.form["Season"],
        "Rainfall_mm": float(request.form["Rainfall_mm"]),
        "Temperature_C": float(request.form["Temperature_C"]),
        "Humidity": float(request.form["Humidity"]),
        "Fertilizer_kg": float(request.form["Fertilizer_kg"]),
        "Irrigation_mm": float(request.form["Irrigation_mm"]),
        "Water_Availability": request.form["Water_Availability"],
        "NPK_Ratio": float(request.form["NPK_Ratio"]),
        "Sunlight_Hours": float(request.form["Sunlight_Hours"]),
        "Pesticide_kg": float(request.form["Pesticide_kg"]),
        "Seed_Quality_Score": float(request.form["Seed_Quality_Score"]),
        "Soil_Moisture": float(request.form["Soil_Moisture"]),
        "Farming_Method": request.form["Farming_Method"],
        "Crop_Disease_Risk": request.form["Crop_Disease_Risk"],
        "Previous_Yield_ton": float(request.form["Previous_Yield_ton"]),
        "Labor_Hours": float(request.form["Labor_Hours"]),
        "Area_Acres": float(request.form["Area_Acres"]),
        "Market_Price_per_ton": float(request.form["Market_Price_per_ton"])
    }

    user_data = pd.DataFrame([user_input])

    # Encode categorical values
    user_encoded = encoder.transform(
        user_data[categorical_columns]
    )

    encoded_df = pd.DataFrame(
        user_encoded.toarray(),
        columns=encoder.get_feature_names_out(
            categorical_columns
        )
    )

    # Combine numerical + encoded categorical data
    user_final = pd.concat(
        [
            user_data[numerical_columns].reset_index(drop=True),
            encoded_df
        ],
        axis=1
    )

    # Make prediction
    prediction = model.predict(user_final)

    predicted_yield = round(prediction[0], 2)

    return render_template(
        "index.html",
        prediction=predicted_yield
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)