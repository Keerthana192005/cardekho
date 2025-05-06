import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder

# Title
st.title("Car Price Prediction App")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv('cardekho_imputated.csv', index_col=0)
    le = LabelEncoder()
    df['model'] = le.fit_transform(df['model'])
    df = pd.get_dummies(df, drop_first=False)
    return df

df = load_data()

# Split data
X = df.drop('selling_price', axis=1)
y = df['selling_price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)
r2 = r2_score(y_test, model.predict(X_test))
st.write(f"Model R² score: {r2:.2f}")

# Input form
st.subheader("Predict Car Price")

def get_input():
    vehicle_age = st.number_input('Vehicle Age', min_value=0, max_value=30, value=3)
    km_driven = st.number_input('KM Driven', value=25000)
    mileage = st.number_input('Mileage (km/l)', value=20.5)
    engine = st.number_input('Engine (CC)', value=1200)
    max_power = st.number_input('Max Power (BHP)', value=85)
    seats = st.number_input('Seats', min_value=2, max_value=10, value=5)

    brand = st.selectbox('Brand', sorted([col.replace('brand_', '') for col in X_train.columns if col.startswith('brand_')]))
    seller_type = st.selectbox('Seller Type', sorted([col.replace('seller_type_', '') for col in X_train.columns if col.startswith('seller_type_')]))
    fuel_type = st.selectbox('Fuel Type', sorted([col.replace('fuel_type_', '') for col in X_train.columns if col.startswith('fuel_type_')]))
    transmission = st.selectbox('Transmission', sorted([col.replace('transmission_type_', '') for col in X_train.columns if col.startswith('transmission_type_')]))

    # Build input dataframe
    input_data = {
        'vehicle_age': vehicle_age,
        'km_driven': km_driven,
        'mileage': mileage,
        'engine': engine,
        'max_power': max_power,
        'seats': seats
    }

    # Add one-hot encoded columns
    for col in X_train.columns:
        if col not in input_data:
            input_data[col] = 0

    input_data[f'brand_{brand}'] = 1
    input_data[f'seller_type_{seller_type}'] = 1
    input_data[f'fuel_type_{fuel_type}'] = 1
    input_data[f'transmission_type_{transmission}'] = 1

    return pd.DataFrame([input_data])[X_train.columns]

if st.button("Predict"):
    input_df = get_input()
    prediction = model.predict(input_df)[0]
    st.success(f"Estimated Selling Price: ₹{prediction:,.2f}")
