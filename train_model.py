import pandas as pd
import numpy as np
import pickle

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

# Load locations
loc_df = pd.read_csv("location.csv")
locations = loc_df["Location"].unique().tolist()

# Generate synthetic data
rng = np.random.default_rng(42)
samples = []

for loc in locations:
    rate = rng.uniform(2000, 10000)

    for _ in range(50):   # 150 se kam karke 50 kar diya
        area = int(rng.integers(300, 4000))
        bedrooms = int(rng.integers(1, 5))
        resale = int(rng.choice([0, 1]))

        price = (
            area * rate
            + bedrooms * rng.uniform(100000, 300000)
            - resale * rng.uniform(20000, 100000)
            + rng.normal(0, 200000)
        )

        price = max(price, 500000)

        samples.append({
            "Location": loc,
            "Area": area,
            "No_of_Bedrooms": bedrooms,
            "Resale": resale,
            "Price": round(price, 2)
        })

df = pd.DataFrame(samples)

X = df[["Location", "Area", "No_of_Bedrooms", "Resale"]]
y = df["Price"]

# Preprocessing
column_trans = make_column_transformer(
    (
        OneHotEncoder(handle_unknown="ignore"),
        ["Location"]
    ),
    remainder="passthrough"
)

# Small model (memory kam lega)
model = RandomForestRegressor(
    n_estimators=20,
    max_depth=10,
    random_state=42
)

pipe = make_pipeline(column_trans, model)

print("Training model...")
pipe.fit(X, y)

# Save model
with open("Nofeature.pkl", "wb") as f:
    pickle.dump(pipe, f)

print("Nofeature.pkl created successfully!")