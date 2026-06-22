from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import os

app = Flask(__name__)

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.jinja_env.auto_reload = True

# Load dataset
df = pd.read_csv('location.csv')

# Model path
MODEL_PATH = os.path.join(os.path.dirname(__file__), "Nofeature.pkl")

# Load model safely
pipe = None
try:
    with open(MODEL_PATH, "rb") as f:
        pipe = pickle.load(f)
    print("Model loaded successfully.")
except Exception as e:
    print("Model load error:", e)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        city = request.form.get('city')
        locations = df[df['City'] == city]['Location'].tolist()
        return jsonify(locations)

    cities = df['City'].unique().tolist()
    return render_template('index.html', cities=cities, locations=[])


@app.route('/predict', methods=['POST'])
def predict():

    if pipe is None:
        return "Model file is missing or corrupted.", 500

    location = request.form.get('location')
    area_val = request.form.get('Area')
    bed_val = request.form.get('Bedrooms')
    resale_val = request.form.get('Re-sale', 'No')

    # Validation
    if not location or not area_val or not bed_val:
        return "Please fill all fields.", 400

    try:
        area = float(area_val)
        bedrooms = int(bed_val)
    except ValueError:
        return "Invalid numeric input.", 400

    resale = 1 if resale_val == "Yes" else 0

    input_df = pd.DataFrame(
        [[location, area, bedrooms, resale]],
        columns=['Location', 'Area', 'No_of_Bedrooms', 'Resale']
    )

    try:
        prediction = round(pipe.predict(input_df)[0], 2)
        return jsonify({"prediction": prediction})
    except Exception as e:
        return str(e), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)