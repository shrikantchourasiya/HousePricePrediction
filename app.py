from flask import Flask,render_template,request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.jinja_env.auto_reload = True

df=pd.read_csv('location.csv')
pipe = None

@app.route('/', methods=['GET', 'POST']) 

def index():
    if request.method == 'POST':
        # Get the selected city from the form
        city = request.form['city']
        # Get a list of locations for the selected city
        locations = df[df['City'] == city]['Location'].tolist()
        # Return the locations as a JSON response
        return jsonify(locations)
    else:
        # Get a list of all the cities
        cities = df['City'].unique().tolist()
        locations = []
        return render_template('index.html', cities=cities, locations=locations)


@app.route('/predict',methods=['POST'])
def predict():
    global pipe
    # Ensure model is loaded (lazy load to pick up updated pickle)
    if pipe is None:
        try:
            pipe = pickle.load(open('Nofeature.pkl','rb'))
        except Exception as e:
            return str(e), 500

    location=request.form.get('location')
    try:
        area=float(request.form.get('Area'))
        Bedrooms=int(request.form.get('Bedrooms'))
    except Exception as e:
        return f'Invalid input: {e}', 400
    Resale=request.form.get('Re-sale')
    if Resale=='Yes':
        Resale=1
    else:
        Resale=0

    input=pd.DataFrame([[location, area, Bedrooms, Resale]],columns=['Location','Area', 'No_of_Bedrooms', 'Resale'])
    try:
        prediction=round(pipe.predict(input)[0],2)
        return str(prediction)
    except Exception as e:
        return str(e), 500

if __name__=="__main__":
    app.run(debug=True,port=5001)


