import os
from flask import Flask, request, render_template
from dotenv import load_dotenv

# defined imports
from .routes.predict_genre import predict
from .routes.train_model import train

# Load environment variables from .env file
load_dotenv('../.env')

# Application code-------------------
app = Flask(__name__)
app.config['ENV'] = os.getenv('FLASK_ENV', 'production')
if app.config['ENV'] == 'development':
    app.config['DEBUG'] = True

# Endpoint definitions
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train', methods=['GET'])
def train_model():
    train()
    return 'Training model'

@app.route('/predict', methods=['POST'])
def predict_genre_from_lyrics():
    lyrics = request.form['lyrics']
    prediction, chart_img = predict(lyrics)
    template_vars = {
        'prediction': prediction,
        'chart_img': chart_img,
        'lyrics': lyrics
    }
    
    return render_template('prediction_results.html', **template_vars)

# Start server
if __name__ == '__main__':
    app.run(debug=app.config['DEBUG'])

