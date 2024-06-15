import os

from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.feature import Tokenizer, HashingTF, IDF
import base64
from io import BytesIO
import matplotlib.pyplot as plt

# Initialize Spark session
spark = SparkSession.builder.appName('MusicLyricsPredictor').getOrCreate()

def visualize(probabilites: dict):
    print('Generating pie chart')
    # Generate pie chart
    plt.figure(figsize=(8, 6))
    plt.pie(probabilites.values(), labels=probabilites.keys(), autopct='%1.1f%%')
    plt.title('Song Genre Probabilities')
    
    # Convert chart to base64-encoded image
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    chart_url = base64.b64encode(buffer.getvalue()).decode('utf-8')
    chart_img = f'data:image/png;base64,{chart_url}'
    
    print('Pie chart generated')
    
    return chart_img

# Function to classify lyrics
def classify_lyrics(lyrics):
    print('Loading model')
    # Load the model using Spark's load method
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), os.path.join(os.getenv('MODEL_DIR'), os.getenv('MODEL_FILE')))
    model = PipelineModel.load(model_path)
    
    print('Model loaded & extracting features')

    # Tokenize lyrics
    tokenizer = Tokenizer(inputCol="lyrics", outputCol="inputWords")
    words_df = spark.createDataFrame([(lyrics,)], ["lyrics"])
    words = tokenizer.transform(words_df)

    # Feature engineering (using TF-IDF)
    hashingTF = HashingTF(inputCol="inputWords", outputCol="inputFeatures")
    idf = IDF(inputCol="inputFeatures", outputCol="tfidf_features")
    features = idf.fit(hashingTF.transform(words)).transform(hashingTF.transform(words))
    
    print('Features extracted')

    # Predict genre
    predictions = model.transform(features)
    genre = predictions.select("predicted_genre").collect()[0][0]
    print(f'Prediction complete: Prediction: {genre}')
    
    # Extract probabilities
    probabilities = predictions.select("probability").collect()[0][0]
    
    # Map probabilities to genre labels
    genre_labels = model.stages[-3].labels
    genre_probabilities = {genre_labels[i]: float(probabilities[i]) for i in range(len(genre_labels))}
    
    chart_img = visualize(genre_probabilities)

    return genre, chart_img


def predict(lyrics: str):
    try:
        predicted_genre, chart_img = classify_lyrics(lyrics)
        print("Predicted Genre:", predicted_genre)
        
        return predicted_genre, chart_img
    except Exception as e:
        print(e)
        return f'Error predicting genre: {e}'

