import os
import json

from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, StringIndexer, IndexToString
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.pipeline import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

def train():
    try:
        # Initialize SparkSession
        spark = SparkSession.builder.appName('MusicLyricsModel').getOrCreate()

        # Load your dataset
        training_csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), os.path.join(os.getenv('TRAIN_DATA_DIR'), os.getenv('TRAIN_DATA_CSV')))
        # Reading the data
        inputDF = spark.read.option("maxRowsInMemory", 1000000).csv(training_csv_path, inferSchema=True, header=True)

        rm_columns = inputDF.select(['lyrics', 'genre'])

        # Drops the data having null values
        result = rm_columns.na.drop()

        # Tokenize lyrics
        tokenizer = Tokenizer(inputCol="lyrics", outputCol="words")

        # Remove stop words
        stopwords_remover = StopWordsRemover(inputCol=tokenizer.getOutputCol(), outputCol="filtered_words")

        # Convert words to features
        count_vectorizer = CountVectorizer(inputCol=stopwords_remover.getOutputCol(), outputCol="features")

        # StringIndexer to convert genre labels into numerical labels
        label_indexer = StringIndexer(inputCol="genre", outputCol="label")

        # Logistic Regression model
        logistic_regression = LogisticRegression(featuresCol=count_vectorizer.getOutputCol(), labelCol=label_indexer.getOutputCol())

        # Split the data into training and testing sets
        (training_data, testing_data) = result.randomSplit([0.8, 0.2], seed=42)

        # IndexToString for label decoding
        label_converter = IndexToString(inputCol="prediction", outputCol="predicted_genre", labels=label_indexer.fit(training_data).labels)

        # Create a Pipeline
        pipeline = Pipeline(stages=[tokenizer, stopwords_remover, count_vectorizer, label_indexer, logistic_regression, label_converter])

        # Train the model
        fit_model = pipeline.fit(training_data)

        # save the trained model
        # Instead of using save_obj, directly save the model using Spark's save method
        fit_model.write().overwrite().save(os.path.join(os.path.dirname(os.path.dirname(__file__)), os.path.join(os.getenv('MODEL_DIR'), os.getenv('MODEL_FILE'))))

        # Make predictions
        predictions = fit_model.transform(testing_data)

        # Evaluate the model
        evaluator = MulticlassClassificationEvaluator(labelCol=label_indexer.getOutputCol(), predictionCol="prediction", metricName="accuracy")
        accuracy = evaluator.evaluate(predictions)
        print("Accuracy:", accuracy)

        # Stop the SparkSession
        # spark.stop()

        return f"Model accuracy: {accuracy}"

    except Exception as e:
        print(e)
        return f"Error training model: {e}"