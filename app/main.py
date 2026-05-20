from fastapi import FastAPI
import joblib

app = FastAPI()

model = joblib.load("model.pkl")

flower_names = ["setosa", "versicolor", "virginica"]

@app.get("/")
def home():
    return {"message": "ML Model API Running"}

@app.get("/predict")
def predict(
    sepal_length: float,
    sepal_width: float,
    petal_length: float,
    petal_width: float
):
    prediction = model.predict([
        [sepal_length, sepal_width, petal_length, petal_width]
    ])

    predicted_class = int(prediction[0])

    return {
        "prediction_number": predicted_class,
        "prediction_name": flower_names[predicted_class]
    }