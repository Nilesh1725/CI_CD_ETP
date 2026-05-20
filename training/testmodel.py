import joblib
from sklearn.datasets import load_iris

model = joblib.load("app/model.pkl")

data = load_iris()

X = data.data
y = data.target

accuracy = model.score(X, y)

print("Accuracy:", accuracy)

assert accuracy > 0.8