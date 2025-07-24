# scripts/preprocess.py
import pandas as pd
from sklearn.datasets import load_iris
from pathlib import Path

# Create directories if they don't exist
Path("data/processed").mkdir(parents=True, exist_ok=True)

# Load iris dataset
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Save to a CSV file
df.to_csv("data/processed/iris_processed.csv", index=False)
print("Processed data saved to data/processed/iris_processed.csv")