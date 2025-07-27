# # src/main.py
# import mlflow
# from mlflow.tracking import MlflowClient
# from fastapi import FastAPI
# from pydantic import BaseModel
# import pandas as pd
# import logging
#
# # Configure basic logging
# logging.basicConfig(level=logging.INFO)
#
# # Define the input data schema
# class IrisInput(BaseModel):
#     sepal_length: float
#     sepal_width: float
#     petal_length: float
#     petal_width: float
#
# app = FastAPI()
#
# # --- NEW ROBUST MODEL LOADING LOGIC ---
# model_name = "iris-classifier"
# stage = "Production"
#
# # Set the tracking URI to the correct path *inside the container*.
# # This forces MLflow to look in the right place, ignoring the bad paths in the metadata files.
# #mlflow.set_tracking_uri("file:///app/mlruns")
# mlflow.set_tracking_uri("sqlite:////app/mlflow.db")
# logging.info(f"MLflow tracking URI set to: {mlflow.get_tracking_uri()}")
#
# # 1. Get the latest production model version information
# client = MlflowClient()
# try:
#     latest_version_info = client.get_latest_versions(model_name, stages=[stage])[0]
#     run_id = latest_version_info.run_id
#     logging.info(f"Found production model '{model_name}', version {latest_version_info.version} from run_id: {run_id}")
#
#     # 2. Construct a portable URI to load the model
#     # This format tells MLflow: "look in the current tracking URI for this run_id,
#     # then find the artifact named 'model' inside it."
#     model_uri = f"runs:/{run_id}/model"
#     logging.info(f"Loading model from portable URI: {model_uri}")
#
#     # 3. Load the model
#     model = mlflow.pyfunc.load_model(model_uri)
#     logging.info("Model loaded successfully.")
#
# except IndexError:
#     logging.error(f"No model named '{model_name}' with stage '{stage}' found.")
#     model = None
# # --- END OF NEW LOGIC ---
#
# @app.post("/predict")
# def predict(data: IrisInput):
#     if model is None:
#         return {"error": "Model not loaded. Please check the logs."}, 500
#
#     input_df = pd.DataFrame([data.dict()])
#     # Rename columns if necessary (adjust to your model's features)
#     input_df.columns = [
#         'sepal length (cm)', 'sepal width (cm)',
#         'petal length (cm)', 'petal width (cm)'
#     ]
#     prediction = model.predict(input_df)
#     return {"prediction": int(prediction[0])}
#
# @app.get("/")
# def read_root():
#     return {"message": "Iris Classifier API is running!"}


# src/main.py

from fastapi import FastAPI
import mlflow.pyfunc

app = FastAPI()

# Load model once when container starts
model = mlflow.pyfunc.load_model("model")

@app.get("/")
def read_root():
    return {"message": "Model is up and running"}

@app.post("/predict")
def predict(input: dict):
    import pandas as pd
    input_df = pd.DataFrame([input])
    prediction = model.predict(input_df)
    return {"prediction": prediction.tolist()}

