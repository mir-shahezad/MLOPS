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

# from fastapi import FastAPI
# import mlflow.pyfunc
#
# app = FastAPI()
#
# # Load model once when container starts
# model = mlflow.pyfunc.load_model("model")
#
# @app.get("/")
# def read_root():
#     return {"message": "Model is up and running"}
#
# @app.post("/predict")
# def predict(input: dict):
#     import pandas as pd
#     input_df = pd.DataFrame([input])
#     prediction = model.predict(input_df)
#     return {"prediction": prediction.tolist()}
#####################################################################
# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# import mlflow.sklearn
# import pandas as pd
# import logging
#
# # Logging setup
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
#
# app = FastAPI()
#
# # === Define Input Schema ===
# class IrisInput(BaseModel):
#     sepal_length: float
#     sepal_width: float
#     petal_length: float
#     petal_width: float
#
# # === Load Model ===
# MODEL_PATH = "exported_model"
# try:
#     model = mlflow.sklearn.load_model(MODEL_PATH)
#     logger.info("Model loaded successfully.")
# except Exception as e:
#     logger.error(f"Model loading failed: {e}")
#     model = None
#
# @app.get("/")
# def health_check():
#     return {"message": "Iris Classifier API is running!"}
#
# @app.post("/predict")
# def predict(input: IrisInput):
#     if model is None:
#         raise HTTPException(status_code=500, detail="Model not loaded.")
#
#     input_df = pd.DataFrame([input.dict()])
#     input_df.columns = [
#         'sepal_length', 'sepal_width',
#         'petal_length', 'petal_width'
#     ]
#
#     prediction = model.predict(input_df)
#     return {"prediction": int(prediction[0])}
##########################################################################################
from fastapi import FastAPI, Request
from pydantic import BaseModel
import mlflow.pyfunc
import pandas as pd
import logging
import time
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response

# Set up logging
logging.basicConfig(level=logging.INFO)

# Initialize FastAPI app
app = FastAPI()

# Load the model on startup
#model = mlflow.pyfunc.load_model("model")
MODEL_PATH = "exported_model"
model = mlflow.sklearn.load_model(MODEL_PATH)
# --- Prometheus Metrics ---
REQUEST_COUNT = Counter("predict_requests_total", "Total prediction requests")
REQUEST_LATENCY = Histogram("predict_request_latency_seconds", "Prediction latency in seconds")

# --- Input Schema ---
# class InputData(BaseModel):
#     sepal_length: float
#     sepal_width: float
#     petal_length: float
#     petal_width: float

# --- Routes ---
@app.get("/")
def read_root():
    return {"message": "Model is up and running"}

@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

# @app.post("/predict")
# def predict(input: InputData):
#     start_time = time.time()
#     REQUEST_COUNT.inc()
#
#     # Convert input to dataframe
#     input_dict = input.dict()
#     logging.info(f"Received input: {input_dict}")
#     input_df = pd.DataFrame([input_dict])
#
#     # Prediction
#     prediction = model.predict(input_df)
#     result = int(prediction[0])
#     logging.info(f"Prediction result: {result}")
#
#     REQUEST_LATENCY.observe(time.time() - start_time)
#     return {"prediction": result}

# @app.post("/predict")
# def predict(input):
#     # # Map keys from short to original feature names
#     # key_map = {
#     #     "sepal_length": "sepal length (cm)",
#     #     "sepal_width": "sepal width (cm)",
#     #     "petal_length": "petal length (cm)",
#     #     "petal_width": "petal width (cm)"
#     # }
#     #
#     # remapped_input = {key_map.get(k, k): v for k, v in input.items()}
#     # input_df = pd.DataFrame([remapped_input])
#     if model is None:
#         raise HTTPException(status_code=500, detail="Model not loaded.")
#
#     input_df = pd.DataFrame([input.dict()])
#     input_df.columns = [
#         'sepal_length', 'sepal_width',
#         'petal_length', 'petal_width'
#     ]
#
#     prediction = model.predict(input_df)
#     return {"prediction": int(prediction[0])}
@app.post("/predict")
def predict(input: dict):
    import pandas as pd
    start_time = time.time()
    REQUEST_COUNT.inc()
    input_df = pd.DataFrame([input])
    prediction = model.predict(input_df)
    result = int(prediction[0])
    logging.info(f"Prediction result: {result}")
    REQUEST_LATENCY.observe(time.time() - start_time)
    return {"prediction": prediction.tolist()}