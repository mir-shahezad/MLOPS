# # A simpler Dockerfile
# FROM python:3.9-slim
#
# WORKDIR /app
# RUN pwd
#
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt
#
# COPY ./src /app/src
# # We no longer copy mlruns here, we will mount it instead
#
# EXPOSE 8000
# # The CMD can stay the same, but it will read from the mounted volume
# ENV MLFLOW_TRACKING_URI=file:///app/mlruns
# CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]

#new
# FROM python:3.9-slim
#
# WORKDIR /app
#
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt
#
# COPY ./src /app/src
# COPY ./exported_model /app/model
#
# EXPOSE 8000
#
# CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]

#new 2


FROM python:3.9-slim

WORKDIR /app

# Install system dependencies if needed
RUN apt-get update && apt-get install -y build-essential

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY exported_model/ model/
COPY src/ src/

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
