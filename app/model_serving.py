"""
FastAPI Model Serving Application
Loads models from MLFlow and provides REST API for inference
"""
import os
import logging
import numpy as np
import pandas as pd
import joblib
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import mlflow
import mlflow.sklearn
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
PREDICTION_COUNTER = Counter('model_predictions_total', 'Total number of predictions')
PREDICTION_LATENCY = Histogram('model_prediction_duration_seconds', 'Prediction latency')
ERROR_COUNTER = Counter('model_prediction_errors_total', 'Total number of prediction errors')

# Pydantic models for API
class PredictionInput(BaseModel):
    """Input schema for predictions"""
    features: List[float] = Field(..., description="List of feature values")
    
    class Config:
        schema_extra = {
            "example": {
                "features": [1.2, -0.5, 0.8, 1.1, -0.2, 0.9, -1.3, 0.4, 0.7, -0.8, 1.5, -0.3, 0.6, 0.2, -1.1]
            }
        }

class PredictionOutput(BaseModel):
    """Output schema for predictions"""
    prediction: int = Field(..., description="Predicted class")
    probability: List[float] = Field(..., description="Class probabilities")
    model_version: str = Field(..., description="Model version used")
    inference_time_ms: float = Field(..., description="Inference time in milliseconds")

class BatchPredictionInput(BaseModel):
    """Input schema for batch predictions"""
    features: List[List[float]] = Field(..., description="List of feature vectors")
    
    class Config:
        schema_extra = {
            "example": {
                "features": [
                    [1.2, -0.5, 0.8, 1.1, -0.2, 0.9, -1.3, 0.4, 0.7, -0.8, 1.5, -0.3, 0.6, 0.2, -1.1],
                    [-0.3, 1.1, -0.7, 0.5, 1.2, -0.4, 0.8, -1.0, 0.3, 1.4, -0.6, 0.9, -0.2, 0.7, 0.1]
                ]
            }
        }

class BatchPredictionOutput(BaseModel):
    """Output schema for batch predictions"""
    predictions: List[int] = Field(..., description="Predicted classes")
    probabilities: List[List[float]] = Field(..., description="Class probabilities for each sample")
    model_version: str = Field(..., description="Model version used")
    batch_size: int = Field(..., description="Number of samples processed")
    total_inference_time_ms: float = Field(..., description="Total inference time in milliseconds")

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    model_version: Optional[str]
    timestamp: str

# Global model container
class ModelContainer:
    def __init__(self):
        self.model = None
        self.model_version = None
        self.feature_names = None
        self.preprocessor = None
        
    def load_model(self):
        """Load model from MLFlow"""
        try:
            # Get environment variables
            mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
            model_name = os.getenv("MODEL_NAME", "classification_model")
            model_stage = os.getenv("MODEL_STAGE", "production")
            
            # Set MLflow tracking URI
            mlflow.set_tracking_uri(mlflow_tracking_uri)
            
            # Load model from MLFlow Model Registry
            model_uri = f"models:/{model_name}/{model_stage}"
            logger.info(f"Loading model from: {model_uri}")
            
            self.model = mlflow.sklearn.load_model(model_uri)
            
            # Get model version
            client = mlflow.tracking.MlflowClient()
            model_version_info = client.get_latest_versions(model_name, stages=[model_stage])[0]
            self.model_version = model_version_info.version
            
            # Try to load preprocessor (if available)
            try:
                preprocessor_artifacts = joblib.load('ml_pipeline/data/processed/features.pkl')
                self.preprocessor = preprocessor_artifacts
                self.feature_names = preprocessor_artifacts.get('feature_names', [])
                logger.info(f"Loaded preprocessor with {len(self.feature_names)} features")
            except Exception as e:
                logger.warning(f"Could not load preprocessor: {e}")
                self.feature_names = [f"feature_{i}" for i in range(15)]  # Default feature names
            
            logger.info(f"Model loaded successfully. Version: {self.model_version}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise e
    
    def predict(self, features: np.ndarray) -> tuple:
        """Make prediction with the loaded model"""
        if self.model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        start_time = time.time()
        
        try:
            # Ensure input is the right shape
            if len(features.shape) == 1:
                features = features.reshape(1, -1)
            
            # Apply preprocessing if available
            if self.preprocessor:
                scaler = self.preprocessor.get('scaler')
                selector = self.preprocessor.get('feature_selector')
                
                if scaler and selector:
                    # Create DataFrame for preprocessing
                    temp_df = pd.DataFrame(features, columns=[f"feature_{i}" for i in range(features.shape[1])])
                    features_scaled = scaler.transform(temp_df)
                    features_selected = selector.transform(features_scaled)
                    features = features_selected
            
            # Make prediction
            prediction = self.model.predict(features)
            probabilities = self.model.predict_proba(features)
            
            inference_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            return prediction, probabilities, inference_time
            
        except Exception as e:
            ERROR_COUNTER.inc()
            logger.error(f"Prediction error: {e}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Initialize FastAPI app
app = FastAPI(
    title="ML Model Serving API",
    description="REST API for machine learning model inference with MLFlow integration",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model container
model_container = ModelContainer()

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    logger.info("Starting model serving application...")
    try:
        model_container.load_model()
        logger.info("Model loaded successfully on startup")
    except Exception as e:
        logger.error(f"Failed to load model on startup: {e}")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    from datetime import datetime
    
    return HealthResponse(
        status="healthy" if model_container.model is not None else "unhealthy",
        model_loaded=model_container.model is not None,
        model_version=model_container.model_version,
        timestamp=datetime.utcnow().isoformat()
    )

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    """Single prediction endpoint"""
    PREDICTION_COUNTER.inc()
    
    with PREDICTION_LATENCY.time():
        try:
            # Convert input to numpy array
            features = np.array(input_data.features)
            
            # Validate input dimensions
            expected_features = len(model_container.feature_names) if model_container.feature_names else 15
            if len(features) != expected_features:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Expected {expected_features} features, got {len(features)}"
                )
            
            # Make prediction
            prediction, probabilities, inference_time = model_container.predict(features)
            
            return PredictionOutput(
                prediction=int(prediction[0]),
                probability=probabilities[0].tolist(),
                model_version=model_container.model_version or "unknown",
                inference_time_ms=inference_time
            )
            
        except HTTPException:
            raise
        except Exception as e:
            ERROR_COUNTER.inc()
            logger.error(f"Prediction error: {e}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionOutput)
async def predict_batch(input_data: BatchPredictionInput):
    """Batch prediction endpoint"""
    PREDICTION_COUNTER.inc(len(input_data.features))
    
    with PREDICTION_LATENCY.time():
        try:
            # Convert input to numpy array
            features = np.array(input_data.features)
            
            # Validate input dimensions
            expected_features = len(model_container.feature_names) if model_container.feature_names else 15
            if features.shape[1] != expected_features:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Expected {expected_features} features, got {features.shape[1]}"
                )
            
            # Make predictions
            predictions, probabilities, inference_time = model_container.predict(features)
            
            return BatchPredictionOutput(
                predictions=predictions.tolist(),
                probabilities=probabilities.tolist(),
                model_version=model_container.model_version or "unknown",
                batch_size=len(input_data.features),
                total_inference_time_ms=inference_time
            )
            
        except HTTPException:
            raise
        except Exception as e:
            ERROR_COUNTER.inc()
            logger.error(f"Batch prediction error: {e}")
            raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.get("/model/info")
async def model_info():
    """Get model information"""
    if model_container.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_version": model_container.model_version,
        "model_type": type(model_container.model).__name__,
        "feature_names": model_container.feature_names,
        "n_features": len(model_container.feature_names) if model_container.feature_names else "unknown",
        "preprocessor_loaded": model_container.preprocessor is not None
    }

@app.post("/model/reload")
async def reload_model():
    """Reload model from MLFlow"""
    try:
        model_container.load_model()
        return {"message": "Model reloaded successfully", "version": model_container.model_version}
    except Exception as e:
        logger.error(f"Failed to reload model: {e}")
        raise HTTPException(status_code=500, detail=f"Model reload failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)