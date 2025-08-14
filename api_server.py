"""
FastAPI server for Synthetic Healthcare Data Generator
REST API for privacy-preserving medical data synthesis
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
import io
import json
import uuid
import asyncio
from datetime import datetime
import logging

# Import our synthetic data generator
from src.synthetic_healthcare import SyntheticDataGenerator
from src.evaluation.metrics import evaluate_synthetic_data

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Synthetic Healthcare Data Generator API",
    description="Privacy-preserving AI for medical data synthesis",
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

# Global storage for trained models (in production, use Redis/database)
trained_models = {}
generation_jobs = {}

# Pydantic models
class GenerationRequest(BaseModel):
    model_type: str = Field("wgan-gp", description="Model type: wgan-gp, ctgan, beta-vae")
    privacy_level: str = Field("medium", description="Privacy level: low, medium, high")
    compliance_mode: str = Field("hipaa", description="Compliance: hipaa, gdpr, fda")
    n_samples: int = Field(1000, description="Number of samples to generate")
    epochs: int = Field(100, description="Training epochs")

class TrainingRequest(BaseModel):
    model_type: str = Field("wgan-gp", description="Model type")
    privacy_level: str = Field("medium", description="Privacy level")
    compliance_mode: str = Field("hipaa", description="Compliance mode")
    target_column: Optional[str] = Field(None, description="Target column name")
    epochs: int = Field(100, description="Training epochs")

class EvaluationRequest(BaseModel):
    privacy_level: str = Field("medium", description="Privacy level")
    compliance_mode: str = Field("hipaa", description="Compliance mode")

class ModelStatus(BaseModel):
    model_id: str
    status: str
    created_at: datetime
    model_type: str
    privacy_level: str
    compliance_mode: str

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "service": "Synthetic Healthcare Data Generator"
    }

# Get available models
@app.get("/models")
async def get_available_models():
    """Get list of available model types"""
    return {
        "model_types": [
            {
                "name": "wgan-gp",
                "description": "Wasserstein GAN with Gradient Penalty",
                "recommended_for": "General medical data",
                "privacy_support": "High"
            },
            {
                "name": "ctgan",
                "description": "Conditional Tabular GAN",
                "recommended_for": "Mixed categorical/numerical data",
                "privacy_support": "Medium"
            },
            {
                "name": "beta-vae",
                "description": "Beta Variational Autoencoder",
                "recommended_for": "Interpretable latent representation",
                "privacy_support": "High"
            }
        ],
        "privacy_levels": ["low", "medium", "high"],
        "compliance_modes": ["hipaa", "gdpr", "fda"]
    }

# Train a new model
@app.post("/train")
async def train_model(
    file: UploadFile = File(...),
    model_type: str = Form("wgan-gp"),
    privacy_level: str = Form("medium"),
    compliance_mode: str = Form("hipaa"),
    target_column: str = Form(None),
    epochs: int = Form(100)
):
    """Train a new synthetic data generation model"""
    
    try:
        # Generate unique model ID
        model_id = str(uuid.uuid4())
        
        # Read uploaded file
        if file.content_type not in ["text/csv", "application/csv"]:
            raise HTTPException(status_code=400, detail="Only CSV files are supported")
        
        contents = await file.read()
        data = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        logger.info(f"Training model {model_id} with data shape: {data.shape}")
        
        # Initialize generator
        generator = SyntheticDataGenerator(
            model_type=model_type,
            privacy_level=privacy_level,
            compliance_mode=compliance_mode
        )
        
        # Store model info
        trained_models[model_id] = {
            "generator": generator,
            "status": "training",
            "created_at": datetime.now(),
            "model_type": model_type,
            "privacy_level": privacy_level,
            "compliance_mode": compliance_mode,
            "original_data_shape": data.shape
        }
        
        # Train the model (in background for large datasets)
        try:
            generator.fit(data, target_column=target_column, epochs=epochs)
            trained_models[model_id]["status"] = "ready"
            trained_models[model_id]["trained_at"] = datetime.now()
            
        except Exception as e:
            trained_models[model_id]["status"] = "failed"
            trained_models[model_id]["error"] = str(e)
            raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")
        
        return {
            "model_id": model_id,
            "status": "ready",
            "message": f"Model trained successfully on {data.shape[0]} samples",
            "data_shape": data.shape,
            "model_type": model_type,
            "privacy_level": privacy_level,
            "compliance_mode": compliance_mode
        }
        
    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Generate synthetic data
@app.post("/generate/{model_id}")
async def generate_synthetic_data(
    model_id: str,
    n_samples: int = 1000,
    format: str = "json"  # json or csv
):
    """Generate synthetic data using a trained model"""
    
    try:
        # Check if model exists and is ready
        if model_id not in trained_models:
            raise HTTPException(status_code=404, detail="Model not found")
        
        model_info = trained_models[model_id]
        if model_info["status"] != "ready":
            raise HTTPException(status_code=400, detail=f"Model status: {model_info['status']}")
        
        generator = model_info["generator"]
        
        logger.info(f"Generating {n_samples} samples with model {model_id}")
        
        # Generate synthetic data
        synthetic_data = generator.generate(n_samples=n_samples)
        
        # Return in requested format
        if format.lower() == "csv":
            # Return as CSV download
            csv_buffer = io.StringIO()
            synthetic_data.to_csv(csv_buffer, index=False)
            csv_content = csv_buffer.getvalue()
            
            return JSONResponse(
                content={
                    "model_id": model_id,
                    "n_samples": len(synthetic_data),
                    "format": "csv",
                    "data": csv_content,
                    "generated_at": datetime.now().isoformat()
                }
            )
        else:
            # Return as JSON
            return {
                "model_id": model_id,
                "n_samples": len(synthetic_data),
                "format": "json",
                "data": synthetic_data.to_dict(orient="records"),
                "columns": list(synthetic_data.columns),
                "generated_at": datetime.now().isoformat(),
                "privacy_level": model_info["privacy_level"],
                "compliance_mode": model_info["compliance_mode"]
            }
    
    except Exception as e:
        logger.error(f"Generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Evaluate synthetic data
@app.post("/evaluate/{model_id}")
async def evaluate_model(
    model_id: str,
    file: UploadFile = File(...),
    privacy_level: str = Form("medium"),
    compliance_mode: str = Form("hipaa")
):
    """Evaluate synthetic data quality against original data"""
    
    try:
        # Check if model exists
        if model_id not in trained_models:
            raise HTTPException(status_code=404, detail="Model not found")
        
        model_info = trained_models[model_id]
        if model_info["status"] != "ready":
            raise HTTPException(status_code=400, detail=f"Model status: {model_info['status']}")
        
        # Read original data file
        contents = await file.read()
        real_data = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        # Generate synthetic data for evaluation
        generator = model_info["generator"]
        synthetic_data = generator.generate(n_samples=len(real_data))
        
        logger.info(f"Evaluating model {model_id}")
        
        # Run evaluation
        evaluation_results = evaluate_synthetic_data(
            synthetic_data=synthetic_data,
            real_data=real_data,
            privacy_level=privacy_level,
            compliance_mode=compliance_mode
        )
        
        return {
            "model_id": model_id,
            "evaluation_results": evaluation_results,
            "real_data_shape": real_data.shape,
            "synthetic_data_shape": synthetic_data.shape,
            "evaluated_at": datetime.now().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Evaluation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Get model status
@app.get("/models/{model_id}/status")
async def get_model_status(model_id: str):
    """Get status of a specific model"""
    
    if model_id not in trained_models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    model_info = trained_models[model_id]
    
    return {
        "model_id": model_id,
        "status": model_info["status"],
        "created_at": model_info["created_at"].isoformat(),
        "model_type": model_info["model_type"],
        "privacy_level": model_info["privacy_level"],
        "compliance_mode": model_info["compliance_mode"],
        "original_data_shape": model_info.get("original_data_shape"),
        "trained_at": model_info.get("trained_at", "").isoformat() if model_info.get("trained_at") else None,
        "error": model_info.get("error")
    }

# List all models
@app.get("/models/list")
async def list_models():
    """List all trained models"""
    
    models_list = []
    for model_id, model_info in trained_models.items():
        models_list.append({
            "model_id": model_id,
            "status": model_info["status"],
            "created_at": model_info["created_at"].isoformat(),
            "model_type": model_info["model_type"],
            "privacy_level": model_info["privacy_level"],
            "compliance_mode": model_info["compliance_mode"],
            "original_data_shape": model_info.get("original_data_shape")
        })
    
    return {"models": models_list, "total_models": len(models_list)}

# Delete a model
@app.delete("/models/{model_id}")
async def delete_model(model_id: str):
    """Delete a trained model"""
    
    if model_id not in trained_models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    del trained_models[model_id]
    
    return {"message": f"Model {model_id} deleted successfully"}

# Get synthetic data statistics
@app.get("/models/{model_id}/stats")
async def get_model_stats(model_id: str, n_samples: int = 1000):
    """Get statistics of synthetic data from a model"""
    
    try:
        if model_id not in trained_models:
            raise HTTPException(status_code=404, detail="Model not found")
        
        model_info = trained_models[model_id]
        if model_info["status"] != "ready":
            raise HTTPException(status_code=400, detail=f"Model status: {model_info['status']}")
        
        # Generate synthetic data
        generator = model_info["generator"]
        synthetic_data = generator.generate(n_samples=n_samples)
        
        # Calculate statistics
        stats = {
            "shape": synthetic_data.shape,
            "columns": list(synthetic_data.columns),
            "data_types": synthetic_data.dtypes.to_dict(),
            "missing_values": synthetic_data.isnull().sum().to_dict(),
            "numeric_stats": {},
            "categorical_stats": {}
        }
        
        # Numeric column statistics
        numeric_cols = synthetic_data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            stats["numeric_stats"][col] = {
                "mean": float(synthetic_data[col].mean()),
                "std": float(synthetic_data[col].std()),
                "min": float(synthetic_data[col].min()),
                "max": float(synthetic_data[col].max()),
                "median": float(synthetic_data[col].median())
            }
        
        # Categorical column statistics
        categorical_cols = synthetic_data.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            value_counts = synthetic_data[col].value_counts()
            stats["categorical_stats"][col] = {
                "unique_values": int(synthetic_data[col].nunique()),
                "most_common": value_counts.head(5).to_dict(),
                "distribution": (value_counts / len(synthetic_data)).head(10).to_dict()
            }
        
        return {
            "model_id": model_id,
            "statistics": stats,
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Stats error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Batch generation endpoint
@app.post("/batch-generate/{model_id}")
async def batch_generate(
    model_id: str,
    batch_sizes: List[int],
    format: str = "json"
):
    """Generate multiple batches of synthetic data"""
    
    try:
        if model_id not in trained_models:
            raise HTTPException(status_code=404, detail="Model not found")
        
        model_info = trained_models[model_id]
        if model_info["status"] != "ready":
            raise HTTPException(status_code=400, detail=f"Model status: {model_info['status']}")
        
        generator = model_info["generator"]
        
        # Generate batches
        batches = []
        for i, batch_size in enumerate(batch_sizes):
            synthetic_data = generator.generate(n_samples=batch_size)
            
            batch_info = {
                "batch_id": i + 1,
                "n_samples": len(synthetic_data),
                "data": synthetic_data.to_dict(orient="records") if format == "json" else synthetic_data.to_csv(index=False)
            }
            batches.append(batch_info)
        
        return {
            "model_id": model_id,
            "batches": batches,
            "total_batches": len(batches),
            "total_samples": sum(batch_sizes),
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Batch generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Example usage endpoint
@app.get("/example")
async def get_example():
    """Get example usage of the API"""
    
    return {
        "api_usage_example": {
            "1_upload_and_train": {
                "method": "POST",
                "url": "/train",
                "description": "Upload CSV file and train a model",
                "form_data": {
                    "file": "your_medical_data.csv",
                    "model_type": "wgan-gp",
                    "privacy_level": "high",
                    "compliance_mode": "hipaa",
                    "epochs": 100
                }
            },
            "2_generate_data": {
                "method": "POST", 
                "url": "/generate/{model_id}",
                "description": "Generate synthetic data",
                "params": {
                    "n_samples": 1000,
                    "format": "json"
                }
            },
            "3_evaluate_quality": {
                "method": "POST",
                "url": "/evaluate/{model_id}",
                "description": "Evaluate synthetic data quality",
                "form_data": {
                    "file": "original_data.csv",
                    "privacy_level": "high"
                }
            }
        },
        "sample_curl_commands": [
            "curl -X POST '/train' -F 'file=@data.csv' -F 'model_type=wgan-gp' -F 'privacy_level=high'",
            "curl -X POST '/generate/{model_id}?n_samples=1000&format=json'",
            "curl -X POST '/evaluate/{model_id}' -F 'file=@original_data.csv'"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    
    print("🏥 Starting Synthetic Healthcare Data Generator API")
    print("📚 Docs available at: http://localhost:8000/docs")
    print("🔄 API available at: http://localhost:8000")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        reload=True,
        log_level="info"
    )