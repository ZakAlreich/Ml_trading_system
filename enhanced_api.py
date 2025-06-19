# enhanced_api.py - Fixed version
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional, Dict
import joblib
import pandas as pd
import yfinance as yf
from datetime import datetime
import logging
import uvicorn
from trading_model import TradingSignalModel  # Use the fixed model
from monitoring import ModelMonitor, MonitoringMiddleware

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Trading Signal API with Monitoring",
    description="AI-powered trading signal generator with comprehensive monitoring",
    version="2.0.0"
)

# Global instances
trading_model = None
monitor = None
monitoring_middleware = None

# Pydantic models
class PredictionRequest(BaseModel):
    symbol: str
    
class PredictionResponse(BaseModel):
    symbol: str
    current_price: float
    date: str
    prediction: int
    signal: str
    confidence: float
    buy_probability: float
    sell_probability: float

class DashboardResponse(BaseModel):
    summary: Dict
    recent_predictions: List[Dict]
    performance_history: List[Dict]
    symbols_analyzed: List[str]

class PerformanceResponse(BaseModel):
    accuracy: float
    precision: float
    recall: float
    total_predictions: int
    correct_predictions: int
    drift_score: float

# Startup event
@app.on_event("startup")
async def startup_event():
    global trading_model, monitor, monitoring_middleware
    
    try:
        # Try to load existing model
        trading_model = TradingSignalModel()
        trading_model.load_model('trading_model.pkl')
        logger.info("✅ Model loaded successfully")
    except Exception as e:
        logger.warning(f"Failed to load model: {e}. Training new model...")
        trading_model = TradingSignalModel()
        trading_model.train('AAPL')
        trading_model.save_model('trading_model.pkl')
        logger.info("✅ New model trained and saved")
    
    # Initialize monitoring
    monitor = ModelMonitor()
    monitoring_middleware = MonitoringMiddleware(monitor)
    logger.info("✅ Monitoring system initialized")

# Background task for daily monitoring
async def run_daily_monitoring():
    if monitoring_middleware:
        performance = monitoring_middleware.run_daily_checks()
        logger.info(f"Daily monitoring completed. Accuracy: {performance.accuracy:.3f}")

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "🚀 Trading Signal API with Monitoring",
        "version": "2.0.0",
        "status": "🟢 Online",
        "features": [
            "🤖 AI Predictions", 
            "📊 Real-time Monitoring", 
            "📈 Performance Tracking", 
            "🔍 Drift Detection"
        ],
        "endpoints": {
            "/predict": "POST - Get trading signal",
            "/predict/batch": "POST - Get signals for multiple stocks",
            "/monitoring/dashboard": "GET - Monitoring dashboard",
            "/monitoring/performance": "GET - Performance metrics",
            "/health": "GET - Health check",
            "/docs": "GET - API documentation"
        }
    }

# Enhanced prediction endpoint with monitoring
@app.post("/predict", response_model=PredictionResponse)
async def predict_signal(request: PredictionRequest, background_tasks: BackgroundTasks):
    if not trading_model or not trading_model.is_trained:
        raise HTTPException(status_code=503, detail="Model not loaded or trained")
    
    try:
        symbol = request.symbol.upper().strip()
        logger.info(f"📈 Predicting signal for {symbol}")
        
        # Get prediction
        result = trading_model.predict_signal(symbol)
        
        # Log prediction for monitoring (run in background)
        if monitoring_middleware:
            features = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'model_version': '2.0'
            }
            
            background_tasks.add_task(
                monitoring_middleware.log_prediction_from_api,
                symbol=symbol,
                prediction_result=result,
                features=features
            )
        
        logger.info(f"✅ Prediction for {symbol}: {result['signal']} (confidence: {result['confidence']})")
        return PredictionResponse(**result)
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"❌ Prediction error for {symbol}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Batch prediction endpoint
@app.post("/predict/batch")
async def predict_batch(request: dict):
    symbols = request.get("symbols", [])
    
    if not trading_model or not trading_model.is_trained:
        raise HTTPException(status_code=503, detail="Model not loaded or trained")
    
    if len(symbols) > 20:  # Limit batch size
        raise HTTPException(status_code=400, detail="Maximum 20 symbols per batch")
    
    results = []
    errors = []
    
    for symbol in symbols:
        try:
            symbol = symbol.upper().strip()
            result = trading_model.predict_signal(symbol)
            results.append(result)
        except Exception as e:
            errors.append({"symbol": symbol, "error": str(e)})
    
    return {
        "predictions": results,
        "errors": errors,
        "total_requested": len(symbols),
        "successful": len(results),
        "failed": len(errors)
    }

# Monitoring dashboard endpoint
@app.get("/monitoring/dashboard", response_model=DashboardResponse)
async def get_monitoring_dashboard(symbol: Optional[str] = None):
    if not monitoring_middleware:
        raise HTTPException(status_code=503, detail="Monitoring not available")
    
    try:
        dashboard_data = monitoring_middleware.get_monitoring_dashboard(symbol)
        return DashboardResponse(**dashboard_data)
    except Exception as e:
        logger.error(f"❌ Dashboard error: {e}")
        raise HTTPException(status_code=500, detail="Dashboard unavailable")

# Performance metrics endpoint
@app.get("/monitoring/performance", response_model=PerformanceResponse)
async def get_performance_metrics(symbol: Optional[str] = None, days_back: int = 30):
    if not monitor:
        raise HTTPException(status_code=503, detail="Monitoring not available")
    
    try:
        # Update actual outcomes first
        monitor.update_actual_outcomes(days_back=7)
        
        # Calculate performance
        performance = monitor.calculate_performance_metrics(symbol, days_back)
        
        return PerformanceResponse(
            accuracy=performance.accuracy,
            precision=performance.precision,
            recall=performance.recall,
            total_predictions=performance.total_predictions,
            correct_predictions=performance.correct_predictions,
            drift_score=performance.drift_score
        )
    except Exception as e:
        logger.error(f"❌ Performance calculation error: {e}")
        raise HTTPException(status_code=500, detail="Performance calculation failed")

# Trigger daily monitoring manually
@app.post("/monitoring/run-daily-check")
async def trigger_daily_monitoring():
    if not monitoring_middleware:
        raise HTTPException(status_code=503, detail="Monitoring not available")
    
    try:
        performance = monitoring_middleware.run_daily_checks()
        return {
            "status": "completed",
            "performance": {
                "accuracy": performance.accuracy,
                "total_predictions": performance.total_predictions,
                "drift_score": performance.drift_score
            }
        }
    except Exception as e:
        logger.error(f"❌ Daily monitoring error: {e}")
        raise HTTPException(status_code=500, detail="Daily monitoring failed")

# Health check with monitoring status
@app.get("/health")
async def health_check():
    model_status = trading_model is not None and trading_model.is_trained
    monitoring_status = monitor is not None
    
    return {
        "status": "🟢 healthy" if model_status and monitoring_status else "🟡 degraded",
        "model_loaded": model_status,
        "monitoring_active": monitoring_status,
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0"
    }

# Model retraining endpoint
@app.post("/model/retrain")
async def retrain_model(background_tasks: BackgroundTasks, symbol: str = "AAPL"):
    global trading_model
    
    try:
        logger.info(f"🔄 Retraining model with {symbol} data...")
        
        def retrain_task():
            global trading_model
            trading_model = TradingSignalModel()
            trading_model.train(symbol)
            trading_model.save_model('trading_model.pkl')
            logger.info("✅ Model retrained successfully")
        
        background_tasks.add_task(retrain_task)
        
        return {"status": "started", "message": f"Model retraining initiated with {symbol} data"}
    except Exception as e:
        logger.error(f"❌ Retraining failed: {e}")
        raise HTTPException(status_code=500, detail=f"Retraining failed: {str(e)}")

if __name__ == "__main__":
    print("🚀 Starting Trading Signal API...")
    uvicorn.run(
        "enhanced_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )