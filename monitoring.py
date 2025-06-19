# monitoring.py - Complete version
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import sqlite3
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass
import yfinance as yf
from scipy import stats

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PredictionLog:
    timestamp: datetime
    symbol: str
    prediction: int
    confidence: float
    current_price: float
    features: Dict

@dataclass
class ModelPerformance:
    accuracy: float
    precision: float
    recall: float
    total_predictions: int
    correct_predictions: int
    drift_score: float

class ModelMonitor:
    def __init__(self, db_path='monitoring.db'):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database for storing predictions and performance"""
        conn = sqlite3.connect(self.db_path)
        
        # Create predictions table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                prediction INTEGER NOT NULL,
                confidence REAL NOT NULL,
                current_price REAL NOT NULL,
                features TEXT NOT NULL,
                actual_outcome INTEGER,
                days_to_outcome INTEGER DEFAULT 1
            )
        ''')
        
        # Create performance metrics table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT,
                accuracy REAL,
                precision_score REAL,
                recall_score REAL,
                total_predictions INTEGER,
                correct_predictions INTEGER,
                drift_score REAL,
                notes TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")
    
    def log_prediction(self, symbol: str, prediction: int, confidence: float, 
                      current_price: float, features: Dict):
        """Log a prediction to the database"""
        conn = sqlite3.connect(self.db_path)
        
        prediction_log = PredictionLog(
            timestamp=datetime.now(),
            symbol=symbol,
            prediction=prediction,
            confidence=confidence,
            current_price=current_price,
            features=features
        )
        
        conn.execute('''
            INSERT INTO predictions 
            (timestamp, symbol, prediction, confidence, current_price, features)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            prediction_log.timestamp.isoformat(),
            prediction_log.symbol,
            prediction_log.prediction,
            prediction_log.confidence,
            prediction_log.current_price,
            json.dumps(prediction_log.features)
        ))
        
        conn.commit()
        conn.close()
        logger.info(f"Logged prediction for {symbol}: {prediction} (confidence: {confidence})")
    
    def update_actual_outcomes(self, days_back=7):
        """Check actual outcomes for predictions made N days ago"""
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        conn = sqlite3.connect(self.db_path)
        
        # Get predictions without actual outcomes
        cursor = conn.execute('''
            SELECT id, symbol, prediction, current_price, timestamp
            FROM predictions
            WHERE actual_outcome IS NULL 
            AND timestamp < ?
        ''', (cutoff_date.isoformat(),))
        
        predictions = cursor.fetchall()
        updated_count = 0
        
        for pred_id, symbol, prediction, original_price, timestamp in predictions:
            try:
                # For demo purposes, simulate outcomes
                # In real implementation, you'd fetch actual price data
                pred_date = datetime.fromisoformat(timestamp)
                
                # Simulate random outcome for demo (replace with real data in production)
                actual_outcome = np.random.choice([0, 1], p=[0.4, 0.6])  # Slightly bullish bias
                
                # Update the database
                conn.execute('''
                    UPDATE predictions 
                    SET actual_outcome = ?, days_to_outcome = ?
                    WHERE id = ?
                ''', (actual_outcome, days_back, pred_id))
                
                updated_count += 1
                logger.info(f"Updated outcome for {symbol}: predicted={prediction}, actual={actual_outcome}")
            
            except Exception as e:
                logger.warning(f"Could not update outcome for {symbol}: {e}")
        
        conn.commit()
        conn.close()
        logger.info(f"Updated {updated_count} prediction outcomes")
    
    def calculate_performance_metrics(self, symbol: Optional[str] = None, 
                                    days_back: int = 30) -> ModelPerformance:
        """Calculate model performance metrics"""
        conn = sqlite3.connect(self.db_path)
        
        # Build query
        query = '''
            SELECT prediction, actual_outcome, confidence
            FROM predictions
            WHERE actual_outcome IS NOT NULL
            AND timestamp > ?
        '''
        params = [(datetime.now() - timedelta(days=days_back)).isoformat()]
        
        if symbol:
            query += ' AND symbol = ?'
            params.append(symbol)
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        if len(df) == 0:
            logger.warning("No predictions with outcomes found")
            return ModelPerformance(0, 0, 0, 0, 0, 0)
        
        # Calculate metrics
        correct = (df['prediction'] == df['actual_outcome']).sum()
        total = len(df)
        accuracy = correct / total if total > 0 else 0
        
        # Precision and recall for the positive class (BUY signals)
        true_positives = ((df['prediction'] == 1) & (df['actual_outcome'] == 1)).sum()
        false_positives = ((df['prediction'] == 1) & (df['actual_outcome'] == 0)).sum()
        false_negatives = ((df['prediction'] == 0) & (df['actual_outcome'] == 1)).sum()
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        
        # Simple drift detection based on confidence distribution
        drift_score = self.detect_drift(symbol, days_back)
        
        performance = ModelPerformance(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            total_predictions=total,
            correct_predictions=correct,
            drift_score=drift_score
        )
        
        # Log performance to database
        self.log_performance_metrics(performance, symbol)
        
        return performance
    
    def detect_drift(self, symbol: Optional[str] = None, days_back: int = 30) -> float:
        """Detect model drift by comparing recent vs historical feature distributions"""
        conn = sqlite3.connect(self.db_path)
        
        # Get recent predictions
        recent_query = '''
            SELECT features, confidence
            FROM predictions
            WHERE timestamp > ?
        '''
        params = [(datetime.now() - timedelta(days=days_back)).isoformat()]
        
        if symbol:
            recent_query += ' AND symbol = ?'
            params.append(symbol)
        
        recent_df = pd.read_sql_query(recent_query, conn, params=params)
        
        # Get historical predictions (previous period)
        historical_query = '''
            SELECT features, confidence
            FROM predictions
            WHERE timestamp BETWEEN ? AND ?
        '''
        hist_params = [
            (datetime.now() - timedelta(days=days_back*2)).isoformat(),
            (datetime.now() - timedelta(days=days_back)).isoformat()
        ]
        
        if symbol:
            historical_query += ' AND symbol = ?'
            hist_params.append(symbol)
        
        historical_df = pd.read_sql_query(historical_query, conn, params=hist_params)
        conn.close()
        
        if len(recent_df) < 10 or len(historical_df) < 10:
            return 0.0  # Not enough data for drift detection
        
        # Compare confidence distributions using KS test
        try:
            ks_stat, p_value = stats.ks_2samp(
                recent_df['confidence'].values,
                historical_df['confidence'].values
            )
            
            # Drift score: higher values indicate more drift
            drift_score = ks_stat
            
            if p_value < 0.05:  # Significant drift detected
                logger.warning(f"Potential model drift detected (KS stat: {ks_stat:.3f}, p-value: {p_value:.3f})")
            
            return drift_score
        
        except Exception as e:
            logger.error(f"Error in drift detection: {e}")
            return 0.0
    
    def log_performance_metrics(self, performance: ModelPerformance, symbol: Optional[str]):
        """Log performance metrics to database"""
        conn = sqlite3.connect(self.db_path)
        
        conn.execute('''
            INSERT INTO performance_metrics 
            (timestamp, symbol, accuracy, precision_score, recall_score, 
             total_predictions, correct_predictions, drift_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            symbol,
            performance.accuracy,
            performance.precision,
            performance.recall,
            performance.total_predictions,
            performance.correct_predictions,
            performance.drift_score
        ))
        
        conn.commit()
        conn.close()
        logger.info(f"Performance metrics logged for {symbol or 'all symbols'}")
    
    def get_dashboard_data(self, symbol: Optional[str] = None, days_back: int = 30) -> Dict:
        """Get data for monitoring dashboard"""
        conn = sqlite3.connect(self.db_path)
        
        # Recent predictions
        query = '''
            SELECT symbol, prediction, confidence, timestamp, actual_outcome
            FROM predictions
            WHERE timestamp > ?
            ORDER BY timestamp DESC
            LIMIT 100
        '''
        params = [(datetime.now() - timedelta(days=days_back)).isoformat()]
        
        if symbol:
            query = query.replace('WHERE', 'WHERE symbol = ? AND')
            params.insert(0, symbol)
        
        recent_predictions = pd.read_sql_query(query, conn, params=params)
        
        # Performance over time
        perf_query = '''
            SELECT timestamp, accuracy, precision_score, recall_score, drift_score
            FROM performance_metrics
            WHERE timestamp > ?
            ORDER BY timestamp
        '''
        performance_history = pd.read_sql_query(perf_query, conn, params=[(datetime.now() - timedelta(days=days_back)).isoformat()])
        
        conn.close()
        
        # Calculate summary statistics
        total_predictions = len(recent_predictions)
        predictions_with_outcomes = recent_predictions.dropna(subset=['actual_outcome'])
        
        if len(predictions_with_outcomes) > 0:
            accuracy = (predictions_with_outcomes['prediction'] == predictions_with_outcomes['actual_outcome']).mean()
            avg_confidence = predictions_with_outcomes['confidence'].mean()
        else:
            accuracy = 0
            avg_confidence = recent_predictions['confidence'].mean() if len(recent_predictions) > 0 else 0
        
        # Signal distribution
        signal_distribution = recent_predictions['prediction'].value_counts().to_dict()
        
        dashboard_data = {
            'summary': {
                'total_predictions': total_predictions,
                'predictions_with_outcomes': len(predictions_with_outcomes),
                'accuracy': round(accuracy, 3),
                'average_confidence': round(avg_confidence, 3),
                'signal_distribution': signal_distribution
            },
            'recent_predictions': recent_predictions.to_dict('records'),
            'performance_history': performance_history.to_dict('records'),
            'symbols_analyzed': recent_predictions['symbol'].unique().tolist() if len(recent_predictions) > 0 else []
        }
        
        return dashboard_data
    
    def generate_alert(self, performance: ModelPerformance, symbol: Optional[str] = None) -> Optional[str]:
        """Generate alerts based on performance metrics"""
        alerts = []
        
        if performance.accuracy < 0.5:
            alerts.append(f"Low accuracy detected: {performance.accuracy:.3f}")
        
        if performance.drift_score > 0.3:
            alerts.append(f"High drift score detected: {performance.drift_score:.3f}")
        
        if performance.total_predictions < 10:
            alerts.append("Insufficient prediction data for reliable metrics")
        
        if alerts:
            alert_message = f"Model Alert for {symbol or 'all symbols'}: " + "; ".join(alerts)
            logger.warning(alert_message)
            return alert_message
        
        return None

# Integration with FastAPI
class MonitoringMiddleware:
    def __init__(self, monitor: ModelMonitor):
        self.monitor = monitor
    
    def log_prediction_from_api(self, symbol: str, prediction_result: Dict, features: Dict):
        """Log prediction from API call"""
        self.monitor.log_prediction(
            symbol=symbol,
            prediction=prediction_result['prediction'],
            confidence=prediction_result['confidence'],
            current_price=prediction_result['current_price'],
            features=features
        )
    
    def get_monitoring_dashboard(self, symbol: Optional[str] = None):
        """Get monitoring dashboard data"""
        return self.monitor.get_dashboard_data(symbol=symbol)
    
    def run_daily_checks(self):
        """Run daily monitoring checks"""
        logger.info("Running daily monitoring checks...")
        
        # Update actual outcomes
        self.monitor.update_actual_outcomes(days_back=7)
        
        # Calculate performance metrics
        performance = self.monitor.calculate_performance_metrics(days_back=30)
        
        # Generate alerts if needed
        alert = self.monitor.generate_alert(performance)
        
        if alert:
            # In a real system, you'd send this to Slack, email, etc.
            logger.warning(f"ALERT: {alert}")
        
        logger.info("Daily monitoring checks completed")
        return performance

# Example usage
if __name__ == "__main__":
    # Initialize monitor
    monitor = ModelMonitor()
    
    # Simulate some predictions (in real use, this comes from API calls)
    import random
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']
    
    print("🧪 Testing monitoring system...")
    
    for i in range(20):
        symbol = random.choice(symbols)
        prediction = random.randint(0, 1)
        confidence = random.uniform(0.5, 0.9)
        price = random.uniform(100, 300)
        features = {
            'ma_ratio': random.uniform(0.9, 1.1),
            'rsi': random.uniform(30, 70),
            'volatility': random.uniform(0.01, 0.05)
        }
        
        monitor.log_prediction(symbol, prediction, confidence, price, features)
    
    # Get dashboard data
    dashboard = monitor.get_dashboard_data()
    print("📊 Dashboard Summary:", dashboard['summary'])
    
    # Calculate performance (will be low since we don't have real outcomes)
    performance = monitor.calculate_performance_metrics()
    print(f"📈 Model Performance: Accuracy={performance.accuracy:.3f}, Drift={performance.drift_score:.3f}")
    
    print("✅ Monitoring system test completed!")