# trading_model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import yfinance as yf
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class TradingSignalModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.feature_columns = None
        self.is_trained = False
    
    def create_features(self, df):
        """
        Create technical indicators as features
        """
        # Price-based features
        df['price_change'] = df['Close'].pct_change()
        df['high_low_ratio'] = df['High'] / df['Low']
        df['volume_change'] = df['Volume'].pct_change()
        
        # Moving averages
        df['ma_5'] = df['Close'].rolling(window=5).mean()
        df['ma_20'] = df['Close'].rolling(window=20).mean()
        df['ma_ratio'] = df['ma_5'] / df['ma_20']
        
        # Volatility (standard deviation)
        df['volatility_5'] = df['price_change'].rolling(window=5).std()
        df['volatility_20'] = df['price_change'].rolling(window=20).std()
        
        # RSI (Relative Strength Index) - simplified version
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['bb_upper'] = df['ma_20'] + (df['Close'].rolling(window=20).std() * 2)
        df['bb_lower'] = df['ma_20'] - (df['Close'].rolling(window=20).std() * 2)
        df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        return df
    
    def create_target(self, df, days_ahead=1):
        """
        Create target: 1 if price goes up in next 'days_ahead' days, 0 otherwise
        """
        df['future_price'] = df['Close'].shift(-days_ahead)
        df['target'] = (df['future_price'] > df['Close']).astype(int)
        return df
    
    def prepare_data(self, symbol='AAPL', period='2y'):
        """
        Download data and prepare features
        """
        print(f"Downloading data for {symbol}...")
        
        # Download stock data
        stock = yf.Ticker(symbol)
        df = stock.history(period=period)
        
        if df.empty:
            raise ValueError(f"No data found for symbol {symbol}")
        
        # Create features and target
        df = self.create_features(df)
        df = self.create_target(df)
        
        # Select feature columns (exclude NaN-prone columns)
        feature_cols = [
            'price_change', 'high_low_ratio', 'volume_change',
            'ma_ratio', 'volatility_5', 'volatility_20', 
            'rsi', 'bb_position'
        ]
        
        # Clean data
        df = df.dropna()
        
        # Store feature columns for later use
        self.feature_columns = feature_cols
        
        X = df[feature_cols]
        y = df['target']
        
        print(f"Prepared {len(df)} samples with {len(feature_cols)} features")
        print(f"Target distribution: {y.value_counts().to_dict()}")
        
        return X, y, df
    
    def train(self, symbol='AAPL', test_size=0.2):
        """
        Train the model
        """
        X, y, df = self.prepare_data(symbol)
        
        # Split data chronologically (important for time series!)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print("Training model...")
        self.model.fit(X_train, y_train)
        
        # Evaluate
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        print(f"Training accuracy: {train_score:.3f}")
        print(f"Testing accuracy: {test_score:.3f}")
        
        # Detailed evaluation
        y_pred = self.model.predict(X_test)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nFeature Importance:")
        print(feature_importance)
        
        self.is_trained = True
        return self
    
    def predict_signal(self, symbol='AAPL', days=5):
        """
        Predict trading signal for recent data
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first!")
        
        # Get recent data
        stock = yf.Ticker(symbol)
        df = stock.history(period='1mo')  # Get last month of data
        
        # Create features
        df = self.create_features(df)
        df = df.dropna()
        
        if len(df) == 0:
            raise ValueError("No valid data available for prediction")
        
        # Get the most recent data point
        latest_features = df[self.feature_columns].iloc[-1:].values
        
        # Predict
        prediction = self.model.predict(latest_features)[0]
        probability = self.model.predict_proba(latest_features)[0]
        
        # Get current price info
        current_price = df['Close'].iloc[-1]
        current_date = df.index[-1]
        
        result = {
            'symbol': symbol,
            'current_price': round(current_price, 2),
            'date': current_date.strftime('%Y-%m-%d'),
            'prediction': int(prediction),
            'signal': 'BUY' if prediction == 1 else 'SELL',
            'confidence': round(max(probability), 3),
            'buy_probability': round(probability[1], 3),
            'sell_probability': round(probability[0], 3)
        }
        
        return result
    
    def save_model(self, filepath='trading_model.pkl'):
        """
        Save the trained model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first!")
        
        model_data = {
            'model': self.model,
            'feature_columns': self.feature_columns,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='trading_model.pkl'):
        """
        Load a pre-trained model
        """
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.feature_columns = model_data['feature_columns']
        self.is_trained = model_data['is_trained']
        print(f"Model loaded from {filepath}")

# Example usage and training
if __name__ == "__main__":
    # Create and train model
    trading_model = TradingSignalModel()
    
    # Train on Apple stock
    trading_model.train('AAPL')
    
    # Save the model
    trading_model.save_model('trading_model.pkl')
    
    # Make predictions
    signal = trading_model.predict_signal('AAPL')
    print(f"\nTrading Signal: {signal}")
    
    # Test on different stocks
    for symbol in ['MSFT', 'GOOGL', 'TSLA']:
        try:
            signal = trading_model.predict_signal(symbol)
            print(f"{symbol}: {signal['signal']} (confidence: {signal['confidence']})")
        except Exception as e:
            print(f"Error with {symbol}: {e}")