# test_api.py
import requests
import json
import time
from typing import Dict, List

class TradingAPITester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def test_health_check(self) -> bool:
        """Test the health endpoint"""
        print("🔍 Testing health check...")
        try:
            response = self.session.get(f"{self.base_url}/health")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Health check passed: {data}")
                return data.get('model_loaded', False) and data.get('monitoring_active', False)
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Health check error: {e}")
            return False
    
    def test_single_prediction(self, symbol: str = "AAPL") -> Dict:
        """Test single prediction endpoint"""
        print(f"📈 Testing prediction for {symbol}...")
        try:
            payload = {"symbol": symbol}
            response = self.session.post(
                f"{self.base_url}/predict",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Prediction successful:")
                print(f"   Symbol: {data['symbol']}")
                print(f"   Signal: {data['signal']}")
                print(f"   Confidence: {data['confidence']}")
                print(f"   Current Price: ${data['current_price']}")
                return data
            else:
                print(f"❌ Prediction failed: {response.status_code} - {response.text}")
                return {}
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return {}
    
    def test_batch_prediction(self, symbols: List[str] = ["AAPL", "MSFT", "GOOGL"]) -> Dict:
        """Test batch prediction endpoint"""
        print(f"📊 Testing batch prediction for {symbols}...")
        try:
            payload = {"symbols": symbols}
            response = self.session.post(
                f"{self.base_url}/predict/batch",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Batch prediction successful:")
                print(f"   Total requested: {data['total_requested']}")
                print(f"   Successful: {data['successful']}")
                print(f"   Failed: {data['failed']}")
                
                for prediction in data['predictions']:
                    print(f"   {prediction['symbol']}: {prediction['signal']} ({prediction['confidence']:.3f})")
                
                return data
            else:
                print(f"❌ Batch prediction failed: {response.status_code} - {response.text}")
                return {}
        except Exception as e:
            print(f"❌ Batch prediction error: {e}")
            return {}
    
    def test_monitoring_dashboard(self) -> Dict:
        """Test monitoring dashboard"""
        print("📊 Testing monitoring dashboard...")
        try:
            response = self.session.get(f"{self.base_url}/monitoring/dashboard")
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Dashboard loaded successfully:")
                print(f"   Total predictions: {data['summary']['total_predictions']}")
                print(f"   Average confidence: {data['summary']['average_confidence']:.3f}")
                print(f"   Symbols analyzed: {data['symbols_analyzed']}")
                return data
            else:
                print(f"❌ Dashboard failed: {response.status_code} - {response.text}")
                return {}
        except Exception as e:
            print(f"❌ Dashboard error: {e}")
            return {}
    
    def test_performance_metrics(self) -> Dict:
        """Test performance metrics endpoint"""
        print("📈 Testing performance metrics...")
        try:
            response = self.session.get(f"{self.base_url}/monitoring/performance")
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Performance metrics retrieved:")
                print(f"   Accuracy: {data['accuracy']:.3f}")
                print(f"   Total predictions: {data['total_predictions']}")
                print(f"   Drift score: {data['drift_score']:.3f}")
                return data
            else:
                print(f"❌ Performance metrics failed: {response.status_code} - {response.text}")
                return {}
        except Exception as e:
            print(f"❌ Performance metrics error: {e}")
            return {}
    
    def run_load_test(self, num_requests: int = 10, symbols: List[str] = None):
        """Run a simple load test"""
        if symbols is None:
            symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN"]
        
        print(f"🚀 Running load test with {num_requests} requests...")
        
        start_time = time.time()
        successful_requests = 0
        failed_requests = 0
        
        for i in range(num_requests):
            symbol = symbols[i % len(symbols)]
            try:
                response = self.session.post(
                    f"{self.base_url}/predict",
                    json={"symbol": symbol},
                    timeout=10
                )
                if response.status_code == 200:
                    successful_requests += 1
                else:
                    failed_requests += 1
                    print(f"   Request {i+1} failed: {response.status_code}")
            except Exception as e:
                failed_requests += 1
                print(f"   Request {i+1} error: {e}")
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"✅ Load test completed:")
        print(f"   Total time: {total_time:.2f} seconds")
        print(f"   Requests per second: {num_requests/total_time:.2f}")
        print(f"   Successful: {successful_requests}")
        print(f"   Failed: {failed_requests}")
        print(f"   Success rate: {successful_requests/num_requests*100:.1f}%")
    
    def run_comprehensive_test(self):
        """Run all tests"""
        print("🎯 Starting comprehensive API test suite...\n")
        
        # Test 1: Health check
        if not self.test_health_check():
            print("❌ Health check failed, stopping tests")
            return False
        print()
        
        # Test 2: Single prediction
        self.test_single_prediction("AAPL")
        print()
        
        # Test 3: Batch prediction
        self.test_batch_prediction(["AAPL", "MSFT", "GOOGL"])
        print()
        
        # Test 4: Monitoring dashboard
        self.test_monitoring_dashboard()
        print()
        
        # Test 5: Performance metrics
        self.test_performance_metrics()
        print()
        
        # Test 6: Load test
        self.run_load_test(20)
        print()
        
        print("🎉 All tests completed!")
        return True

def main():
    # Test local deployment
    print("Testing Trading Signal API")
    print("=" * 50)
    
    tester = TradingAPITester("http://localhost:8000")
    
    # Wait for API to be ready
    print("Waiting for API to be ready...")
    max_retries = 30
    for i in range(max_retries):
        try:
            response = requests.get("http://localhost:8000/health", timeout=5)
            if response.status_code == 200:
                print("API is ready!")
                break
        except:
            pass
        time.sleep(2)
        print(f"Waiting... ({i+1}/{max_retries})")
    else:
        print("❌ API not ready after 60 seconds")
        return
    
    # Run tests
    tester.run_comprehensive_test()

if __name__ == "__main__":
    main()