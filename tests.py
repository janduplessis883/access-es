"""
Simple tests for Access ES calculations
Run with: python tests.py
"""

import pandas as pd
import numpy as np
from utils import calculate_apps_per_1000, calculate_time_metrics, extract_duration_minutes
from config import THRESHOLD_100_PERCENT, THRESHOLD_75_PERCENT, DAYS_PER_WEEK


def test_apps_per_1000_calculation():
    """Test the core calculation: apps per 1000 per week"""
    # Test case 1: 1000 apps, list size 10000, 10 weeks
    # Expected: (1000 / 10000) * 1000 / 10 = 10
    result = calculate_apps_per_1000(total_apps=1000, list_size=10000, weeks=10)
    expected = 10.0
    assert abs(result - expected) < 0.01, f"Expected {expected}, got {result}"
    print(f":material/check: Test 1 passed: apps_per_1000 = {result}")
    
    # Test case 2: 8500 apps, list size 10000, 10 weeks
    # Expected: (8500 / 10000) * 1000 / 10 = 85 (threshold)
    result = calculate_apps_per_1000(total_apps=8500, list_size=10000, weeks=10)
    expected = THRESHOLD_100_PERCENT
    assert abs(result - expected) < 0.01, f"Expected {expected}, got {result}"
    print(f":material/check: Test 2 passed: apps_per_1000 = {result} (100% threshold)")
    
    # Test case 3: 52 weeks, list size 1000
    # Apps needed for 85 per 1000 per week = 85 * 1 * 52
    total_apps = THRESHOLD_100_PERCENT * 1 * 52
    result = calculate_apps_per_1000(total_apps=total_apps, list_size=1000, weeks=52)
    assert abs(result - THRESHOLD_100_PERCENT) < 0.01, f"Expected {THRESHOLD_100_PERCENT}, got {result}"
    print(f":material/check: Test 3 passed: Annual calculation correct")
    

def test_time_metrics_calculation():
    """Test time metrics calculation"""
    # Test: 7 days = 1 week
    from datetime import date
    start_date = date(2025, 1, 1)
    end_date = date(2025, 1, 8)
    
    metrics = calculate_time_metrics((start_date, end_date))
    assert abs(metrics['weeks'] - 1.0) < 0.01, f"Expected 1 week, got {metrics['weeks']}"
    print(f":material/check: Test 4 passed: 7 days = {metrics['weeks']:.2f} weeks")
    
    # Test: 30 days = ~1 month
    end_date = date(2025, 1, 31)
    metrics = calculate_time_metrics((start_date, end_date))
    assert 0.95 < metrics['months'] < 1.05, f"Expected ~1 month, got {metrics['months']}"
    print(f":material/check: Test 5 passed: 30 days = {metrics['months']:.2f} months")


def test_duration_extraction():
    """Test duration extraction from text"""
    # Create test series
    durations = pd.Series(['1h 30m', '45m', '2h', '0m', '1h 15m'])
    
    result = extract_duration_minutes(durations)
    expected = [90, 45, 120, 0, 75]
    
    for i, (r, e) in enumerate(zip(result, expected)):
        assert r == e, f"Duration {i}: Expected {e}, got {r}"
    
    print(f":material/check: Test 6 passed: Duration extraction working correctly")


def test_payment_thresholds():
    """Test payment threshold values"""
    # Verify thresholds are correct
    assert THRESHOLD_100_PERCENT == 85.0, "100% threshold should be 85"
    assert THRESHOLD_75_PERCENT == 75.0, "75% threshold should be 75"
    print(f":material/check: Test 7 passed: Thresholds correct (85/75)")
    
    # Test: What list size needs 850 apps per week for 85 per 1000?
    # 85 = (850 / list_size) * 1000
    # list_size = 850 * 1000 / 85 = 10000
    apps_per_week = 850
    list_size = apps_per_week * 1000 / THRESHOLD_100_PERCENT
    assert abs(list_size - 10000) < 0.01, f"Expected list size 10000, got {list_size}"
    print(f":material/check: Test 8 passed: List size calculation correct")


def test_forecasted_apps_column():
    """Test that aggregation includes forecasted_apps column"""
    from utils import create_weekly_aggregation
    
    # Create sample dataframe
    dates = pd.date_range('2025-01-01', periods=10, freq='D')
    df = pd.DataFrame({
        'appointment_date': dates,
        'count': [5] * 10
    })
    
    # Mock ARRS params
    arrs_params = {
        'estimated_weekly_arrs': 10.0,
        'future_arrs_apps': 100,
        'arrs_future': False
    }
    
    cutoff = pd.Timestamp('2025-02-01')
    result = create_weekly_aggregation(df, list_size=1000, arrs_params=arrs_params, cutoff_date=cutoff)
    
    # Check forecasted_apps column exists
    assert 'forecasted_apps' in result.columns, "forecasted_apps column missing"
    assert 'total_with_forecast' in result.columns, "total_with_forecast column missing"
    print(f":material/check: Test 9 passed: Forecasted columns exist in weekly aggregation")


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print(":material/science: Running Access ES Calculation Tests")
    print("="*60 + "\n")
    
    try:
        test_apps_per_1000_calculation()
        test_time_metrics_calculation()
        test_duration_extraction()
        test_payment_thresholds()
        test_forecasted_apps_column()
        
        print("\n" + "="*60)
        print(":material/done_all: All tests passed successfully!")
        print("="*60 + "\n")
        return True
        
    except AssertionError as e:
        print(f"\n:material/error: Test failed: {e}\n")
        return False
    except Exception as e:
        print(f"\n:material/warning: Error running tests: {e}\n")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
