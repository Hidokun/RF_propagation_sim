"""Pytest configuration file"""
import pytest
import sys
import os

# Add the project root to the Python path so we can import RF_prop_sim modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Custom pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers and settings"""
    # Register custom markers
    config.addinivalue_line(
        "markers", "propagation: mark test as propagation model test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "edge_case: mark test as edge case test"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as performance test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running test"
    )
    config.addinivalue_line(
        "markers", "requires_itmlogic: mark test as requiring itmlogic package"
    )
    config.addinivalue_line(
        "markers", "requires_sionna: mark test as requiring sionna-rt package"
    )

def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test location"""
    for item in items:
        # Add markers based on test file location
        if "propagation" in str(item.fspath):
            item.add_marker(pytest.mark.propagation)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "edge_cases" in str(item.fspath):
            item.add_marker(pytest.mark.edge_case)
        elif "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
        
        # Add slow marker to performance tests
        if "performance" in str(item.fspath):
            item.add_marker(pytest.mark.slow)