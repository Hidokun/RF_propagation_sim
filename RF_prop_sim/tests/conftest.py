"""Pytest configuration file"""
import pytest
import sys
import os

# Add paths so we can import both flat modules (propagation_model) and
# package-style imports (RF_prop_sim.*) used across tests and the UI.
tests_dir = os.path.dirname(os.path.abspath(__file__))
package_dir = os.path.join(tests_dir, '..')
project_root = os.path.join(package_dir, '..')
for p in (project_root, package_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

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