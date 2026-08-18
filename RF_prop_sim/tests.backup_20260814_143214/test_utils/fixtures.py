"""Common test fixtures and data for RF simulator testing"""
import numpy as np

# Sample valid configurations for testing
SAMPLE_VALID_CONFIGS = {
    'fspl': {
        'model': 'fspl',
        'frequency_mhz': 300.0,
        'distance_km': 5.0,
        'tx_height_m': 50.0,
        'rx_height_m': 10.0
    },
    'rain': {
        'model': 'rain',
        'frequency_ghz': 10.0,
        'distance_km': 5.0,
        'rain_rate_mmh': 25.0,
        'tx_height_m': 50.0,
        'rx_height_m': 10.0
    },
    'gas': {
        'model': 'gas',
        'frequency_ghz': 30.0,
        'distance_km': 5.0,
        'temperature_c': 15.0,
        'pressure_hpa': 1013.25,
        'relative_humidity': 50.0,
        'tx_height_m': 50.0,
        'rx_height_m': 10.0
    },
    'fog': {
        'model': 'fog',
        'frequency_ghz': 100.0,
        'distance_km': 2.0,
        'fog_density_gm3': 0.5,
        'tx_height_m': 50.0,
        'rx_height_m': 10.0
    },
    'ci': {
        'model': 'ci',
        'frequency_mhz': 900.0,
        'distance_km': 1.0,
        'reference_distance_m': 1.0,
        'tx_height_m': 50.0,
        'rx_height_m': 10.0
    }
}

# Sample invalid configurations for testing edge cases
SAMPLE_INVALID_CONFIGS = {
    'negative_distance': {
        'model': 'fspl',
        'frequency_mhz': 300.0,
        'distance_km': -1.0,  # Invalid: negative distance
        'tx_height_m': 50.0,
        'rx_height_m': 10.0
    },
    'zero_frequency': {
        'model': 'fspl',
        'frequency_mhz': 0.0,  # Invalid: zero frequency
        'distance_km': 5.0,
        'tx_height_m': 50.0,
        'rx_height_m': 10.0
    },
    'negative_height': {
        'model': 'fspl',
        'frequency_mhz': 300.0,
        'distance_km': 5.0,
        'tx_height_m': -10.0,  # Invalid: negative height
        'rx_height_m': 10.0
    }
}

# Expected output ranges for validation (min, max in dB)
EXPECTED_OUTPUT_RANGES = {
    'fspl': (0, 200),
    'rain': (0, 50),
    'gas': (0, 10),
    'fog': (0, 20),
    'ci': (0, 200),
    'itm': (0, 200),
    'ray_tracing': (0, 200)
}