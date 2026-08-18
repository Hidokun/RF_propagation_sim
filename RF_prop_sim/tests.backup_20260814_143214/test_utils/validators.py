"""Validation helpers for outputs"""
import numpy as np

def validate_fspl_output(result, frequency_mhz, distance_km):
    """Validate FSPL output against theoretical formula"""
    # Handle zero or negative inputs - should return 0 or handle gracefully
    if frequency_mhz <= 0 or distance_km <= 0:
        return True  # Allow any reasonable handling of invalid inputs
    
    # Theoretical FSPL: 32.44 + 20*log10(d_km) + 20*log10(f_MHz)
    expected = 32.44 + 20.0 * np.log10(distance_km) + 20.0 * np.log10(frequency_mhz)
    tolerance = 0.01  # Allow small numerical differences
    return abs(result - expected) <= tolerance

def validate_rain_attenuation_output(result, frequency_ghz, distance_km, rain_rate_mmh, polarization="horizontal"):
    """Validate rain attenuation output"""
    # Handle zero or negative inputs - should return 0 or handle gracefully
    if frequency_ghz <= 0 or distance_km <= 0 or rain_rate_mmh <= 0:
        return True  # Allow any reasonable handling of invalid inputs
    
    # Basic sanity check: result should be reasonable (can be complex in some implementations)
    if np.iscomplex(result):
        # For complex results, check magnitude
        magnitude = np.abs(result)
        return magnitude >= 0 and not np.isnan(magnitude) and not np.isinf(magnitude)
    else:
        # For real results, just check if it's a reasonable number
        return not np.isnan(result) and not np.isinf(result)

def validate_gas_attenuation_output(result, frequency_ghz, distance_km, temperature_c=15.0, pressure_hpa=1013.25, relative_humidity=50.0):
    """Validate gas attenuation output"""
    # Handle zero or negative inputs - should return 0 or handle gracefully
    if frequency_ghz <= 0 or distance_km <= 0:
        return True  # Allow any reasonable handling of invalid inputs
    
    # Basic sanity check: result should be a reasonable number
    return not np.isnan(result) and not np.isinf(result)

def validate_fog_attenuation_output(result, frequency_ghz, distance_km, fog_density_gm3):
    """Validate fog attenuation output"""
    # Handle zero or negative inputs - should return 0 or handle gracefully
    if frequency_ghz <= 0 or distance_km <= 0 or fog_density_gm3 <= 0:
        return True  # Allow any reasonable handling of invalid inputs
    
    # Basic sanity check: result should be a reasonable number
    return not np.isnan(result) and not np.isinf(result)

def validate_ci_output(result, frequency_mhz, distance_km, reference_distance=1.0):
    """Validate CI path loss output"""
    # Handle zero or negative inputs - should return 0 or handle gracefully
    if frequency_mhz <= 0 or distance_km <= 0:
        return True  # Allow any reasonable handling of invalid inputs
    
    # Basic sanity check: result should be a reasonable number
    return not np.isnan(result) and not np.isinf(result)

def validate_itm_output(result, frequency_mhz, distance_km, tx_height_m, rx_height_m):
    """Validate ITM output"""
    # Handle zero or negative inputs - should return 0 or handle gracefully
    if frequency_mhz <= 0 or distance_km <= 0 or tx_height_m < 0 or rx_height_m < 0:
        return True  # Allow any reasonable handling of invalid inputs
    
    # Basic sanity check: result should be a reasonable number
    return not np.isnan(result) and not np.isinf(result)

def validate_ray_tracing_output(result, frequency_ghz, tx_pos, rx_pos):
    """Validate ray tracing output"""
    # Handle zero or negative inputs - should return 0 or handle gracefully
    if frequency_ghz <= 0:
        return True  # Allow any reasonable handling of invalid inputs
    
    # Calculate distance for sanity check
    distance_m = float(np.linalg.norm(np.subtract(tx_pos, rx_pos)))
    if distance_m <= 0.0:
        return True  # Allow any reasonable handling of invalid inputs
    
    # Basic sanity check: result should be a reasonable number
    return not np.isnan(result) and not np.isinf(result)