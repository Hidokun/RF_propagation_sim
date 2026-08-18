"""End-to-end flow integration debugger tests"""
import pytest
import json
import tempfile
import os
from unittest.mock import patch, Mock
from RF_prop_sim.input_data_collection.ingestion import SimulationConfig
from RF_prop_sim.antenna_data.parser import AntennaConfig
from RF_prop_sim.mapping.location_service import Geocoder as LocationService
from propagation_model.models import free_space_path_loss, rain_attenuation
from test_utils.fixtures import SAMPLE_VALID_CONFIGS, SAMPLE_VALID_ANTENNA_CONFIGS

class TestEndToEndFlowDebugger:
    """Test suite for end-to-end flow integration debugger"""
    
    @pytest.mark.integration
    def test_complete_simulation_pipeline(self):
        """Test complete simulation pipeline from config to results"""
        # Create test configuration
        config_data = {
            "frequency": 3000.0,  # MHz
            "distance": 5.0,      # km
            "tx_height": 10.0,    # m
            "rx_height": 1.5,     # m
            "system_type": "point_to_point",
            "climate": "temperate"
        }
        
        # Create test antenna configuration
        antenna_data = {
            "gain": 15.0,         # dBi
            "beamwidth": 65.0,    # degrees
            "polarization": "vertical",
            "frequency": 3000.0   # MHz
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as config_file, \
             tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as antenna_file:
            
            json.dump(config_data, config_file)
            config_path = config_file.name
            
            json.dump(antenna_data, antenna_file)
            antenna_path = antenna_file.name
        
        try:
            # Step 1: Parse configuration
            config = SimulationConfig(config_path)
            assert config is not None
            
            # Step 2: Parse antenna configuration
            antenna_config = AntennaConfig(antenna_path)
            assert antenna_config is not None
            
            # Step 3: Resolve locations (if needed)
            location_service = LocationService()
            
            # Mock geocoding for consistent testing
            with patch.object(location_service, '_geocode_request') as mock_geocode:
                mock_geocode.return_value = {
                    'latitude': 40.7128,
                    'longitude': -74.0060,
                    'display_name': "Test Location"
                }
                
                tx_location = location_service.geocode("Transmitter Location")
                rx_location = location_service.geocode("Receiver Location")
                
                assert tx_location is not None
                assert rx_location is not None
            
            # Step 4: Calculate propagation losses
            frequency_ghz = config.frequency / 1000.0  # Convert to GHz
            distance_km = config.distance
            
            # Free space path loss
            fspl = free_space_path_loss(config.frequency, distance_km)
            assert fspl is not None
            assert fspl >= 0
            
            # Rain attenuation (assuming moderate rain)
            rain_attenuation_val = rain_attenuation(frequency_ghz, distance_km, 25.0)
            assert rain_attenuation_val is not None
            assert rain_attenuation_val >= 0
            
            # Step 5: Apply antenna gains
            tx_gain = antenna_config.gain
            rx_gain = antenna_config.gain  # Simplified - same antenna for tx/rx
            
            # Step 6: Calculate total link budget
            # Simplified: Tx Power + Tx Gain - Losses + Rx Gain
            tx_power_dbm = 30.0  # 1W transmitter
            total_loss = fspl + rain_attenuation_val  # Simplified - only modeling these two
            received_power_dbm = tx_power_dbm + tx_gain - total_loss + rx_gain
            
            # Validate reasonable results
            assert isinstance(received_power_dbm, (int, float))
            # Received power should be reasonable (not extremely high or low)
            assert -200 <= received_power_dbm <= 100  # Reasonable range for wireless link
            
        finally:
            os.unlink(config_path)
            os.unlink(antenna_path)
    
    @pytest.mark.integration
    def test_pipeline_with_error_conditions(self):
        """Test pipeline behavior under error conditions"""
        # Test with invalid configuration
        invalid_config_data = {
            "frequency": -1000.0,  # Invalid negative frequency
            "distance": 5.0,
            "tx_height": 10.0,
            "rx_height": 1.5
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(invalid_config_data, f)
            config_path = f.name
        
        try:
            # Should handle invalid config gracefully
            with pytest.raises((ValueError, AssertionError)):
                SimulationConfig(config_path)
        finally:
            os.unlink(config_path)
    
    @pytest.mark.integration
    def test_pipeline_with_missing_components(self):
        """Test pipeline when optional components are missing"""
        # Test with minimal configuration
        minimal_config_data = {
            "frequency": 3000.0,
            "distance": 5.0
            # Missing tx_height, rx_height, etc.
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(minimal_config_data, f)
            config_path = f.name
        
        try:
            config = SimulationConfig(config_path)
            
            # Should still work with defaults for missing parameters
            assert config is not None
            assert config.frequency == 3000.0
            assert config.distance == 5.0
            # Other parameters should have default values
            
        finally:
            os.unlink(config_path)
    
    @pytest.mark.integration
    def test_pipeline_consistency_checks(self):
        """Test consistency checks across pipeline components"""
        # Test that reasonable input produces reasonable output
        config_data = {
            "frequency": 3000.0,  # 3 GHz
            "distance": 1.0,      # 1 km
            "tx_height": 10.0,
            "rx_height": 1.5
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_path = f.name
        
        try:
            config = SimulationConfig(config_path)
            
            # Calculate expected FSPL for 3 GHz over 1 km
            # FSPL = 32.44 + 20*log10(f_MHz) + 20*log10(d_km)
            # FSPL = 32.44 + 20*log10(3000) + 20*log10(1)
            # FSPL ≈ 32.44 + 20*3.477 + 0 ≈ 32.44 + 69.54 = 101.98 dB
            expected_fspl_approx = 102.0
            
            actual_fspl = free_space_path_loss(config.frequency, config.distance)
            
            # Should be close to expected value (within 1 dB for simplicity)
            assert abs(actual_fspl - expected_fspl_approx) < 1.0
            assert actual_fspl > 0
            
        finally:
            os.unlink(config_path)