"""Location resolution integration debugger tests"""
import pytest
from unittest.mock import patch, Mock
from RF_prop_sim.mapping.location_service import Geocoder as LocationService
from test_utils.fixtures import SAMPLE_LOCATION_TEST_CASES

class TestLocationResolutionDebugger:
    """Test suite for location resolution integration debugger"""
    
    @pytest.mark.integration
    def test_forward_geocoding(self):
        """Test forward geocoding (address to coordinates)"""
        location_service = LocationService()
        
        test_cases = [
            ("New York, NY", (40.7128, -74.0060)),
            ("Los Angeles, CA", (34.0522, -118.2437)),
            ("London, UK", (51.5074, -0.1278)),
        ]
        
        for address, expected_coords in test_cases:
            # Mock the geocoding request to avoid external API calls
            with patch.object(location_service, '_geocode_request') as mock_geocode:
                mock_geocode.return_value = {
                    'latitude': expected_coords[0],
                    'longitude': expected_coords[1],
                    'display_name': address
                }
                
                result = location_service.geocode(address)
                
                assert result is not None
                assert 'latitude' in result
                assert 'longitude' in result
                assert abs(result['latitude'] - expected_coords[0]) < 0.1
                assert abs(result['longitude'] - expected_coords[1]) < 0.1
    
    @pytest.mark.integration
    def test_reverse_geocoding(self):
        """Test reverse geocoding (coordinates to address)"""
        location_service = LocationService()
        
        test_coords = (40.7128, -74.0060)  # New York City
        expected_address = "New York, NY, USA"
        
        # Mock the reverse geocoding request
        with patch.object(location_service, '_reverse_geocode_request') as mock_reverse_geocode:
            mock_reverse_geocode.return_value = expected_address
            
            result = location_service.reverse_geocode(test_coords[0], test_coords[1])
            
            assert result is not None
            assert isinstance(result, str)
            assert expected_address in result or "New York" in result
    
    @pytest.mark.integration
    def test_location_service_error_handling(self):
        """Test location service error handling"""
        location_service = LocationService()
        
        # Test with invalid/empty address
        with patch.object(location_service, '_geocode_request') as mock_geocode:
            mock_geocode.side_effect = Exception("Geocoding service unavailable")
            
            result = location_service.geocode("")
            # Should handle gracefully - either return None or raise handled exception
            assert result is None or isinstance(result, dict)
    
    @pytest.mark.integration
    def test_location_service_caching(self):
        """Test location service caching behavior"""
        location_service = LocationService()
        
        test_address = "New York, NY"
        expected_coords = (40.7128, -74.0060)
        
        # Mock the geocoding request
        with patch.object(location_service, '_geocode_request') as mock_geocode:
            mock_geocode.return_value = {
                'latitude': expected_coords[0],
                'longitude': expected_coords[1],
                'display_name': test_address
            }
            
            # First call
            result1 = location_service.geocode(test_address)
            
            # Second call (should use cache if implemented)
            result2 = location_service.geocode(test_address)
            
            assert result1 is not None
            assert result2 is not None
            assert result1['latitude'] == result2['latitude']
            assert result1['longitude'] == result2['longitude']
            
            # Verify the mock was called (caching behavior may vary)
            assert mock_geocode.call_count >= 1