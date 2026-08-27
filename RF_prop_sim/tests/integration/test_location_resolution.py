"""Location resolution integration debugger tests"""
import pytest
from unittest.mock import patch, MagicMock
from RF_prop_sim.mapping.location_service import Geocoder, geocode_location
from config import DEFAULTS


def _mock_googlemaps_client(lat, lng, formatted_address):
    """Build a mock replacing googlemaps.Client with canned geocode output."""
    client = MagicMock()
    client.geocode.return_value = [
        {
            "geometry": {"location": {"lat": lat, "lng": lng}},
            "formatted_address": formatted_address,
        }
    ]
    return client


class TestLocationResolutionDebugger:
    """Test suite for location resolution integration debugger"""

    @pytest.mark.integration
    def test_forward_geocoding(self):
        """Forward geocoding returns lat/lng plus formatted address"""
        service = Geocoder(api_key="fake-key-for-testing")
        fake_client = _mock_googlemaps_client(40.7128, -74.0060, "New York, NY, USA")

        with patch("RF_prop_sim.mapping.location_service.googlemaps.Client", return_value=fake_client):
            result = service.geocode("New York, NY")

        assert result is not None
        assert "lat" in result and "lng" in result
        assert abs(result["lat"] - 40.7128) < 0.1
        assert abs(result["lng"] - (-74.0060)) < 0.1
        assert "New York" in result["formatted_address"]

    @pytest.mark.integration
    def test_geocode_unresolvable_address_raises_inside_and_returns_none(self):
        """Empty results from the API surface as None (service catches errors)"""
        service = Geocoder(api_key="fake-key-for-testing")
        fake_client = MagicMock()
        fake_client.geocode.return_value = []  # nothing found

        with patch("RF_prop_sim.mapping.location_service.googlemaps.Client", return_value=fake_client):
            result = service.geocode("Nowhere At All XYZ123")

        assert result is None

    @pytest.mark.integration
    def test_location_service_error_handling(self):
        """API failures are caught; geocode degrades to None instead of raising"""
        service = Geocoder(api_key="fake-key-for-testing")
        fake_client = MagicMock()
        fake_client.geocode.side_effect = Exception("Geocoding service unavailable")

        with patch("RF_prop_sim.mapping.location_service.googlemaps.Client", return_value=fake_client):
            result = service.geocode("")

        assert result is None

    @pytest.mark.integration
    def test_fallback_without_api_key(self):
        """Without an API key the documented Casablanca fallback is returned"""
        service = Geocoder(api_key=None)
        result = service.geocode("anything at all")
        assert result is not None
        assert result == DEFAULTS["FALLBACK_GEOCODE_LOCATION"]

    @pytest.mark.integration
    def test_helper_function_delegates(self):
        """Module-level geocode_location helper delegates to Geocoder"""
        fake_client = _mock_googlemaps_client(33.5883, -7.6114, "Casablanca, Morocco")
        with patch("RF_prop_sim.mapping.location_service.googlemaps.Client", return_value=fake_client):
            result = geocode_location("Casablanca", api_key="fake-key-for-testing")
        assert result is not None
        assert abs(result["lat"] - 33.5883) < 0.1

    @pytest.mark.integration
    def test_repeated_calls_consistent(self):
        """Repeated calls with same input give identical results"""
        service = Geocoder(api_key="fake-key-for-testing")
        fake_client = _mock_googlemaps_client(40.7128, -74.0060, "New York, NY, USA")

        with patch("RF_prop_sim.mapping.location_service.googlemaps.Client", return_value=fake_client):
            r1 = service.geocode("New York, NY")
            r2 = service.geocode("New York, NY")

        assert r1 == r2
