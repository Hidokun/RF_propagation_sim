import os
import sys

import googlemaps

# Add the project root directory to the path so we can import config
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from config import get_api_key, DEFAULTS


class Geocoder:
    """Light wrapper for Google Maps geocoding."""

    def __init__(self, api_key=None):
        self.api_key = api_key or get_api_key("GOOGLE_MAPS")
        if not self.api_key:
            print("WARNING: No Google Maps API key found. Geocoder will use fallback coordinates.")

    def geocode(self, address):
        """Resolve an address or place name to lat/lng."""
        if not self.api_key:
            return DEFAULTS["FALLBACK_GEOCODE_LOCATION"]

        try:
            gmaps = googlemaps.Client(key=self.api_key)
            geocode_result = gmaps.geocode(address)

            if geocode_result:
                location = geocode_result[0]["geometry"]["location"]
                return {
                    "lat": location["lat"],
                    "lng": location["lng"],
                    "formatted_address": geocode_result[0]["formatted_address"],
                }

            raise ValueError(f"Could not resolve address: {address}")

        except Exception as e:
            print(f"Google Maps API error: {e}")
            return None


def geocode_location(address, api_key=None):
    """Backward-compatible helper to resolve a place name."""
    return Geocoder(api_key=api_key).geocode(address)
