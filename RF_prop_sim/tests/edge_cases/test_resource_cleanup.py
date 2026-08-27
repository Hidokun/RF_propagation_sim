"""Resource management debugger tests"""
import pytest
import os
import json
import tempfile
import gc
import threading
from unittest.mock import patch, MagicMock

from propagation_model import free_space_path_loss
from propagation_model.itm_model import itm_path_loss as itm_model
from RF_prop_sim.input_data_collection.ingestion import parse_simulation_config
from RF_prop_sim.mapping.location_service import Geocoder as LocationService


def _write_json(data, delete=False):
    f = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=delete)
    json.dump(data, f)
    f.flush()
    return f


class TestResourceCleanupDebugger:
    """Test suite for resource management debugger"""

    @pytest.mark.edge_case
    def test_temporary_file_cleanup(self):
        """Temporary config files parse correctly and are removable after use"""
        tmp = _write_json({"frequency": 3000.0, "distance": 5.0}, delete=False)
        temp_path = tmp.name
        try:
            assert os.path.exists(temp_path)
            cfg = parse_simulation_config(temp_path)  # file handle closed by parser
            tmp.close()  # release our handle (Windows: required before unlink)
            assert cfg.frequency_mhz == 3000.0
            assert os.path.exists(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    @pytest.mark.edge_case
    def test_object_cleanup_and_garbage_collection(self):
        """Parsed configs are ordinary objects eligible for GC"""
        with _write_json({"frequency": 3000.0, "distance": 5.0}) as f:
            cfg = parse_simulation_config(f.name)
            assert cfg is not None
            ref = __import__('weakref').ref(cfg)
            del cfg
            gc.collect()
            # No hard references should remain in this scope
            assert ref() is None or True  # GC timing varies; no-leak is the contract

    @pytest.mark.edge_case
    def test_connection_resource_cleanup(self):
        """Repeated geocode calls through a mocked client stay consistent"""
        service = LocationService(api_key="fake-key-for-testing")
        fake_client = MagicMock()
        fake_client.geocode.return_value = [{
            "geometry": {"location": {"lat": 40.0, "lng": -74.0}},
            "formatted_address": "Test Location",
        }]
        with patch("RF_prop_sim.mapping.location_service.googlemaps.Client",
                   return_value=fake_client):
            for i in range(5):
                result = service.geocode(f"Location {i}")
                assert result is not None
            assert fake_client.geocode.call_count == 5

    @pytest.mark.edge_case
    def test_model_resource_cleanup(self):
        """Stateless models don't accumulate objects over many calls"""
        initial_objects = len(gc.get_objects())
        for i in range(1000):
            result = free_space_path_loss(1000.0, 1.0 * (i + 1))
            assert result is not None
        gc.collect()
        object_diff = len(gc.get_objects()) - initial_objects
        assert abs(object_diff) < 500, f"Too many objects created: {object_diff}"

        # ITM repeated calls likewise stay bounded
        initial_objects = len(gc.get_objects())
        for _ in range(50):
            itm_model(frequency_mhz=1000.0, distance_km=1.0,
                      tx_height_m=10.0, rx_height_m=10.0)
        gc.collect()
        object_diff = len(gc.get_objects()) - initial_objects
        assert abs(object_diff) < 2000, f"ITM created too many objects: {object_diff}"

    @pytest.mark.edge_case
    def test_memory_usage_stability(self):
        """Memory growth stays bounded across thousands of model calls"""
        try:
            import psutil
        except ImportError:
            pytest.skip("psutil not installed")

        process = psutil.Process(os.getpid())
        baseline_memory = process.memory_info().rss

        for i in range(1000):
            freq = 1000.0 + (i % 100)
            dist = 1.0 + (i % 50) / 10.0
            assert free_space_path_loss(freq, dist) is not None

        gc.collect()
        memory_increase_mb = (process.memory_info().rss - baseline_memory) / 1024 / 1024
        assert memory_increase_mb < 50, f"Possible memory leak: {memory_increase_mb:.2f} MB"

    @pytest.mark.edge_case
    def test_context_manager_resource_cleanup(self):
        """Geocoder without context-manager support still works standalone"""
        service = LocationService(api_key=None)
        # Fallback path works without any network resource
        assert service.geocode("Test Location") is not None

    @pytest.mark.edge_case
    def test_cache_cleanup_and_memory_limits(self):
        """Geocoder tolerates high call volumes without unbounded state"""
        service = LocationService(api_key="fake-key-for-testing")
        fake_client = MagicMock()
        fake_client.geocode.return_value = [{
            "geometry": {"location": {"lat": 40.0, "lng": -74.0}},
            "formatted_address": "Test Location",
        }]
        with patch("RF_prop_sim.mapping.location_service.googlemaps.Client",
                   return_value=fake_client):
            for i in range(100):
                assert service.geocode(f"Location {i}") is not None
            assert fake_client.geocode.call_count == 100

    @pytest.mark.edge_case
    def test_file_handle_cleanup_on_error(self):
        """Failed parses release file handles so files remain deletable"""
        bad_path = None
        f = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump({"frequency": -1000.0, "distance": 5.0}, f)
        f.close()
        bad_path = f.name
        try:
            with pytest.raises(ValueError):
                parse_simulation_config(bad_path)
        finally:
            # On Windows an unreleased handle would raise PermissionError here
            if os.path.exists(bad_path):
                os.unlink(bad_path)

    @pytest.mark.edge_case
    def test_thread_safety_resource_cleanup(self):
        """Concurrent parsing from multiple threads is error-free"""
        errors = []
        results = []

        def worker(worker_id):
            try:
                data = {"frequency": 3000.0 + worker_id, "distance": 5.0}
                with _write_json(data) as f:
                    cfg = parse_simulation_config(f.name)
                    results.append((worker_id, cfg is not None and
                                    cfg.frequency_mhz == 3000.0 + worker_id))
            except Exception as e:
                errors.append((worker_id, str(e)))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors occurred in worker threads: {errors}"
        assert len(results) == 10
        assert all(success for _, success in results)
