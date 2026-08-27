import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "RF_prop_sim"))

from input_data_collection.ingestion import parse_simulation_config, load_config_from_file
from antenna_data.parser import parse_antenna_config


class ConfigLoadingTests(unittest.TestCase):
    def test_load_config_from_file(self):
        config_path = os.path.join(os.path.dirname(__file__), "..", "..", "examples", "sample_sim_config.json")
        cfg = load_config_from_file(config_path)
        self.assertEqual(cfg.model, "fspl")
        self.assertGreater(cfg.distance_km, 0)
        self.assertIsNotNone(cfg.antenna_config_path)

    def test_parse_antenna_config_from_file(self):
        antenna_path = os.path.join(os.path.dirname(__file__), "..", "..", "examples", "sample_antenna.json")
        cfg = parse_antenna_config(antenna_path)
        self.assertGreater(cfg.frequency_mhz, 0)
        self.assertGreater(cfg.gain_dbi, 0)


if __name__ == "__main__":
    unittest.main()
