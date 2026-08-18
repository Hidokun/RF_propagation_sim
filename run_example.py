import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "RF_prop_sim"))

from main import run_simulation
from input_data_collection.ingestion import load_config_from_file
from antenna_data.parser import parse_antenna_config


def test_model(model_name):
    base = os.path.dirname(__file__)
    cfg_path = os.path.join(base, "examples", "sample_sim_config.json")
    
    # Load the base config
    cfg = load_config_from_file(cfg_path)
    
    # Override the model
    cfg.model = model_name
    
    # Save to temporary file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        # Convert to dict, modify model, then save
        cfg_dict = {
            "address": cfg.address,
            "model": cfg.model,
            "frequency_mhz": cfg.frequency_mhz,
            "distance_km": cfg.distance_km,
            "tx_height_m": cfg.tx_height_m,
            "rx_height_m": cfg.rx_height_m,
            "area_radius_m": cfg.area_radius_m,
            "tx_power_dbm": cfg.tx_power_dbm,
            "antenna_gain_dbi": cfg.antenna_gain_dbi,
            "center_lat": cfg.center_lat,
            "center_lng": cfg.center_lng,
            "antenna_config_path": cfg.antenna_config_path
        }
        json.dump(cfg_dict, f)
        temp_cfg_path = f.name
    
    try:
        cfg = load_config_from_file(temp_cfg_path)
        antenna_cfg = parse_antenna_config(cfg.antenna_config_path)
        result = run_simulation(cfg, antenna_cfg=antenna_cfg)
        return result
    finally:
        # Clean up temp file
        os.unlink(temp_cfg_path)


def main():
    models_to_test = ["fspl", "rain", "ci", "itm", "gas", "fog"]  # Skip sionna for now as it might be heavy
    
    print("Testing various propagation models:")
    print("=" * 50)
    
    for model in models_to_test:
        try:
            print(f"\nTesting {model.upper()} model:")
            result = test_model(model)
            print(f"  Status: {result['status']}")
            print(f"  Path Loss: {result['path_loss_db']:.2f} dB")
            if result['status'] == 'ok':
                print(f"  {model.upper()} model working correctly")
            else:
                print(f"  {model.upper()} model failed: {result.get('error', 'Unknown error')}")
        except Exception as e:
            print(f"   {model.upper()} model failed with exception: {e}")


if __name__ == "__main__":
    main()
