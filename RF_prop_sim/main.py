import argparse
import os
import sys
import math
import datetime
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from propagation_model import (
    free_space_path_loss,
    rain_attenuation,
    gas_attenuation,
    fog_attenuation,
    close_in_path_loss,
    itm_path_loss,
    ray_tracing_path_loss,
)

from mapping import (
    geocode_location,
    download_buildings,
    create_coverage_map,
    render_3d_scene,
    download_sample_dem,
    render_dem_3d,
    get_elevation,
    geodetic_to_enu,
)

from optimization.optimizer import optimize_antenna_placement
from input_data_collection.ingestion import parse_simulation_config, SimulationConfig
from antenna_data.parser import parse_antenna_config, AntennaConfig

MODEL_CHOICES = ["fspl", "rain", "ci", "itm", "sionna", "gas", "fog"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="RF Coverage Simulator: gather parameters from mapping, antenna, and input data modules."
    )
    parser.add_argument("--config", help="Simulation configuration file path")
    parser.add_argument("--antenna", help="Antenna configuration file path")
    parser.add_argument("--address", help="Place name or address to geocode")
    parser.add_argument("--model", choices=MODEL_CHOICES, help="Propagation model to evaluate")
    parser.add_argument("--frequency_mhz", type=float, help="Center frequency in MHz")
    parser.add_argument("--distance_km", type=float, help="Path distance in kilometers")
    parser.add_argument("--tx_height_m", type=float, help="Transmitter height in meters")
    parser.add_argument("--rx_height_m", type=float, help="Receiver height in meters")
    parser.add_argument("--rain_rate_mmh", type=float, help="Rain rate in mm/h")
    parser.add_argument("--rain_k", type=float, help="Rain coefficient k")
    parser.add_argument("--rain_alpha", type=float, help="Rain exponent alpha")
    parser.add_argument("--antenna_power_dbm", type=float, help="Antenna transmit power in dBm")
    parser.add_argument("--antenna_gain_dbi", type=float, help="Antenna gain in dBi")
    parser.add_argument("--antenna_lat", type=float, help="Antenna latitude")
    parser.add_argument("--antenna_lng", type=float, help="Antenna longitude")
    parser.add_argument("--antenna_alt_m", type=float, help="Antenna altitude in meters")
    parser.add_argument("--show_dem", action="store_true", help="Download and render a sample DEM in 3D")
    parser.add_argument("--optimize", action="store_true", help="Run antenna placement optimization instead of a single propagation test")
    parser.add_argument("--report", action="store_true", help="Write a timestamped markdown report to the output directory")
    return parser.parse_args()


def build_simulation_config(args) -> SimulationConfig:
    cfg = parse_simulation_config(args.config) if args and args.config else parse_simulation_config(None)

    if args is None:
        return cfg

    if args.address:
        cfg.address = args.address
    if args.model:
        cfg.model = args.model
    if args.frequency_mhz is not None:
        cfg.frequency_mhz = args.frequency_mhz
    if args.distance_km is not None:
        cfg.distance_km = args.distance_km
    if args.tx_height_m is not None:
        cfg.tx_height_m = args.tx_height_m
    if args.rx_height_m is not None:
        cfg.rx_height_m = args.rx_height_m
    if args.rain_rate_mmh is not None:
        cfg.rain_rate_mmh = args.rain_rate_mmh
    if args.rain_k is not None:
        cfg.rain_k = args.rain_k
    if args.rain_alpha is not None:
        cfg.rain_alpha = args.rain_alpha
    if args.antenna_power_dbm is not None:
        cfg.tx_power_dbm = args.antenna_power_dbm
    if args.antenna_gain_dbi is not None:
        cfg.antenna_gain_dbi = args.antenna_gain_dbi
    if args.antenna_lat is not None:
        cfg.antenna_lat = args.antenna_lat
    if args.antenna_lng is not None:
        cfg.antenna_lng = args.antenna_lng
    if args.antenna_alt_m is not None:
        cfg.antenna_alt_m = args.antenna_alt_m
    if args.antenna:
        cfg.antenna_config_path = args.antenna
    if args.optimize:
        cfg.run_optimization = True
    if args.show_dem:
        cfg.run_dem = True

    return cfg


def build_antenna_config(cfg: SimulationConfig) -> AntennaConfig:
    if cfg.antenna_config_path:
        try:
            return parse_antenna_config(cfg.antenna_config_path)
        except Exception as exc:
            print(f"Warning: failed to load antenna config from '{cfg.antenna_config_path}': {exc}")

    inputs = {
        "frequency_mhz": cfg.frequency_mhz,
        "tx_power_dbm": cfg.tx_power_dbm,
        "gain_dbi": cfg.antenna_gain_dbi,
        "height_m": cfg.tx_height_m,
        "lat": cfg.antenna_lat,
        "lng": cfg.antenna_lng,
        # `or` would swallow a legitimate 0 m ground-level antenna (audit M-10)
        "alt": cfg.antenna_alt_m if cfg.antenna_alt_m is not None else cfg.tx_height_m,
    }
    antenna_cfg = parse_antenna_config(inputs)

    if cfg.antenna_lat is not None:
        antenna_cfg.lat = cfg.antenna_lat
    if cfg.antenna_lng is not None:
        antenna_cfg.lng = cfg.antenna_lng
    if cfg.antenna_alt_m is not None:
        antenna_cfg.alt = cfg.antenna_alt_m

    return antenna_cfg


def resolve_location(cfg: SimulationConfig):
    if cfg.center_lat is not None and cfg.center_lng is not None:
        return {
            "lat": cfg.center_lat,
            "lng": cfg.center_lng,
            "formatted_address": cfg.address,
        }

    return geocode_location(cfg.address)


def compute_model_loss(cfg: SimulationConfig, antenna_cfg: AntennaConfig, location: dict):
    model = cfg.model.lower() if cfg.model else "fspl"
    distance_km = cfg.distance_km
    frequency_mhz = cfg.frequency_mhz

    if model == "fspl":
        return free_space_path_loss(frequency_mhz, distance_km)
    if model == "rain":
        return rain_attenuation(frequency_mhz / 1000.0, distance_km, cfg.rain_rate_mmh, k=cfg.rain_k, alpha=cfg.rain_alpha)
    if model == "ci":
        return close_in_path_loss(frequency_mhz, distance_km, reference_distance_m=cfg.ci_reference_distance_m, path_loss_exponent=cfg.ci_path_loss_exponent)
    if model == "itm":
        return itm_path_loss(
            frequency_mhz,
            distance_km,
            tx_height_m=cfg.tx_height_m,
            rx_height_m=cfg.rx_height_m,
            terrain_type=cfg.terrain_type,
            surface_refractivity=cfg.surface_refractivity,
            effective_earth_radius_factor=cfg.effective_earth_radius_factor,
            ground_permittivity=cfg.ground_permittivity,
            ground_conductivity=cfg.ground_conductivity,
        )
    if model == "sionna":
        origin = (location["lat"], location["lng"], get_elevation(location["lat"], location["lng"]))
        tx_altitude = antenna_cfg.alt if antenna_cfg.alt is not None else cfg.tx_height_m
        tx_pos = geodetic_to_enu(antenna_cfg.lat or location["lat"], antenna_cfg.lng or location["lng"], origin[2] + tx_altitude, origin)
        # Place the receiver at the configured path length due east of the center
        rx_lat = location["lat"]
        rx_lng = location["lng"] + (distance_km * 1000.0) / (111320.0 * math.cos(math.radians(location["lat"])))
        rx_alt = get_elevation(rx_lat, rx_lng) + cfg.rx_height_m
        rx_pos = geodetic_to_enu(rx_lat, rx_lng, rx_alt, origin)
        return ray_tracing_path_loss(frequency_mhz / 1000.0, tx_pos, rx_pos, scene_name="munich")
    if model == "gas":
        return gas_attenuation(frequency_mhz / 1000.0, distance_km, temperature_c=cfg.temperature_c, pressure_hpa=cfg.pressure_hpa, relative_humidity=cfg.relative_humidity)
    if model == "fog":
        return fog_attenuation(frequency_mhz / 1000.0, distance_km, cfg.fog_liquid_water_density_gm3)

    print(f"Unknown model: {model}. Defaulting to free-space path loss.")
    return free_space_path_loss(frequency_mhz, distance_km)


def propagation_kwargs_from_cfg(cfg: SimulationConfig) -> dict:
    """Mirror cfg fields into coverage-engine propagation kwargs.

    Audit fix #1: receiver reports previously dropped every user-configured
    weather/terrain parameter (a 'rain' run with rain_rate_mmh=0 silently
    priced 5 mm/h from engine defaults). The engine ignores kwargs it does
    not need, so forwarding the full set is safe.
    """
    return {
        "rain_rate_mmh": cfg.rain_rate_mmh,
        "rain_k": cfg.rain_k,
        "rain_alpha": cfg.rain_alpha,
        "temperature_c": cfg.temperature_c,
        "pressure_hpa": cfg.pressure_hpa,
        "relative_humidity": cfg.relative_humidity,
        "fog_liquid_water_density_gm3": cfg.fog_liquid_water_density_gm3,
        "tx_height_m": cfg.tx_height_m,
        "rx_height_m": cfg.rx_height_m,
        "ci_reference_distance_m": cfg.ci_reference_distance_m,
        "ci_path_loss_exponent": cfg.ci_path_loss_exponent,
        "terrain_type": cfg.terrain_type,
        "surface_refractivity": cfg.surface_refractivity,
        "effective_earth_radius_factor": cfg.effective_earth_radius_factor,
        "ground_permittivity": cfg.ground_permittivity,
        "ground_conductivity": cfg.ground_conductivity,
    }


def print_simulation_summary(cfg: SimulationConfig, antenna_cfg: AntennaConfig, location: dict, elevation_m: float):
    print("\n--- Simulation Parameters ---")
    print(f"Model: {cfg.model}")
    print(f"Frequency: {cfg.frequency_mhz:.2f} MHz")
    print(f"Distance: {cfg.distance_km:.3f} km")
    print(f"Transmitter height: {cfg.tx_height_m:.2f} m")
    print(f"Receiver height: {cfg.rx_height_m:.2f} m")
    print(f"Location: {location.get('formatted_address', 'unknown')} (lat={location['lat']:.6f}, lng={location['lng']:.6f})")
    print(f"Ground elevation at center: {elevation_m:.2f} m")
    print(f"Antenna TX power: {antenna_cfg.tx_power_dbm:.1f} dBm")
    print(f"Antenna gain: {antenna_cfg.gain_dbi:.1f} dBi")


def run_simulation(cfg: SimulationConfig, antenna_cfg: Optional[AntennaConfig] = None) -> dict:
    """Run a simulation from a prepared SimulationConfig.

    Returns a dict with results and produced file paths.
    """
    # Build antenna config if not provided
    antenna_cfg = antenna_cfg or build_antenna_config(cfg)

    if cfg.run_optimization:
        opt_result = optimize_antenna_placement(area_bounds_km=cfg.opt_area_km,
                                                freq_mhz=cfg.frequency_mhz,
                                                tx_power_dbm=cfg.tx_power_dbm)
        # Audit L-13: the optimized position is the product â€” report it.
        placement = getattr(opt_result, "x", None)
        if placement is None and isinstance(opt_result, (list, tuple)):
            placement = opt_result[0]
        print(f"Optimal antenna placement (x, y, h): {placement}")
        return {"status": "optimization_ran", "optimal_placement": placement}

    if cfg.run_dem:
        dem_file = download_sample_dem()
        if dem_file:
            render_dem_3d(dem_file)
            return {"status": "dem_rendered", "dem_file": dem_file}
        return {"status": "dem_failed"}

    location_data = resolve_location(cfg)
    if not location_data:
        return {"status": "location_failed"}

    if cfg.center_lat is None or cfg.center_lng is None:
        cfg.center_lat = location_data["lat"]
        cfg.center_lng = location_data["lng"]

    elevation_m = get_elevation(location_data["lat"], location_data["lng"])

    buildings = download_buildings(location_data["lat"], location_data["lng"], dist=int(cfg.area_radius_m))
    output_dir = cfg.output_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "Output")
    os.makedirs(output_dir, exist_ok=True)
    map_file = os.path.join(output_dir, "coverage_map.html")
    create_coverage_map(location_data["lat"], location_data["lng"], buildings_gdf=buildings, output_file=map_file)

    # Ensure a model is selected (no interactive prompts in API)
    cfg.model = (cfg.model or "fspl").lower()

    loss = compute_model_loss(cfg, antenna_cfg, location_data)

    if cfg.model == "sionna":
        try:
            render_3d_scene(scene_name="casablanca")
        except Exception as render_exc:  # audit L-16: log, don't vanish
            print(f"NOTE: 3D scene render skipped ({render_exc})")

    result = {
        "status": "ok",
        "model": cfg.model,
        "path_loss_db": float(loss),
        "map_file": map_file,
        "elevation_m": float(elevation_m),
        "location": location_data,
        "antenna": antenna_cfg.to_dict(),
    }

    # Audit M-8: parsed receiver lists (multi-row CSV) are now actually
    # evaluated instead of being silently ignored.
    if getattr(cfg, "receivers", None):
        tx_dicts = [{
            "name": antenna_cfg.name,
            "lat": antenna_cfg.lat if antenna_cfg.lat is not None else location_data["lat"],
            "lng": antenna_cfg.lng if antenna_cfg.lng is not None else location_data["lng"],
            "frequency_mhz": antenna_cfg.frequency_mhz,
            "tx_power_dbm": antenna_cfg.tx_power_dbm,
            "gain_dbi": antenna_cfg.gain_dbi,
            "height_m": antenna_cfg.height_m,
            "nature": "transmitter",
        }]
        rx_dicts = [{
            "name": f"RX{i + 1}", "lat": rx.lat, "lng": rx.lng,
            "height_m": rx.height_m, "nature": "receiver",
        } for i, rx in enumerate(cfg.receivers)]
        from coverage_engine import evaluate_receivers
        result["receiver_reports"] = evaluate_receivers(
            rx_dicts, tx_dicts,
            model=cfg.model,
            combining=getattr(cfg, "combining", "superposition"),
            buildings_gdf=buildings,
            **propagation_kwargs_from_cfg(cfg),
        )

    return result


def generate_report(results: dict, output_dir: str) -> str:
    """Write a timestamped markdown report (and optional key-injected map
    viewer) for a simulation result dict. Returns the report path."""
    import datetime

    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Simulation Report\n\n")
        f.write(f"Date: {datetime.datetime.now(datetime.timezone.utc).isoformat()} UTC\n\n")
        f.write("## Summary\n")
        for k, v in results.items():
            f.write(f"- **{k}**: {v}\n")

    # Optional Google Maps viewer when a key is configured.
    # NOTE (audit M-9 decision): the key is intentionally embedded so the
    # artifact works standalone; treat Output/ as a shareable-sensitive dir.
    api_key = None
    try:
        from config import get_api_key
        api_key = get_api_key("GOOGLE_MAPS")
    except Exception:
        pass
    if api_key:
        import shutil
        root_maps = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "maps.js")
        out_maps = os.path.join(output_dir, "maps.js")
        if os.path.exists(root_maps) and not os.path.exists(out_maps):
            try:
                shutil.copy(root_maps, out_maps)
            except Exception as copy_exc:
                print(f"WARNING: could not copy maps.js into report ({copy_exc})")
        map_path = os.path.join(output_dir, "map_view.html")
        with open(map_path, "w", encoding="utf-8") as f:
            f.write(
                "<!doctype html>\n<html>\n  <head>\n"
                '    <meta charset="utf-8">\n    <title>Map View</title>\n'
                "    <style>html,body,#map{height:100%;margin:0;padding:0}</style>\n"
                "  </head>\n  <body>\n"
                '    <div id="map" style="width:100%;height:100vh"></div>\n'
                f"    <script>window.GOOGLE_MAPS_API_KEY = '{api_key}';</script>\n"
                '    <script src="maps.js"></script>\n  </body>\n</html>\n'
            )
        print(f"Map viewer written to: {map_path}")

    return report_path


def main():
    args = parse_args()
    cfg = build_simulation_config(args)
    antenna_cfg = build_antenna_config(cfg)

    res = run_simulation(cfg, antenna_cfg=antenna_cfg)

    # Print summary for CLI usage
    if isinstance(res, dict) and res.get("status") == "ok":
        print("Welcome to the RF Coverage Simulator (RF_prop_sim)")
        print("--------------------------------------------------")
        loc = res.get("location", {})
        print(f"Location resolved: {loc.get('formatted_address', 'unknown')} (Lat: {loc.get('lat',0):.6f}, Lng: {loc.get('lng',0):.6f})")
        print_simulation_summary(cfg, antenna_cfg, loc, res.get("elevation_m", 0.0))
        print("\n--- Running Simulation ---")
        print(f"Estimated path loss: {res.get('path_loss_db'):.2f} dB")

        if args.report:
            out_dir = cfg.output_dir or os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "Output",
                datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
            )
            report_path = generate_report(res, out_dir)
            print(f"Report written to: {report_path}")
    else:
        print(f"Simulation ended with status: {res.get('status')}")


if __name__ == "__main__":
    main()