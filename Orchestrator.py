import argparse
import json
import os
import sys
import datetime
import shutil

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from RF_prop_sim.main import run_simulation
from RF_prop_sim.input_data_collection.ingestion import parse_simulation_config

#Should focus solely on importing functions and results from other folders,including UI, and applying em in a way to present to the user

def load_google_key_from_json(path: str) -> str | None:
    """Try to extract a Google Maps API key from a JSON file.

    Heuristics: look for values named 'api_key', 'key', 'GOOGLE_MAPS_API_KEY',
    or any string value containing the typical 'AIza' prefix.
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        return None

    # Flatten values and search
    def walk(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, (dict, list)):
                    val = walk(v)
                    if val:
                        return val
                else:
                    if isinstance(v, str):
                        if k.lower() in ('api_key', 'key', 'google_maps_api_key'):
                            return v
                        if v.startswith('AIza'):
                            return v
        elif isinstance(obj, list):
            for item in obj:
                val = walk(item)
                if val:
                    return val
        return None

    return walk(data)


def write_report(output_dir: str, results: dict):
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, 'report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('# Simulation Report\n\n')
        f.write(f"Date: {datetime.datetime.utcnow().isoformat()} UTC\n\n")
        f.write('## Summary\n')
        for k, v in results.items():
            f.write(f'- **{k}**: {v}\n')
    return report_path


def parse_args():
    p = argparse.ArgumentParser(description='Orchestrator for RF Simulator')
    p.add_argument('--config', help='Path to simulation config (json/csv)')
    p.add_argument('--antenna', help='Path to antenna config (json/csv)')
    p.add_argument('--google-json', help='Optional JSON file containing Google Maps API key')
    p.add_argument('--out', help='Output folder', default='Output')
    return p.parse_args()


def main():
    args = parse_args()

    # Optionally load Google Maps API key from provided JSON and set env var
    if args.google_json:
        key = load_google_key_from_json(args.google_json)
        if key:
            os.environ['GOOGLE_MAPS_API_KEY'] = key
            print('Loaded Google Maps API key from JSON and set environment variable.')
        else:
            print('Could not find a Google Maps API key in the provided JSON file.')

    cfg = parse_simulation_config(args.config) if args.config else parse_simulation_config(None)
    if args.antenna:
        cfg.antenna_config_path = args.antenna

    # Run simulation
    results = run_simulation(cfg)

    # Save results
    timestamp = datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    out_dir = os.path.join(args.out, timestamp)
    report = write_report(out_dir, results)
    print(f'Report written to: {report}')

    # If we have a Google Maps key, create a small HTML viewer that injects it
    key = os.environ.get('GOOGLE_MAPS_API_KEY')
    if key:
        os.makedirs(out_dir, exist_ok=True)

        # Copy local maps.js into the output directory so the generated HTML
        # can load it without referencing files outside the output folder.
        try:
            src_maps = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'maps.js')
            dst_maps = os.path.join(out_dir, 'maps.js')
            if os.path.exists(src_maps):
                shutil.copy(src_maps, dst_maps)
        except Exception as e:
            print(f'Warning: failed to copy maps.js into output folder: {e}')

        map_html = f'''<!doctype html>
<html>
  <head>
    <meta charset="utf-8">
    <title>Map View</title>
    <style>html,body,#map{{height:100%;margin:0;padding:0}}</style>
  </head>
  <body>
    <div id="map" style="width:100%;height:100vh"></div>
    <script>window.GOOGLE_MAPS_API_KEY = '{key}';</script>
    <script src="maps.js"></script>
  </body>
</html>
'''
        map_path = os.path.join(out_dir, 'map_view.html')
        with open(map_path, 'w', encoding='utf-8') as f:
            f.write(map_html)
        print(f'Map viewer written to: {map_path}')


if __name__ == '__main__':
    main()