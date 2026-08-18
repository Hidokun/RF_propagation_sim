"""
Ray Tracing Propagation Model using Sionna RT.
"""

import numpy as np


def _free_space_fallback(frequency_ghz, tx_pos, rx_pos):
    distance_m = float(np.linalg.norm(np.subtract(tx_pos, rx_pos)))
    if distance_m <= 0.0:
        return float("inf")
    distance_km = distance_m / 1000.0
    return 32.44 + 20.0 * np.log10(frequency_ghz) + 20.0 * np.log10(distance_km)


def ray_tracing_path_loss(frequency_ghz, tx_pos, rx_pos, scene_name="munich"):
    """
    Estimate the path loss using Sionna RT.

    If Sionna is unavailable, returns a free-space fallback estimate.
    """
    try:
        import sionna.rt
        from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, PathSolver
    except ImportError:
        print("Warning: sionna or mitsuba is not installed. Falling back to free-space path loss.")
        return _free_space_fallback(frequency_ghz, tx_pos, rx_pos)

    print(f"Loading Sionna scene: {scene_name}...")
    try:
        if scene_name == "munich":
            scene = load_scene(sionna.rt.scene.munich)
        else:
            scene = load_scene(sionna.rt.scene.simple_street_canyon)

        scene.frequency = frequency_ghz * 1e9
        scene.tx_array = PlanarArray(num_rows=1, num_cols=1, pattern="tr38901", polarization="V")
        scene.rx_array = PlanarArray(num_rows=1, num_cols=1, pattern="dipole", polarization="cross")

        tx = Transmitter(name="tx", position=tx_pos, display_radius=2)
        rx = Receiver(name="rx", position=rx_pos, display_radius=2)
        scene.add(tx)
        scene.add(rx)
        tx.look_at(rx)

        p_solver = PathSolver()
        print("Computing propagation paths via Ray Tracing...")
        paths = p_solver(scene=scene, max_depth=3, los=True, specular_reflection=True,
                         diffuse_reflection=False, refraction=False)

        a, _ = paths.cir(normalize_delays=True, out_type="numpy")
        a_abs = np.abs(a)
        total_power = np.sum(a_abs**2)
        if total_power == 0.0:
            return float("inf")

        return -10.0 * np.log10(total_power)
    except Exception as e:
        print(f"Error during Sionna ray tracing: {e}")
        return _free_space_fallback(frequency_ghz, tx_pos, rx_pos)
