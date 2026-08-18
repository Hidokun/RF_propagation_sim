import numpy as np

### 1. Free Space Propagation Model ###
def free_space_path_loss(frequency_mhz: float, distance_km: float) -> float:
    """Return the free-space path loss in dB.

    Uses the standard FSPL expression with frequency in MHz and distance in km:
        FSPL(dB) = 32.44 + 20*log10(f_MHz) + 20*log10(d_km)
    """
    if distance_km <= 0.0 or frequency_mhz <= 0.0:
        return 0.0
    return 32.44 + 20.0 * np.log10(distance_km) + 20.0 * np.log10(frequency_mhz)


### 2. Rain Propagation Model ###
def rain_attenuation(
    frequency_ghz: float,
    distance_km: float,
    rain_rate_mmh: float,
    k: float = None,
    alpha: float = None,
    polarization: str = "horizontal",
) -> float:
    """Return rain attenuation in dB for a slant path.

    A simplified ITU-R P.838-style power law is used:
        gamma_R = k * R^alpha
        A = gamma_R * d
    """
    if distance_km <= 0.0 or rain_rate_mmh <= 0.0:
        return 0.0

    polarization = polarization.lower()
    if k is None or alpha is None:
        if polarization.startswith("h"):
            k = 0.0001 * frequency_ghz ** 0.88
            alpha = 0.90
        elif polarization.startswith("v"):
            k = 0.00012 * frequency_ghz ** 0.84
            alpha = 0.91
        else:
            k = 0.00011 * frequency_ghz ** 0.86
            alpha = 0.90

    gamma_r = k * rain_rate_mmh ** alpha
    return gamma_r * distance_km


### 3. Gas Propagation Model ###
def gas_attenuation(
    frequency_ghz: float,
    distance_km: float,
    temperature_c: float = 15.0,
    pressure_hpa: float = 1013.25,
    relative_humidity: float = 50.0,
) -> float:
    """Return atmospheric gas attenuation in dB using temperature, pressure, and humidity."""
    if distance_km <= 0.0:
        return 0.0

    temperature_k = temperature_c + 273.15
    saturation_vapor_pressure = 6.1121 * np.exp((17.502 * temperature_c) / (temperature_c + 240.97))
    partial_pressure = relative_humidity / 100.0 * saturation_vapor_pressure
    water_vapor_density = 216.7 * partial_pressure / temperature_k

    gamma_o = 0.0001 * pressure_hpa * frequency_ghz ** 2 / (frequency_ghz ** 2 + 0.1)
    gamma_w = 0.000045 * water_vapor_density * frequency_ghz ** 2 / (frequency_ghz ** 2 + 0.5)
    return (gamma_o + gamma_w) * distance_km


### 4. Fog Propagation Model ###
def fog_attenuation(frequency_ghz: float, distance_km: float, fog_density_gm3: float) -> float:
    """Return fog attenuation in dB using an ITU-R P.840-style approximation."""
    if distance_km <= 0.0 or fog_density_gm3 <= 0.0:
        return 0.0

    gamma_f = 0.2 * fog_density_gm3 * frequency_ghz ** 2 / (frequency_ghz ** 2 + 0.7)
    return gamma_f * distance_km


### 5. Close-In Propagation Model ###
def close_in_path_loss(
    frequency_mhz: float,
    distance_km: float,
    reference_distance_m: float = 1.0,
    path_loss_exponent: float = 2.0,
) -> float:
    """Return a close-in reference path loss estimate in dB."""
    d_m = max(distance_km * 1e3, reference_distance_m)
    fspl_ref = 32.44 + 20.0 * np.log10(frequency_mhz)
    return fspl_ref + 10.0 * path_loss_exponent * np.log10(d_m / reference_distance_m)


### 6. TIREM-style Propagation Model ###
def tirem_path_loss(
    frequency_mhz: float,
    distance_km: float,
    terrain_type: str = "average",
    surface_permittivity: float = 15.0,
    ground_conductivity: float = 0.005,
    effective_earth_radius_factor: float = 4.0 / 3.0,
) -> float:
    """Return a terrain-aware path loss estimate that mimics TIREM-style penalties."""
    baseline = free_space_path_loss(frequency_mhz, distance_km)
    terrain_penalty = {"average": 5.0, "hilly": 12.0, "mountainous": 20.0}.get(terrain_type, 8.0)
    conductivity_penalty = 5.0 * np.log10(1.0 + ground_conductivity * 1e3)
    permittivity_penalty = 2.0 if surface_permittivity > 10.0 else 0.0
    earth_curvature_penalty = 10.0 * np.log10(effective_earth_radius_factor)
    return baseline + terrain_penalty + conductivity_penalty + permittivity_penalty + earth_curvature_penalty
