from datetime import datetime
import xml.etree.ElementTree as ET
from scipy.interpolate import CubicHermiteSpline
from scipy.optimize import minimize_scalar
from pyproj import Transformer
from scipy.constants import c
import numpy as np
import random
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

def get_state_vectors(filename):
    tree = ET.parse(filename)
    root = tree.getroot()

    platform = root.find('platform')
    orbit_tag = platform.find('orbit')

    time_str = orbit_tag.find('orbitHeader/firstStateTime/firstStateTimeUTC').text
    time_format = "%Y-%m-%dT%H:%M:%S.%f"
    reference_time = datetime.strptime(time_str, time_format)

    t = []
    p = []
    v = []

    # Parse each <stateVec>
    for state_vec in orbit_tag.findall('stateVec'):
        time_utc = state_vec.find('timeUTC').text
        posX = float(state_vec.find('posX').text)
        posY = float(state_vec.find('posY').text)
        posZ = float(state_vec.find('posZ').text)
        velX = float(state_vec.find('velX').text)
        velY = float(state_vec.find('velY').text)
        velZ = float(state_vec.find('velZ').text)

        # Calculate time difference from reference time

        current_time = datetime.strptime(time_utc, time_format)
        time_diff = (current_time - reference_time).total_seconds()

        t.append(time_diff)
        p.append([posX, posY, posZ])
        v.append([velX, velY, velZ])

    return t, p, v, reference_time

def get_spacing(filename):
    tree = ET.parse(filename)
    root = tree.getroot()
    product_info = root.find('productInfo')
    scene_info = product_info.find('sceneInfo')
    time_format = "%Y-%m-%dT%H:%M:%S.%fZ"

    start_time_utc = scene_info.find('start/timeUTC')

    first_azimuth_datetime = datetime.strptime(start_time_utc.text, time_format)
    range_time_1st_pix = float(scene_info.find('rangeTime/firstPixel').text) / 2.0

    imageRaster = product_info.find('imageDataInfo').find('imageRaster')

    range_spacing = float(imageRaster.find('rowSpacing').text) / 2.0
    azimuth_spacing = float(imageRaster.find('columnSpacing').text)

    return range_time_1st_pix, first_azimuth_datetime, range_spacing, azimuth_spacing

def get_rd_parameters(filename):
    t, p, v, reference_time = get_state_vectors(filename)
    range_time_1st_pix, first_azimuth_datetime, range_spacing, azimuth_spacing = get_spacing(filename)

    first_line_time = (first_azimuth_datetime - reference_time).total_seconds()

    times = np.array(t)
    positions = np.array(p)
    velocities = np.array(v)
    spline_x = CubicHermiteSpline(times, positions[:, 0], velocities[:, 0])
    spline_y = CubicHermiteSpline(times, positions[:, 1], velocities[:, 1])
    spline_z = CubicHermiteSpline(times, positions[:, 2], velocities[:, 2])

    return spline_x, spline_y, spline_z, first_line_time, times

def get_satpos(spline_x, spline_y, spline_z, first_line_time, times, range_time_1st_pix, range_spacing, azimuth_spacing, target):
    time_min = times[0]
    time_max = times[-1]
    time_tol = 1e-11

    def objective(time):
        orbit_pos = np.array([spline_x(time), spline_y(time), spline_z(time)])
        return np.sum((orbit_pos - target) ** 2)

    result = minimize_scalar(
        objective,
        bounds=(time_min, time_max),
        method='bounded',
        options={'xatol': time_tol}
    )

    if result.success:
        az_time = result.x
        sat_pos = np.array([spline_x(az_time), spline_y(az_time), spline_z(az_time)])
        return sat_pos

    else:
        return None, None


def calculate_baselines(ground_pos, master_pos, slave_pos):
    """
    Calculate the baseline, parallel baseline, and perpendicular baseline.

    Parameters:
    ground_pos (np.array): Ground position in geocentric
    coordinates (X, Y, Z).
    master_pos (np.array): Master satellite position in geocentric
    coordinates (X, Y, Z).
    slave_pos (np.array): Slave satellite position in geocentric
    coordinates (X, Y, Z).

    Returns:
    baseline (float): Total baseline magnitude.
    B_parallel (float): Parallel baseline.
    B_perpendicular (float): Perpendicular baseline.
    """
    # Calculate the baseline vector
    baseline_vector = slave_pos - master_pos

    # Calculate the line of sight (LOS) vector from master
    # satellite to ground position
    los_vector = ground_pos - master_pos

    # Normalize the LOS vector
    los_vector_unit = los_vector / np.linalg.norm(los_vector)

    # Calculate the parallel baseline (projection of
    # baseline_vector onto LOS)
    B_parallel = np.dot(baseline_vector, los_vector_unit)

    # Calculate the perpendicular baseline
    B_perpendicular_vector = baseline_vector - B_parallel * los_vector_unit
    B_perpendicular = np.linalg.norm(B_perpendicular_vector)

    # Calculate the total baseline magnitude
    baseline = np.linalg.norm(baseline_vector)

    return baseline, B_parallel, B_perpendicular

def get_pixel(spline_x, spline_y, spline_z, first_line_time, times, range_time_1st_pix, range_spacing, azimuth_spacing, target):
    time_min = times[0]
    time_max = times[-1]
    time_tol = 1e-11

    def objective(time):
        orbit_pos = np.array([spline_x(time), spline_y(time), spline_z(time)])
        return np.sum((orbit_pos - target) ** 2)

    result = minimize_scalar(
        objective,
        bounds=(time_min, time_max),
        method='bounded',
        options={'xatol': time_tol}
    )

    if result.success:
        az_time = result.x
        sat_pos = np.array([spline_x(az_time), spline_y(az_time), spline_z(az_time)])
        distance = np.linalg.norm(sat_pos - target)
        rg_time = distance / c

        az_pix = (az_time - first_line_time) / azimuth_spacing
        rg_pix = (rg_time - range_time_1st_pix) / range_spacing

        return rg_pix, az_pix
    else:
        return None, None

def get_random_points(number_poitns, left, right, bottom, top,
                             min_height, max_height):
    result = []
    for _ in range(number_poitns):
        lon = random.uniform(left, right)
        lat = random.uniform(top, bottom)
        height = random.uniform(min_height, max_height)
        result.append((lat, lon, height))

    return result

if __name__ == "__main__":

    xml_path_master = ''
    xml_path_slave = ''

    left = 130.99
    right = 131.1
    bottom = -25.38
    top = -25.31
    min_height = 0
    max_height = 2000

    mas_spline_x, mas_spline_y, mas_spline_z, mas_first_line_time, mas_times = get_rd_parameters(xml_path_master)
    mas_range_time_1st_pix, mas_first_azimuth_datetime, mas_range_spacing, mas_azimuth_spacing = get_spacing(xml_path_master)

    sl_spline_x, sl_spline_y, sl_spline_z, sl_first_line_time, sl_times = get_rd_parameters(xml_path_slave)
    sl_range_time_1st_pix, sl_first_azimuth_datetime, sl_range_spacing, sl_azimuth_spacing = get_spacing(xml_path_slave)

    transformer = Transformer.from_crs("EPSG:4326", "EPSG:4978", always_xy=True)


    # Sample random points in the burst footprint
    pnts = get_random_points(400, left, right, bottom, top,
                             min_height, max_height)

    # Collect features (x, y, height) and targets
    # (perpendicular baseline)
    X = []
    y = []
    for lat, lon, h in pnts:
        # Convert to geocentric coordinates
        XX, YY, ZZ = transformer.transform(lon, lat, h)
        target = np.array([XX, YY, ZZ])

        # Get pixel coordinates (x, y) in the primary burst
        x_pixel, y_pixel = get_pixel(mas_spline_x, mas_spline_y, mas_spline_z, mas_first_line_time, mas_times, mas_range_time_1st_pix, mas_range_spacing, mas_azimuth_spacing, target)

        # Get satellite positions for both bursts
        primary_pos = get_satpos(mas_spline_x, mas_spline_y, mas_spline_z, mas_first_line_time, mas_times,
                                    mas_range_time_1st_pix, mas_range_spacing, mas_azimuth_spacing, target)

        secondary_pos = get_satpos(sl_spline_x, sl_spline_y, sl_spline_z, sl_first_line_time, sl_times,
                                   sl_range_time_1st_pix, sl_range_spacing, sl_azimuth_spacing, target)

        # Calculate perpendicular baseline
        _, _, perp_baseline = calculate_baselines(target, primary_pos,
                                                  secondary_pos)

        # Store features and target
        X.append([x_pixel, y_pixel, h])
        y.append(perp_baseline)

    X = np.array(X)
    y = np.array(y)

    # Train a polynomial regression model
    model = make_pipeline(
        PolynomialFeatures(degree=3, include_bias=False),
        LinearRegression()
    )
    model.fit(X, y)

    x = np.linspace(0, 10000, int(10000 / 100))
    y = np.linspace(0, 6000, int(6000 / 100))
    xx, yy = np.meshgrid(x, y)
    hh = np.full(len(xx.ravel()), 0.0)
    X = np.column_stack((xx.ravel(), yy.ravel(), hh))


    pred_dx1 = model.predict(X).reshape(xx.shape)


    # Create a figure with two subplots
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))

    # Plot the estimated dx shifts
    im1 = ax1.imshow(pred_dx1,
                     extent=[0, 10000, 0, 6000],
                     origin='lower', cmap='viridis')
    ax1.set_title('Estimated Baselines')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    fig.colorbar(im1, ax=ax1, label='perpendicular')
    plt.show()