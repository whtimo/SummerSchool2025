from datetime import datetime
import xml.etree.ElementTree as ET
from scipy.interpolate import CubicHermiteSpline
from scipy.optimize import minimize_scalar
from pyproj import Transformer
from scipy.constants import c
import numpy as np

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

if __name__ == "__main__":

    xml_path_master = '/Users/timo/Desktop/SummerSchool/data/TSX/Australia/TSX1_SAR__SSC______HS_S_SRA_20090212T204239_20090212T204240/TSX1_SAR__SSC______HS_S_SRA_20090212T204239_20090212T204240.xml'
    xml_path_slave = '/Users/timo/Desktop/SummerSchool/data/TSX/Australia/TSX1_SAR__SSC______HS_S_SRA_20090223T204240_20090223T204241/TSX1_SAR__SSC______HS_S_SRA_20090223T204240_20090223T204241.xml'

    lon = 1.31031108389863789E+02
    lat = -2.53445301056790093E+01
    height = 5.31374065174557472E+02

    # Transform to geocentric coordinates
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:4978", always_xy=True)
    X, Y, Z = transformer.transform(lon, lat, height)
    target = np.array([X, Y, Z])

    mas_spline_x, mas_spline_y, mas_spline_z, mas_first_line_time, mas_times = get_rd_parameters(xml_path_master)
    mas_range_time_1st_pix, mas_first_azimuth_datetime, mas_range_spacing, mas_azimuth_spacing = get_spacing(xml_path_master)

    mas_rg_pix, mas_az_pix = get_pixel(mas_spline_x, mas_spline_y, mas_spline_z, mas_first_line_time, mas_times, mas_range_time_1st_pix, mas_range_spacing, mas_azimuth_spacing, target)

    sl_spline_x, sl_spline_y, sl_spline_z, sl_first_line_time, sl_times = get_rd_parameters(xml_path_slave)
    sl_range_time_1st_pix, sl_first_azimuth_datetime, sl_range_spacing, sl_azimuth_spacing = get_spacing(
        xml_path_slave)

    sl_rg_pix, sl_az_pix = get_pixel(sl_spline_x, sl_spline_y, sl_spline_z, sl_first_line_time, sl_times,
                                       sl_range_time_1st_pix, sl_range_spacing, sl_azimuth_spacing, target)


    print(f'Master Pixel: {mas_rg_pix} / {mas_az_pix}')
    print(f'Slave Pixel: {sl_rg_pix} / {sl_az_pix}')
    print(f'Shift: {sl_rg_pix - mas_rg_pix} / {sl_az_pix - mas_az_pix}')