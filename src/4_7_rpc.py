import rasterio
import numpy as np
import matplotlib.pyplot as plt
from rpcfit import rpc_fit
from datetime import datetime
import xml.etree.ElementTree as ET
from scipy.interpolate import CubicHermiteSpline
from scipy.optimize import minimize_scalar
from pyproj import Transformer
from scipy.constants import c

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


def get_img_coordinate(lon, lat, height, transformer, time_min, time_max, spline_x, spline_y, spline_z, first_line_time, azimuth_spacing, range_time_1st_pix, range_spacing ):
    X, Y, Z = transformer.transform(lon, lat, height)
    target = np.array([X, Y, Z])
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
        return -1, -1


cos_path = ''
transformer = Transformer.from_crs("EPSG:4326", "EPSG:4978", always_xy=True)

# Perform the transformation


t, p, v, reference_time = get_state_vectors(cos_path)
range_time_1st_pix, first_azimuth_datetime, range_spacing, azimuth_spacing = get_spacing(cos_path)

first_line_time = (first_azimuth_datetime - reference_time).total_seconds()

times = np.array(t)
positions = np.array(p)
velocities = np.array(v)
spline_x = CubicHermiteSpline(times, positions[:, 0], velocities[:, 0])
spline_y = CubicHermiteSpline(times, positions[:, 1], velocities[:, 1])
spline_z = CubicHermiteSpline(times, positions[:, 2], velocities[:, 2])

time_min = times[0]
time_max = times[-1]

min_lon = 1.30995614203457961E+02 - 0.01E+02
max_lon = 1.31073130794070636E+02 + 0.01E+02
min_lat = -2.53759574264882914E+01 - 0.01E+02
max_lat = -2.53145561714714979E+01 + 0.01E+02


lons = np.linspace(min_lon, max_lon, 80)
lats = np.linspace(min_lat, max_lat, 80)
heights = np.linspace(0, 2000, 80)

lon_grid, lat_grid, height_grid = np.meshgrid(lons, lats, heights)
# height_grid = np.random.uniform(low=0, high=2000, size=(400, 400))

locations = []
targets = []

print('Train RPC')
for i in range(80):
    for j in range(80):
        for k in range(80):
            sar_x, sar_y = get_img_coordinate(lon_grid[i, j, k], lat_grid[i, j, k], height_grid[i, j, k], transformer, time_min, time_max, spline_x, spline_y, spline_z,
                                      first_line_time, azimuth_spacing, range_time_1st_pix, range_spacing)

            locations.append([lon_grid[i, j, k], lat_grid[i, j, k], height_grid[i, j, k]])
            targets.append([sar_x, sar_y])

locs_train = np.array(locations)
target_train = np.array(targets)

rpc_calib, log = rpc_fit.calibrate_rpc(target_train, locs_train, separate=False, tol=1e-10
                                       , max_iter=20, method='initLcurve'
                                       , plot=False, orientation='projloc', get_log=True)

rmse_err, mae, planimetry = rpc_fit.evaluate(rpc_calib, locs_train, target_train)
print('Training set :   Mean X-RMSE {:e}     Mean Y-RMSE {:e}'.format(*rmse_err))



lon = 1.31031108389863789E+02
lat = -2.53445301056790093E+01
height = 5.31374065174557472E+02
x, y = rpc_calib.projection(lon, lat, height)
print(f'Pixel: {x} / {y} | Expected center = 5445 / 3091')

