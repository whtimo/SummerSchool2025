import numpy as np
from scipy.signal.windows import kaiser
import pandas as pd
import rasterio
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import RANSACRegressor, LinearRegression, HuberRegressor, TheilSenRegressor
from sklearn.pipeline import make_pipeline

def sinc_kernel(size, beta=5.0):
    """
    Generate a 2D windowed sinc kernel.
    Args:
        size (int): Size of the kernel (odd number).
        beta (float): Kaiser window beta parameter.
    Returns:
        ndarray: 2D windowed sinc kernel.
    """
    if size % 2 == 0:
        raise ValueError("Kernel size must be odd.")
    window = kaiser(size, beta)
    kernel_1d = np.sinc(np.linspace(-2, 2, size)) * window
    kernel_2d = np.outer(kernel_1d, kernel_1d)
    return kernel_2d / np.sum(kernel_2d)  # Normalize

def resample_sar_image(slave_image: np.ndarray, master_shape: tuple, ransac_pipeline_dx, ransac_pipeline_dy, kernel_size=11, output = None) -> np.ndarray:
    """
    Resample a slave SAR image to match the master image shape using predicted shifts.
    Args:
        slave_image (ndarray): Slave image as a complex ndarray.
        master_shape (tuple): Shape of the master image (height, width).
        ransac_pipeline: Trained RANSAC pipeline for shift prediction.
        output_path (str): Path to save the resampled image.
        kernel_size (int): Size of the sinc kernel (must be odd).
    """
    # Generate the sinc kernel
    kernel = sinc_kernel(kernel_size)

    # Create an empty array for the resampled image
    resampled_image = np.zeros(master_shape, dtype=np.complex64)

    # Get coordinates for the master image
    height, width = master_shape
    x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))

    # Predict shifts for all pixels
    coords = np.column_stack([x_coords.ravel(), y_coords.ravel()])
    shifts_dx = ransac_pipeline_dx.predict(coords)
    shifts_dy = ransac_pipeline_dy.predict(coords)
    dx = shifts_dx.reshape(height, width)
    dy = shifts_dy.reshape(height, width)

    # Resample the slave image
    for y in range(height):
        print(f'Resampling {y} / {height}')

        for x in range(width):
            # Calculate the corresponding position in the slave image
            x_slave = x + dx[y, x]
            y_slave = y + dy[y, x]

            # Interpolate using the sinc kernel
            x_start = int(np.floor(x_slave)) - kernel_size // 2
            y_start = int(np.floor(y_slave)) - kernel_size // 2
            x_end = x_start + kernel_size
            y_end = y_start + kernel_size

            # Extract the region from the slave image
            if x_start >= 0 and y_start >= 0 and x_end < slave_image.shape[1] and y_end < slave_image.shape[0]:
                region = slave_image[y_start:y_end, x_start:x_end]
                resampled_image[y, x] = np.sum(region * kernel)

    return resampled_image

def parameter_estimation(shifts, degree:int = 2):

    coords = shifts[['masterX', 'masterY', 'shiftX', 'shiftY']].to_numpy()

    print(f'Mean: {np.mean(coords[:,2])}, {np.mean(coords[:,3])}')

    ransac_pipeline_dx = make_pipeline(
        PolynomialFeatures(degree),
        RANSACRegressor(
            estimator=LinearRegression(),
            residual_threshold=0.001,  # Tune based on noise level
            max_trials=10000
        )
    )

    ransac_pipeline_dy = make_pipeline(
        PolynomialFeatures(degree),
        RANSACRegressor(
            estimator=LinearRegression(),
            residual_threshold=0.001,
            max_trials=10000
        )
    )

    # Fit models
    ransac_pipeline_dx.fit(coords[:,0:2], coords[:,2])
    ransac_pipeline_dy.fit(coords[:,0:2], coords[:,3])

    return ransac_pipeline_dx, ransac_pipeline_dy

if __name__ == "__main__":

    cos_path_master = '/Users/timo/Desktop/SummerSchool/data/TSX/Australia/TSX1_SAR__SSC______HS_S_SRA_20090212T204239_20090212T204240/IMAGEDATA/IMAGE_HH_SRA_spot_068.cos'
    cos_path_slave = '/Users/timo/Desktop/SummerSchool/data/TSX/Australia/TSX1_SAR__SSC______HS_S_SRA_20090223T204240_20090223T204241/IMAGEDATA/IMAGE_HH_SRA_spot_068.cos'
    shifts_file = '/Users/timo/Desktop/SummerSchool/australia/shifts_py.csv'

    out_tiff = '/Users/timo/Desktop/SummerSchool/australia/resampled.tiff'
    shifts = pd.read_csv(shifts_file)

    print(f'Shifts mean: {shifts['shiftX'].to_numpy().mean().round(2)}, {shifts['shiftY'].to_numpy().mean().round(2)}')

    est_dx, est_dy = parameter_estimation(shifts, 2)

    print('Read data')
    with rasterio.open(cos_path_master) as img_master:
        master = img_master.read(1)
        with rasterio.open(cos_path_slave) as img_slave:
            slave = img_slave.read(1)

            print('Start resampling')
            slave_resample = resample_sar_image(slave, master.shape, est_dx, est_dy)
            # slave_resample = np.zeros(master.shape, dtype=np.complex64)

            meta = {
                'driver': 'GTiff',  # GeoTIFF driver
                'dtype': slave_resample.dtype,  # Data type of the array
                'width': slave_resample.shape[1],  # Width of the raster
                'height': slave_resample.shape[0],  # Height of the raster
                'count': 1,  # Number of bands
            }

            with rasterio.open(out_tiff, 'w', **meta) as dst:
                dst.write(slave_resample, 1)