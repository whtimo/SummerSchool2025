import numpy as np
from scipy import fftpack
import random
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import RANSACRegressor, LinearRegression, HuberRegressor, TheilSenRegressor
from sklearn.pipeline import make_pipeline
import pandas as pd

import rasterio

from skimage import data
from skimage.registration import phase_cross_correlation
from skimage.registration._phase_cross_correlation import _upsampled_dft
from scipy.ndimage import fourier_shift
import matplotlib.pyplot as plt

def get_random_points(count: int, left: float, right: float, top: float, bottom: float, height: float = 0):

    result = []
    for _ in range(count):
        lon = random.uniform(left, right)
        lat = random.uniform(top, bottom)
        result.append((lat, lon, height))

    return result


def subpixel_shift(master: np.ndarray, slave: np.ndarray, search_pix: int, search_line: int,  upsample_factor=16):
    """
    Compute sub-pixel shift between master and slave images.
    Args:
        master (ndarray): Smaller master window (2D array).
        slave (ndarray): Larger search window (2D array).
        upsample_factor (int): Factor for upsampling (default: 16).
    Returns:
        ndarray: Sub-pixel shift [dy, dx].
    """

    m, n = slave.shape
    upsampled_s = fftpack.fftshift(fftpack.fft2(slave, (m * upsample_factor, n * upsample_factor)))
    upsampled_slave = fftpack.ifft2(upsampled_s).real
    f_slave = fftpack.fft2(upsampled_slave, upsampled_slave.shape)

    max_corr = 0
    max_shifts = (0,0)

    for l in range(-search_line, search_line+1, 1):
        for p in range(-search_pix, search_pix+1, 1):
            # Upsample region using FFT
            y_sub = l + search_line
            x_sub = p + search_line

            upsampled_m = fftpack.fftshift(fftpack.fft2(master[l:l+m, p:p+n], (m*upsample_factor, n*upsample_factor)))
            upsampled_master = fftpack.ifft2(upsampled_m).real

            f_master = fftpack.fft2(upsampled_master, upsampled_master.shape)
            cross_corr = fftpack.fftshift(fftpack.ifft2(f_slave * np.conj(f_master))).real

            # Integer-pixel shift
            max_idx = np.argmax(cross_corr)
            peak = np.unravel_index(max_idx, cross_corr.shape)
            center = np.array(cross_corr.shape) // 2
            shifts = (np.array(peak) - center) / upsample_factor
            max_corr_sub = cross_corr[peak]
            if max_corr_sub > max_corr:
                max_corr = max_corr_sub
                max_shifts = (shifts[0] + p, shifts[1] + l)

    return max_shifts, max_corr


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

    cos_path_master = ''
    cos_path_slave = ''
    shifts_file = ''

    window_size_x = 128
    window_size_y = 128
    no_points = 800
    win_radius_x = window_size_x // 2
    win_radius_y = window_size_y // 2
    orbit_shift_x = -461
    orbit_shift_y = -95


    with rasterio.open(cos_path_master) as img_master:
        master_data = img_master.read(1)
        with rasterio.open(cos_path_slave) as img_slave:
            slave_data = img_slave.read(1)
            shifts = pd.DataFrame(columns=['masterX', 'masterY', 'shiftX', 'shiftY'])

            for i in range(no_points):
                m_x = random.randint(window_size_x, img_master.width - window_size_x)
                m_y = random.randint(window_size_y, img_master.height - window_size_y)
                s_x, s_y = m_x + orbit_shift_x, m_y + orbit_shift_y

                if s_x > window_size_x and s_x < img_master.width - window_size_x and s_y > window_size_y and s_y < img_master.height - window_size_y:
                    mas_subset = np.abs(
                        master_data[m_y - win_radius_y:m_y + win_radius_y, m_x - win_radius_x:m_x + win_radius_x])
                    sl_subset = np.abs(
                        slave_data[s_y - win_radius_y:s_y + win_radius_y, s_x - win_radius_x:s_x + win_radius_x])

                    shift, _, _ = phase_cross_correlation(mas_subset, sl_subset, upsample_factor=100)
                    c_shift_x = s_x - m_x
                    c_shift_y = s_y - m_y
                    shifts.loc[len(shifts)] = [m_x, m_y, c_shift_x - shift[1], c_shift_y - shift[0]]
                    print(f'Estimating Subpixel Shifts: {len(shifts)} - {c_shift_x - shift[1]}, {c_shift_y - shift[0]}')

                # Create a figure with two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

            # Scatter plot for shiftX
            sc1 = ax1.scatter(shifts['masterX'], shifts['masterY'], c=shifts['shiftX'], cmap='viridis')
            ax1.set_title('Shift in X')
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            fig.colorbar(sc1, ax=ax1, label='shiftX')

            # Scatter plot for shiftY
            sc2 = ax2.scatter(shifts['masterX'], shifts['masterY'], c=shifts['shiftY'], cmap='viridis')
            ax2.set_title('ShiftY in Y')
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            fig.colorbar(sc2, ax=ax2, label='shiftY')

            # Show the plot
            plt.tight_layout()
            plt.show()

            master_shape = img_master.shape  # (height, width)

            # Generate sample data for x and y coordinates
            x = np.linspace(0, master_shape[1] - 1, int(master_shape[1] / 100))
            y = np.linspace(0, master_shape[0] - 1, int(master_shape[0] / 100))
            xx, yy = np.meshgrid(x, y)
            X = np.column_stack((xx.ravel(), yy.ravel()))

            est_dx_1, est_dy_1 = parameter_estimation(shifts, 1)
            pred_dx1 = est_dx_1.predict(X).reshape(xx.shape)
            pred_dy1 = est_dy_1.predict(X).reshape(yy.shape)

            # Create a figure with two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

            # Plot the estimated dx shifts
            im1 = ax1.imshow(pred_dx1,
                             extent=[0, master_shape[1], 0, master_shape[0]],
                             origin='lower', cmap='viridis')
            ax1.set_title('Estimated Shifts (dx) with degree = 1')
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            fig.colorbar(im1, ax=ax1, label='dx')

            # Plot the estimated dy shifts
            im2 = ax2.imshow(pred_dy1,
                             extent=[0, master_shape[1], 0, master_shape[0]],
                             origin='lower', cmap='viridis')
            ax2.set_title('Estimated Shifts (dy) with degree = 1')
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            fig.colorbar(im2, ax=ax2, label='dy')

            # Show the plot
            plt.tight_layout()
            plt.show()

            est_dx_2, est_dy_2 = parameter_estimation(shifts, 2)
            pred_dx2 = est_dx_2.predict(X).reshape(xx.shape)
            pred_dy2 = est_dy_2.predict(X).reshape(yy.shape)

            # Create a figure with two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

            # Plot the estimated dx shifts
            im1 = ax1.imshow(pred_dx2,
                             extent=[0, master_shape[1], 0, master_shape[0]],
                             origin='lower', cmap='viridis')
            ax1.set_title('Estimated Shifts (dx) with degree = 2')
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            fig.colorbar(im1, ax=ax1, label='dx')

            # Plot the estimated dy shifts
            im2 = ax2.imshow(pred_dy2,
                             extent=[0, master_shape[1], 0, master_shape[0]],
                             origin='lower', cmap='viridis')
            ax2.set_title('Estimated Shifts (dy) with degree = 2')
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            fig.colorbar(im2, ax=ax2, label='dy')

            # Show the plot
            plt.tight_layout()
            plt.show()

            shifts.to_csv(shifts_file, index=False)