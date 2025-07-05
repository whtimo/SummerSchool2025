import numpy as np
from scipy.ndimage import uniform_filter, generic_filter, laplace, gaussian_filter
#import pywt


def lee_filter(image, window_size=3, sigma_n=0.5):
    """
    Apply the Lee filter to a SAR image to reduce speckle noise.

    Parameters:
        image (np.ndarray): Input SAR image as a 2D numpy array.
        window_size (int): Size of the moving window (default is 3x3).
        sigma_n (float): Noise standard deviation (default is 0.5).

    Returns:
        np.ndarray: Filtered image as a 2D numpy array.
    """
    # Ensure the image is a numpy array
    image = np.asarray(image, dtype=np.float32)

    # Compute the local mean using a uniform filter
    local_mean = uniform_filter(image, size=window_size)

    # Compute the local variance
    local_sqr_mean = uniform_filter(image**2, size=window_size)
    local_variance = local_sqr_mean - local_mean**2

    # Compute the weighting factor (K)
    k = 1 - (sigma_n**2 / local_variance)
    k = np.clip(k, 0, 1)  # Ensure K is between 0 and 1

    # Apply the Lee filter formula
    filtered_image = local_mean + k * (image - local_mean)

    return filtered_image

import numpy as np
from scipy.ndimage import uniform_filter

def enhanced_lee_filter(image, window_size=3, sigma_n=0.5, cu=0.523, cmax=1.0):
    """
    Apply the Enhanced Lee filter to a SAR image to reduce speckle noise.

    Parameters:
        image (np.ndarray): Input SAR image as a 2D numpy array.
        window_size (int): Size of the moving window (default is 3x3).
        sigma_n (float): Noise standard deviation (default is 0.5).
        cu (float): Threshold for homogeneous areas (default is 0.523).
        cmax (float): Threshold for heterogeneous areas (default is 1.0).

    Returns:
        np.ndarray: Filtered image as a 2D numpy array.
    """
    # Ensure the image is a numpy array
    image = np.asarray(image, dtype=np.float32)

    # Compute the local mean and variance
    local_mean = uniform_filter(image, size=window_size)
    local_sqr_mean = uniform_filter(image**2, size=window_size)
    local_variance = local_sqr_mean - local_mean**2

    # Compute the coefficient of variation (C)
    local_std = np.sqrt(local_variance)
    C = local_std / local_mean

    # Compute the weighting factor (K)
    K = np.zeros_like(image)
    mask_homogeneous = C <= cu  # Homogeneous areas
    mask_heterogeneous = (C > cu) & (C <= cmax)  # Heterogeneous areas
    mask_edges = C > cmax  # Edges/point targets

    # Homogeneous areas: strong smoothing
    K[mask_homogeneous] = 1 - (sigma_n**2 / local_variance[mask_homogeneous])
    K[mask_homogeneous] = np.clip(K[mask_homogeneous], 0, 1)

    # Heterogeneous areas: moderate smoothing
    K[mask_heterogeneous] = np.exp(-(C[mask_heterogeneous] - cu) / (cmax - cu))

    # Edges/point targets: no smoothing (K = 0)
    K[mask_edges] = 0

    # Apply the Enhanced Lee filter formula
    filtered_image = local_mean + K * (image - local_mean)

    return filtered_image




def frost_filter(image, window_size=3, damping_factor=1.0):
    """
    Apply the Frost filter to a SAR image to reduce speckle noise.

    Parameters:
        image (np.ndarray): Input SAR image as a 2D numpy array.
        window_size (int): Size of the moving window (default is 3x3).
        damping_factor (float): Controls the strength of the exponential damping (default is 1.0).

    Returns:
        np.ndarray: Filtered image as a 2D numpy array.
    """
    # Ensure the image is a numpy array
    image = np.asarray(image, dtype=np.float32)

    # Compute the local mean and variance
    local_mean = uniform_filter(image, size=window_size)
    local_sqr_mean = uniform_filter(image**2, size=window_size)
    local_variance = local_sqr_mean - local_mean**2

    # Compute the coefficient of variation (C)
    local_std = np.sqrt(local_variance)
    C = local_std / local_mean

    # Compute the exponential damping factor (K)
    K = np.exp(-damping_factor * C)

    # Pad the image to handle borders
    pad_size = window_size // 2
    padded_image = np.pad(image, pad_size, mode='reflect')

    # Initialize the filtered image
    filtered_image = np.zeros_like(image)

    # Apply the Frost filter
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Extract the local window
            window = padded_image[i:i + window_size, j:j + window_size]
            # Compute the weights
            weights = np.exp(-damping_factor * C[i, j] * np.abs(window - local_mean[i, j]))
            # Normalize the weights
            weights /= np.sum(weights)
            # Compute the filtered pixel value
            filtered_image[i, j] = np.sum(weights * window)

    return filtered_image

def kuan_filter(image, window_size=3, sigma_n=0.5):
    """
    Apply the Kuan filter to a SAR image to reduce speckle noise.

    Parameters:
        image (np.ndarray): Input SAR image as a 2D numpy array.
        window_size (int): Size of the moving window (default is 3x3).
        sigma_n (float): Noise standard deviation (default is 0.5).

    Returns:
        np.ndarray: Filtered image as a 2D numpy array.
    """
    # Ensure the image is a numpy array
    image = np.asarray(image, dtype=np.float32)

    # Compute the local mean and variance
    local_mean = uniform_filter(image, size=window_size)
    local_sqr_mean = uniform_filter(image**2, size=window_size)
    local_variance = local_sqr_mean - local_mean**2

    # Compute the coefficient of variation (C)
    local_std = np.sqrt(local_variance)
    C = local_std / local_mean

    # Compute the noise variance to signal ratio (C_u)
    C_u = sigma_n  # Noise coefficient of variation

    # Compute the weighting factor (K)
    K = (1 - (C_u**2 / C**2)) / (1 + C_u**2)
    K = np.clip(K, 0, 1)  # Ensure K is between 0 and 1

    # Apply the Kuan filter formula
    filtered_image = local_mean + K * (image - local_mean)

    return filtered_image


def gamma_map_filter(image, window_size=3, enl=5.0):
    """
    Apply the Gamma MAP filter to a SAR image to reduce speckle noise.

    Parameters:
        image (np.ndarray): Input SAR image as a 2D numpy array.
        window_size (int): Size of the moving window (default is 3x3).
        enl (float): Equivalent Number of Looks (ENL) of the SAR image (default is 5.0).

    Returns:
        np.ndarray: Filtered image as a 2D numpy array.
    """
    # Ensure the image is a numpy array
    image = np.asarray(image, dtype=np.float32)

    # Compute the local mean and variance
    local_mean = uniform_filter(image, size=window_size)
    local_sqr_mean = uniform_filter(image**2, size=window_size)
    local_variance = local_sqr_mean - local_mean**2

    # Compute the coefficient of variation (C)
    local_std = np.sqrt(local_variance)
    C = local_std / local_mean

    # Compute the parameters for the Gamma MAP filter
    alpha = enl + 1  # Shape parameter of the Gamma distribution
    beta = enl / local_mean  # Rate parameter of the Gamma distribution

    # Compute the weighting factor (K)
    K = (local_variance - (local_mean**2 / enl)) / (local_variance * (1 + 1 / enl))
    K = np.clip(K, 0, 1)  # Ensure K is between 0 and 1

    # Apply the Gamma MAP filter formula
    filtered_image = local_mean + K * (image - local_mean)

    return filtered_image



# def wavelet_filter(image, wavelet='db1', level=3, threshold=0.1, mode='soft'):
#     """
#     Apply a wavelet-based filter to a SAR image to reduce speckle noise.
#
#     Parameters:
#         image (np.ndarray): Input SAR image as a 2D numpy array.
#         wavelet (str): Type of wavelet to use (default is 'db1' for Daubechies 1).
#         level (int): Decomposition level (default is 3).
#         threshold (float): Threshold value for noise reduction (default is 0.1).
#         mode (str): Thresholding mode ('soft' or 'hard', default is 'soft').
#
#     Returns:
#         np.ndarray: Filtered image as a 2D numpy array.
#     """
#     # Ensure the image is a numpy array
#     image = np.asarray(image, dtype=np.float32)
#
#     # Perform wavelet decomposition
#     coeffs = pywt.wavedec2(image, wavelet, level=level)
#
#     # Threshold the detail coefficients
#     coeffs_thresh = [coeffs[0]]  # Keep the approximation coefficients
#     for i in range(1, len(coeffs)):
#         # Apply thresholding to the detail coefficients
#         detail_coeffs = [pywt.threshold(c, threshold, mode=mode) for c in coeffs[i]]
#         coeffs_thresh.append(detail_coeffs)
#
#     # Reconstruct the image from the thresholded coefficients
#     filtered_image = pywt.waverec2(coeffs_thresh, wavelet)
#
#     return filtered_image



def anisotropic_diffusion_filter(image, num_iterations=10, kappa=50, gamma=0.1):
    """
    Apply the Anisotropic Diffusion Filter to a SAR image to reduce speckle noise.

    Parameters:
        image (np.ndarray): Input SAR image as a 2D numpy array.
        num_iterations (int): Number of iterations (default is 10).
        kappa (float): Gradient threshold parameter (default is 50).
        gamma (float): Step size for the update (default is 0.1).

    Returns:
        np.ndarray: Filtered image as a 2D numpy array.
    """
    # Ensure the image is a numpy array
    image = np.asarray(image, dtype=np.float32)

    # Initialize the filtered image
    filtered_image = image.copy()

    # Iteratively update the image
    for _ in range(num_iterations):
        # Compute the gradient in the x and y directions
        grad_x = np.roll(filtered_image, -1, axis=1) - filtered_image
        grad_y = np.roll(filtered_image, -1, axis=0) - filtered_image

        # Compute the diffusion coefficient
        c_x = np.exp(-(grad_x / kappa)**2)
        c_y = np.exp(-(grad_y / kappa)**2)

        # Update the image
        filtered_image += gamma * (
            c_x * np.roll(filtered_image, -1, axis=1) +
            c_y * np.roll(filtered_image, -1, axis=0) -
            (c_x + c_y) * filtered_image
        )

    return filtered_image




def bilateral_filter(image, sigma_spatial=5, sigma_intensity=50):
    """
    Apply the Bilateral Filter to a SAR image to reduce speckle noise.

    Parameters:
        image (np.ndarray): Input SAR image as a 2D numpy array.
        sigma_spatial (float): Standard deviation for spatial closeness (default is 5).
        sigma_intensity (float): Standard deviation for intensity similarity (default is 50).

    Returns:
        np.ndarray: Filtered image as a 2D numpy array.
    """
    # Ensure the image is a numpy array
    image = np.asarray(image, dtype=np.float32)

    # Initialize the filtered image
    filtered_image = np.zeros_like(image)

    # Define the window size based on sigma_spatial
    window_size = int(2 * sigma_spatial) + 1

    # Pad the image to handle borders
    pad_size = window_size // 2
    padded_image = np.pad(image, pad_size, mode='reflect')

    # Create a spatial Gaussian kernel
    x, y = np.meshgrid(np.arange(-pad_size, pad_size + 1), np.arange(-pad_size, pad_size + 1))
    spatial_kernel = np.exp(-(x**2 + y**2) / (2 * sigma_spatial**2))

    # Apply the bilateral filter
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Extract the local window
            window = padded_image[i:i + window_size, j:j + window_size]

            # Compute the intensity difference
            intensity_diff = window - image[i, j]

            # Compute the intensity Gaussian kernel
            intensity_kernel = np.exp(-(intensity_diff**2) / (2 * sigma_intensity**2))

            # Combine the spatial and intensity kernels
            weights = spatial_kernel * intensity_kernel

            # Normalize the weights
            weights /= np.sum(weights)

            # Compute the filtered pixel value
            filtered_image[i, j] = np.sum(weights * window)

    return filtered_image



def non_local_means_filter(image, patch_size=7, search_window_size=21, h=10):
    """
    Apply the Non-Local Means (NLM) Filter to a SAR image to reduce speckle noise.

    Parameters:
        image (np.ndarray): Input SAR image as a 2D numpy array.
        patch_size (int): Size of the patches used for comparison (default is 5).
        search_window_size (int): Size of the search window for finding similar patches (default is 11).
        h (float): Smoothing parameter that controls the decay of the weights (default is 10).

    Returns:
        np.ndarray: Filtered image as a 2D numpy array.
    """
    # Ensure the image is a numpy array
    image = np.asarray(image, dtype=np.float32)

    # Pad the image to handle borders
    pad_size = search_window_size // 2
    padded_image = np.pad(image, pad_size, mode='reflect')

    # Initialize the filtered image
    filtered_image = np.zeros_like(image)

    # Compute the patch radius
    patch_radius = patch_size // 2

    # Iterate over each pixel in the image
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Extract the reference patch
            reference_patch = padded_image[i:i + patch_size, j:j + patch_size]

            # Initialize the weighted sum and weight sum
            weighted_sum = 0.0
            weight_sum = 0.0

            # Iterate over the search window
            for x in range(i, i + search_window_size):
                for y in range(j, j + search_window_size):
                    # Extract the target patch
                    target_patch = padded_image[x:x + patch_size, y:y + patch_size]

                    if target_patch.shape == reference_patch.shape:
                        # Compute the Euclidean distance between patches
                        distance = np.sum((reference_patch - target_patch)**2)

                        # Compute the weight
                        weight = np.exp(-distance / (h**2))

                        # Update the weighted sum and weight sum
                        weighted_sum += weight * padded_image[x + patch_radius, y + patch_radius]
                        weight_sum += weight

            # Compute the filtered pixel value
            if weight_sum > 0:
                filtered_image[i, j] = weighted_sum / weight_sum
            else:
                filtered_image[i, j] = 0.0

    return filtered_image