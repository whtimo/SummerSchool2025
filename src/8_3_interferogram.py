import rasterio
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    cos_path_master = ''
    resampled_tiff = ''

    with rasterio.open(cos_path_master) as img_master:
        master = img_master.read(1)
        with rasterio.open(resampled_tiff) as img_slave:
            slave = img_slave.read(1)

            interefero = master * np.conj(slave)

            fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))

            # Plot the estimated dx shifts
            im1 = ax1.imshow(np.angle(interefero), cmap='hsv', vmin = -3.14, vmax = 3.14)
            plt.show()


