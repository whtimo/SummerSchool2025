import rasterio
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    cos_path_master = '/Users/timo/Desktop/SummerSchool/data/TSX/Australia/TSX1_SAR__SSC______HS_S_SRA_20090212T204239_20090212T204240/IMAGEDATA/IMAGE_HH_SRA_spot_068.cos'
    resampled_tiff = '/Users/timo/Desktop/SummerSchool/australia/resampled.tiff'

    with rasterio.open(cos_path_master) as img_master:
        master = img_master.read(1)
        with rasterio.open(resampled_tiff) as img_slave:
            slave = img_slave.read(1)

            interefero = master * np.conj(slave)

            fig, ax1 = plt.subplots(1, 1, figsize=(12, 6))

            # Plot the estimated dx shifts
            im1 = ax1.imshow(np.angle(interefero), cmap='hsv', vmin = -3.14, vmax = 3.14)
            plt.show()


