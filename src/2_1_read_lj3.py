import rasterio
import numpy as np
import matplotlib.pyplot as plt

tiff_path = '<filename>'

with rasterio.open(tiff_path) as img:
    data_r = img.read(4)
    data_g = img.read(3)
    data_b = img.read(2)

    data = np.dstack((data_r, data_g, data_b))
    print(f'Data Shape: {data.shape} \ Data Type: {data.dtype}')

    plt.figure(figsize=(6,8))
    plt.imshow(data)
    plt.show()


