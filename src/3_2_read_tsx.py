import rasterio
import numpy as np
import matplotlib.pyplot as plt

cos_path = '<filename>'

with rasterio.open(cos_path) as img:
    data = img.read(1)
    print(f'Data Shape: {data.shape} \ Data Type: {data.dtype}')

    plt.figure(figsize=(8,6))
    plt.imshow(abs(data), cmap='gray', vmin=0, vmax=256)
    plt.show()


