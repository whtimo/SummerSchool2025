import rasterio
import matplotlib.pyplot as plt
import numpy as np
from rasterio.vrt import WarpedVRT

filename = '<filename>'

with rasterio.open(filename) as img:
    data_orig = img.read(1)
    print(f'Data Shape: {data_orig.shape} \ Data Type: {data_orig.dtype}')

    # Create a WarpeVRT to reproject using RPCs
    with WarpedVRT(img, crs='EPSG:4326', rpc=True) as vrt:
        data = vrt.read(1)
        print(f'Data Shape: {data.shape} \ Data Type: {data.dtype}')

        left, bottom, right, top = rasterio.transform.array_bounds(
            vrt.height, vrt.width, vrt.transform
        )

        print(f'Bounds: left: {left}, bottom: {bottom}, right: {right}, top: {top}')
        print(f'Width: {vrt.width}, Height: {vrt.height}')

        # Plot the image with geographic coordinates
        plt.figure(figsize=(8, 10))
        plt.imshow(data, extent=(left, right, top, bottom),
                   cmap='gray', vmin=0, vmax=1024)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.show()	