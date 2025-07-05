import mitsuba as mi
from matplotlib import pyplot as plt
import numpy as np

mi.set_variant('llvm_ad_rgb')

scene = mi.load_dict({
    'type': 'scene',
    'integrator': {'type': 'path'},
    'light': {'type': 'constant'},
    'ground': {
        'type': 'rectangle',
        'to_world': mi.ScalarTransform4f().scale([10.0, 10.0, 1.0]),
        'bsdf': {
            'type': 'diffuse',
            'reflectance': {
                'type': 'rgb',
                'value': 0.3
            }
        },
    },
    'box': {
        'type': 'cube',
        'to_world': mi.ScalarTransform4f().scale([2, 2, 4]),
        'bsdf': {
            'type': 'diffuse',
            'reflectance': {
                'type': 'rgb',
                'value': 0.5
            }
        },
    },
})



min_dist = 44
max_dist = 72
step = (max_dist - min_dist) / 256.0

bsdf_ctx = mi.BSDFContext()

img = np.zeros((256, 256), dtype=np.float32)

xv = np.asarray([np.linspace(-0.2, -0.8, 256),
                 np.full(256, 0),
                 np.full(256, -0.5)])

magnitude = np.linalg.norm(xv, axis=0)
xv = xv / magnitude
yv = np.linspace(-11, 11, 256)

for y in range(256):
    ray = mi.Ray3f(np.asarray([20, yv[y], 20]), xv)
    si = scene.ray_intersect(ray)
    if si.is_valid().numpy().any():
        bsdf = si.bsdf(ray)
        col = bsdf.eval(bsdf_ctx, si, -(si.p - ray.o))
        dist = si.t * 2
        pix = ((dist.numpy() - min_dist) / step).astype(np.int32)
        mask2 = (pix >= 0) & (pix < 256)
        img[y, pix[mask2]] += col[0].numpy()[mask2]

fig = plt.figure(figsize=(10, 7))
plt.imshow(img, cmap='gray', vmin=0, vmax=5.01)
plt.show()