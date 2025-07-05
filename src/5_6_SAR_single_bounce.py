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

xv = np.asarray([ np.linspace(0.2, 0.8, 256), np.full(256, 0),np.full(256, -0.5)])
#xv = np.asarray([0.42, 0, -0.5])
magnitude = np.linalg.norm(xv, axis=0)
xv = xv / magnitude
#yv = np.linspace(-11, 11, 256)
yv = np.asarray([np.full(256, -20), np.linspace(-11, 11, 256), np.full(256, 20)])
#yv = np.asarray([-20, 0, 20])
xx0, yy0 = np.meshgrid(xv[0], yv[0])
xx1, yy1 = np.meshgrid(xv[1], yv[1])
xx2, yy2 = np.meshgrid(xv[2], yv[2])

xx0 = xx0.flatten()
xx1 = xx1.flatten()
xx2 = xx2.flatten()
yy0 = yy0.flatten()
yy1 = yy1.flatten()
yy2 = yy2.flatten()

xx3 = np.asarray([xx0, xx1, xx2])
yy3 = np.asarray([yy0, yy1, yy2])

#test_ray = mi.Ray3f([0,0,0],[-1.0, 0.0, 1.0])
#si2 = scene_sar.ray_intersect(test_ray)

orig_ray = mi.Ray3f(yy3, xx3)
si = scene.ray_intersect(orig_ray)
bsdf = si.bsdf(orig_ray)
up = np.asarray([0, 0, 1])
up = mi.Vector3f(up)

#single bounce back
col_diffuse = bsdf.eval_diffuse_reflectance(si)
distance_back = np.linalg.norm(si.p - orig_ray.o, axis=0)
dist = si.time + si.t + distance_back
pix = ((dist - min_dist) / step).numpy().astype(np.int32)
line = (((orig_ray.o[1] - -11) / 22) * 256).numpy().astype(np.int32)
signal = col_diffuse[0].numpy()
mask_sb_hack_ok = (pix >= 0) & (pix < 256) & (line >= 0) & (line < 256)
img[line[mask_sb_hack_ok], pix[mask_sb_hack_ok]] += signal[mask_sb_hack_ok]


print(f'Min: {np.min(img)} - max: {np.max(img)}')
fig = plt.figure(figsize=(10, 7))
plt.imshow(img, cmap='gray', vmin=0, vmax=1.0)
plt.savefig('/home/timo/Documents/radar-book/ss_lecture_notes/Images/sar_scene5.png', dpi=150)
plt.show()