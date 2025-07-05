import mitsuba as mi
from matplotlib import pyplot as plt
import numpy as np
import drjit as dj

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

#specular reflection
w_dir = orig_ray.d - 2 * dj.dot(orig_ray.d, si.n) * si.n
w_mat_dir = -si.wi - 2 * dj.dot(-si.wi, up) * up
col_forward = bsdf.eval(bsdf_ctx, si, w_mat_dir)
#colxx2 = bsdf.eval(bsdf_ctx, si, wocalc2)

ray_forward = mi.Ray3f(si.p + w_dir * 0.00001, w_dir , si.t)
si_forward = scene.ray_intersect(ray_forward)
#mask_db_forward =  si_forward.is_valid()
bsdf_db = si_forward.bsdf(ray_forward)
w_db_mat_dir = -si_forward.wi - 2 * dj.dot(-si_forward.wi, up) * up
col_db_forward = bsdf_db.eval(bsdf_ctx, si_forward, w_db_mat_dir)
db_forward_signal = col_db_forward[0] * col_forward[0]
col_db_diffuse = bsdf_db.eval_diffuse_reflectance(si_forward)
db_diffuse_signal = col_db_diffuse[0]

w_db_dir_back = -(si_forward.p - orig_ray.o)
db_distance_back = np.linalg.norm(si_forward.p - orig_ray.o, axis=0)
dist_db_back = si_forward.time + si_forward.t + db_distance_back
pix = ((dist_db_back - min_dist) / step).numpy().astype(np.int32)
line = (((orig_ray.o[1] - -11) / 22) * 256).numpy().astype(np.int32)
mask_db_hack_ok = (pix >= 0) & (pix < 256) & (line >= 0) & (line < 256)
img[line[mask_db_hack_ok], pix[mask_db_hack_ok]] += db_diffuse_signal.numpy()[mask_db_hack_ok]

rand3_ray = mi.Ray3f(si_forward.p + w_db_dir_back * 0.00001, w_db_dir_back, si_forward.t)
si_db_back = scene.ray_intersect(rand3_ray)
mask_db_back = ~si_db_back.is_valid()
mask_db = mask_db_hack_ok & mask_db_back
img[line[mask_db], pix[mask_db]] += db_forward_signal.numpy()[mask_db]

print(f'Min: {np.min(img)} - max: {np.max(img)}')
fig = plt.figure(figsize=(10, 7))
plt.imshow(img, cmap='gray', vmin=0, vmax=1.0)
plt.savefig('/home/timo/Documents/radar-book/ss_lecture_notes/Images/sar_scene9.png', dpi=150)
plt.show()