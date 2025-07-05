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
	'sensor': {
		'type': 'orthographic',
		'to_world': mi.ScalarTransform4f().look_at(
			origin=[20, 0, 20],
			target=[0, 0, 0],
			up=[0, 0, 1]
		) @ mi.ScalarTransform4f().scale([10, 10, 1]),

	}
})

sensors = scene.sensors()
sensor = sensors[0]
sampler = sensor.sampler()

min_dist = 44
max_dist = 72
step = (max_dist - min_dist) / 256.0
bounces = 2
rays_per_bounce = 32

bsdf_ctx = mi.BSDFContext()

img = np.zeros((256, 256), dtype=np.float32)

xv = np.asarray([np.linspace(-0.2, -0.8, 256),
				 np.full(256, 0),
				 np.full(256, -0.5)])

magnitude = np.linalg.norm(xv, axis=0)
xv = xv / magnitude

yv = np.asarray([np.full(256, 20),
				 np.linspace(-11, 11, 256),
				 np.full(256, 20)])

xx0, yy0 = np.meshgrid(xv[0, :], yv[0, :])
xx1, yy1 = np.meshgrid(xv[1, :], yv[1, :])
xx2, yy2 = np.meshgrid(xv[2, :], yv[2, :])

xx0 = xx0.flatten()
xx1 = xx1.flatten()
xx2 = xx2.flatten()
yy0 = yy0.flatten()
yy1 = yy1.flatten()
yy2 = yy2.flatten()

xx3 = np.asarray([xx0, xx1, xx2])
yy3 = np.asarray([yy0, yy1, yy2])

orig_ray = mi.Ray3f(yy3, xx3)
si = scene.ray_intersect(orig_ray)
bsdf = si.bsdf(orig_ray)
col = bsdf.eval(bsdf_ctx, si, -(si.p - orig_ray.o))
distance_back = np.linalg.norm(si.p - orig_ray.o, axis=0)
dist = si.time + si.t + distance_back
pix = ((dist.numpy() - min_dist) / step).astype(np.int32)
line = (((orig_ray.o[1].numpy() - -11) / 22) * 256).astype(np.int32)
mask2 = (pix >= 0) & (pix < 256) & (line >= 0) & (line < 256)
img[line[mask2], pix[mask2]] += col[0].numpy()[mask2]

si_p_arr = []
wo_arr = []
time_arr = []
orig_ray_arr = []
signal_arr = []

for i in range(rays_per_bounce):
	bsdf_sample, bsdf_weight = bsdf.sample(bsdf_ctx, si,
										   sampler.next_1d(),
										   sampler.next_2d())

	col = bsdf.eval(bsdf_ctx, si, bsdf_sample.wo)
	signal = (col[0] * 1.0)
	weight = bsdf_weight[0]
	signalnp = (signal + weight).numpy()
	mask3 = signalnp > 0.001
	si_p_np = si.p.numpy()
	si_p_np = np.asarray([si_p_np[0, :][mask3],
						  si_p_np[1, :][mask3],
						  si_p_np[2, :][mask3]])

	wp_np = bsdf_sample.wo.numpy()
	wp_np = np.asarray([wp_np[0, :][mask3],
						wp_np[1, :][mask3],
						wp_np[2, :][mask3]])

	time_np = si.t.numpy()[mask3]
	orig_ray_np = orig_ray.o.numpy()
	orig_ray_np = np.asarray([orig_ray_np[0, :][mask3],
							  orig_ray_np[1, :][mask3],
							  orig_ray_np[2, :][mask3]])

	si_p_arr.append(si_p_np)
	wo_arr.append(wp_np)
	time_arr.append(time_np)
	orig_ray_arr.append(orig_ray_np)
	signal_arr.append(signalnp[mask3])

si_p = np.concatenate(si_p_arr, axis=1)
wo = np.concatenate(wo_arr, axis=1)
orig_o = np.concatenate(orig_ray_arr, axis=1)
time = np.concatenate(time_arr)
signal = np.concatenate(signal_arr)

rand2_ray = mi.Ray3f(si_p + wo * 0.01, wo, time)
si = scene.ray_intersect(rand2_ray)
bsdf = si.bsdf(rand2_ray)
col = bsdf.eval(bsdf_ctx, si, -(si.p - orig_o))
distance_back = np.linalg.norm(si.p - orig_o, axis=0)
dist = si.time + si.t + distance_back
pix = ((dist.numpy() - min_dist) / step).astype(np.int32)
line = (((orig_o[1, :] - -11) / 22) * 256).astype(np.int32)
mask4 = (pix >= 0) & (pix < 256) & (line >= 0) & (line < 256)
img[line[mask4], pix[mask4]] += col[0].numpy()[mask4] * signal[mask4]

fig = plt.figure(figsize=(10, 7))
plt.imshow(img, cmap='gray', vmin=0, vmax=7.01)
plt.show()