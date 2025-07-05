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
yv = np.linspace(-11, 11, 256)

for y in range(256):
	print(f'{y} / 256')
	orig_ray = mi.Ray3f(np.asarray([20, yv[y], 20]), xv)
	pw_orig = np.full(256, 1.00)

	rays = []
	rays.append((orig_ray, pw_orig))

	for bounce in range(bounces):
		new_rays = []

		for ray, pw in rays:
			si = scene.ray_intersect(ray)
			bsdf = si.bsdf(ray)
			col = bsdf.eval(bsdf_ctx, si, -(si.p - orig_ray.o))
			distance_back = np.linalg.norm(si.p - orig_ray.o, axis=0)
			dist = si.time + si.t + distance_back
			pix = ((dist.numpy() - min_dist) / step).astype(np.int32)
			mask2 = (pix >= 0) & (pix < 256)
			if mask2.any():
				img[y, pix[mask2]] += col[0].numpy()[mask2] * pw[mask2]

				for a in range(rays_per_bounce):
					bsdf_sample, bsdf_weight = bsdf.sample(bsdf_ctx,
														si,
														sampler.next_1d(),
														sampler.next_2d())

					col = bsdf.eval(bsdf_ctx, si, bsdf_sample.wo)
					signal = (col[0] * pw).numpy()
					rand2_ray = mi.Ray3f(si.p + bsdf_sample.wo * 0.01,
										  bsdf_sample.wo, si.time + si.t)
					new_rays.append((rand2_ray, signal))


		rays = new_rays


fig = plt.figure(figsize=(10, 7))
plt.imshow(img, cmap='gray', vmin=0, vmax=5.01)
plt.show()