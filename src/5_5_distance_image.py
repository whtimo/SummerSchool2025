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
			'reflectance': {'type': 'rgb', 'value': [0.0, 0.5, 0.0]},
		},
	},
	'box': {
		'type': 'cube',
		'to_world': mi.ScalarTransform4f().scale([2, 2, 4]),
		'bsdf': {
			'type': 'diffuse',
			'reflectance': {'type': 'rgb', 'value': [0.4, 0.0, 0.0]},
		},
	},
	'sensor': {
		'type': 'perspective',
		'fov': 39.3077,
		'to_world': mi.ScalarTransform4f().look_at(
		origin=[20, 0, 20],
		target=[0, 0, 0],
		up=[0, 0, 1]),
		'sampler': {
			'type': 'independent',
			'sample_count': 16
		}
}})

sensors = scene.sensors()
sensor = sensors[0]
sampler = sensor.sampler()

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
	mask = si.is_valid()
	distance = si.t.numpy()[mask] * 2

	if len(distance) > 0:
		min_dist = min(filter(lambda x: x != np.inf
										and not np.isnan(x), distance))
		max_dist = max(filter(lambda x: x != np.inf
										and not np.isnan(x), distance))

		img[y, :][mask] = distance

print(f'Minimum distance = {min_dist} / maximum distance = {max_dist}')

fig = plt.figure(figsize=(10, 7))
plt.imshow(img, cmap='gray', vmin=min_dist, vmax=max_dist)
plt.show()