import mitsuba as mi
from matplotlib import pyplot as plt

mi.set_variant('scalar_rgb')

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

img = mi.render(scene, spp=16, sensor=sensor)

fig = plt.figure(figsize=(10, 7))
plt.imshow(img)
plt.show()