import mitsuba as mi
import matplotlib.pyplot as plt

mi.set_variant('scalar_rgb')
img = mi.render(mi.load_dict(mi.cornell_box()))
#mi.Bitmap(img).write('cbox.exr')

plt.figure(figsize=(8, 6))
plt.imshow(img*5)
plt.show()