import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl

fig = plt.figure()
ax = fig.add_subplot(111)

r1 = patches.Rectangle((0, 0), 20, 40, color="blue", alpha=0.50)
r2 = patches.Rectangle((0, 0), 20, 40, color="red",  alpha=0.50)

t2 = mpl.transforms.Affine2D().rotate_around(10, 20, -45) + ax.transData
r2.set_transform(t2)

ax.add_patch(r1)
ax.add_patch(r2)

plt.xlim(-20, 60)
plt.ylim(-20, 60)

plt.grid(True)

plt.show()
