import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl

import copy

fig = plt.figure()
ax = fig.add_subplot(111)


t_start = ax.transData
r1 = patches.Rectangle((0, 0), 40, 20, color="blue", alpha=0.50)
ax.add_patch(r1)

r2 = patches.Rectangle((0, 0), 40, 20, color="red", alpha=0.50)
t2 = mpl.transforms.Affine2D().rotate_around(20, 10, np.deg2rad(5)) + t_start
r2.set_transform(t2)
ax.add_patch(r2)

r3 = patches.Rectangle((0, 0), 40, 20, angle=0, color="green", alpha=0.50)
t3 = mpl.transforms.Affine2D().rotate_around(20, 10, np.deg2rad(10)) + t_start
r3.set_transform(t3)
ax.add_patch(r3)

r3 = patches.Rectangle((0, 0), 40, 20, angle=0, color="orange", alpha=0.50)
t3 = mpl.transforms.Affine2D().rotate_around(20, 10, np.deg2rad(-5)) + t_start
r3.set_transform(t3)
ax.add_patch(r3)

plt.xlim(-20, 60)
plt.ylim(-20, 60)
plt.axis('equal')

#ax.set_aspect('equal', 'box')

plt.show()
input('Close all?')
plt.close('all')
