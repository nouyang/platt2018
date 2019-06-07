import pybullet as p
import pybullet_data
import time

import utils
from collections import deque
import numpy as np

physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# p.resetSimulation()

plane = p.loadURDF("plane.urdf")
mug = p.loadURDF("../ycb_urdf/urdf/mug.urdf")
knife = p.loadURDF("../ycb_urdf/urdf/knife.urdf")


# import os
# # this may take a while...
# os.system("git clone https://github.com/ros-industrial/kuka_experimental.git")

ur5UrdfPath = "./urdf/sisbot.urdf"
robot = p.loadURDF(ur5UrdfPath, [0, 0, 0], useFixedBase=1)
shelf = p.loadSDF("kiva_shelf/model.sdf", globalScaling=0.7)

joints, controlRobotiqC2, controlJoints, mimicParentName = utils.setup_sisbot(
    p, robot)

print('\n\n\n!_---------------------------------------------------!\n\n\n')


print('num joints', p.getNumJoints(robot))

for _ in range(10000):
    p.stepSimulation()
    time.sleep(1. / 240.)


p.disconnect()
