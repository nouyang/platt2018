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
# table = p.loadURDF("table/table.urdf")

table = p.loadURDF("table/table.urdf", 0.5000000,
                   0.00000, -.820000, 0.000000, 0.000000, 0.0, 1.0)

tray = p.loadURDF("tray/tray.urdf", 0.640000, 0.075000, -
                  0.190000, 0.000000, 0.000000, 1.000000, 0.000000)

# table2 = p.loadURDF("../ycb_urdf/urdf/table.urdf")
# cube = p.loadURDF("cube.urdf") # can't figure out where this is
# list of more build in urdf: https://github.com/bulletphysics/bullet3/tree/master/data
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
