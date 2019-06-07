# guide at http://alexanderfabisch.github.io/pybullet.html

import pybullet as p
import pybullet_data

physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# p.resetSimulation()

plane = p.loadURDF("plane.urdf")


# import os
# # this may take a while...
# os.system("git clone https://github.com/ros-industrial/kuka_experimental.git")

robot = p.loadURDF("kuka_kr210_support/urdf/kr210l150.urdf", [0, 0, 0],
                   useFixedBase=1)

pos, orn = p.getBasePositionAndOrientation(robot)
print('\n\n\n!_---------------------------------------------------!\n\n\n')
print('orientation', orn)

print('num joints', p.getNumJoints(robot))

joint_index = 2

# print(len(p.getJointInfo(robot, joint_index)))

# from quickstart guide
# https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit#
# jointIndex
# jointName
# jointType
# qIndex
# uIndex
# flags
# jointDamping
# jointFriction
# jointLowerLimit
# jointUpperLimit
# jointMaxForce
# jointMaxVelocity
# linkName
# jointAxis
# parentFramePos
# parentFrameOrn
# parentIndex

_, name, joint_type, _, _, _, _, _, lower_limit, upper_limit, _, _, _, _, _, _, _ = \
    p.getJointInfo(robot, joint_index)

print('joint info: idx, name, type, low limit, up limit',
      joint_index, name, joint_type, lower_limit, upper_limit)

joint_positions = [j[0] for j in p.getJointStates(robot, range(6))]
print('joint state positions', joint_positions)


world_position, world_orientation = p.getLinkState(robot, 2)[:2]
print('joint world position', world_position)

p.setGravity(0, 0, -9.81)   # everything should fall down
# this slows everything down, but let's be accurate...
p.setTimeStep(0.0001)  # 1/240 by default
p.setRealTimeSimulation(0)  # we want to be faster than real time :)

p.setJointMotorControlArray(
    robot, range(6), p.POSITION_CONTROL,
    targetPositions=[0.1] * 6)

for _ in range(10000):
    p.stepSimulation()

p.disconnect()
