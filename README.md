# grasping_regions
Final year dissertation project for grasping regions and determining the most efficient path to a cylindrical object to pick-and-place it somewhere else.

Grasping objects is one of the basic and most common tasks that a robot (such as manipulator arms or mobile manipulators) has to do. While it is considered a simple task for humans, sometimes it could be a challenging robot operation. For a robot, grasping an object normally involves perception to detect the object, planning to approach and grasp the object, and control to drive the robot through the planned motion. With the increasing popularity of learning techniques, new frameworks/libraries have been proposed to automatically detect the best grasping poses of an object.

This project aims to implement a learning and computer vision-based framework to determine the grasp affordances of an object. More specifically, the framework uses computer vision to detect and learn the best grasping poses of an object that might be surrounded by obstacles in cluttered settings. The expected output of the proposed framework is one or more grasping regions around the object or, alternatively, a set of grasping poses.

This project is simulation based and uses the CoppeliaSim robotic simulation program with the Franka Emika Arm to demonstrate the applications of the proposed framework.
