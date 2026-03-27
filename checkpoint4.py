from checkpoint3 import CubePoseDetector

import cv2, numpy, time
from xarm.wrapper import XArmAPI

from utils.vis_utils import draw_pose_axes
from utils.zed_camera import ZedCamera
from checkpoint1 import grasp_cube, place_cube, GRIPPER_LENGTH

# Height of one cube in meters (50 mm)
STACK_HEIGHT = 0.022

robot_ip = '192.168.1.182'

def main():

    # Initialize ZED Camera
    zed = ZedCamera()
    camera_intrinsic = zed.camera_intrinsic

    # Initialize Cube Pose Detector
    cube_pose_detector = CubePoseDetector(camera_intrinsic)

    # Initialize Lite6 Robot
    arm = XArmAPI(robot_ip)
    arm.connect()
    arm.motion_enable(enable=True)
    arm.set_tcp_offset([0, 0, GRIPPER_LENGTH, 0, 0, 0])
    arm.set_mode(0)
    arm.set_state(0)
    arm.move_gohome(wait=True)
    time.sleep(0.5)

    try:

        cv_image = zed.image


        result_blue  = cube_pose_detector.get_transforms(cv_image, 'blue cube')
        result_green = cube_pose_detector.get_transforms(cv_image, 'green cube')
        result_red   = cube_pose_detector.get_transforms(cv_image, 'red cube')

        if result_blue is None or result_green is None or result_red is None:
            print('One or more cubes not detected. Aborting.')
            return

        t_robot_blue,  _ = result_blue
        t_robot_green, _ = result_green
        t_robot_red,   _ = result_red

        def make_target_pose(base_pose, level):
            """Return a 4x4 target pose at base XY, stacked level * STACK_HEIGHT above base Z."""
            target = numpy.eye(4)
            target[:3, :3] = base_pose[:3, :3]
            target[0, 3] = base_pose[0, 3]
            target[1, 3] = base_pose[1, 3]
            target[2, 3] = base_pose[2, 3] + level * STACK_HEIGHT
            return target

        green_target = make_target_pose(t_robot_blue, 1)
        red_target   = make_target_pose(t_robot_blue, 2)

  
        grasp_cube(arm, t_robot_green)
        place_cube(arm, green_target)


        cv_image = zed.image
        result_red = cube_pose_detector.get_transforms(cv_image, 'red cube')
        if result_red is None:
            print('Red cube not detected after placing green. Aborting.')
            return
        t_robot_red, _ = result_red

        # Place red on green (which is now on blue)
        grasp_cube(arm, t_robot_red)
        place_cube(arm, red_target)

    finally:
        # Close Lite6 Robot
        arm.move_gohome(wait=True)
        time.sleep(0.5)
        arm.disconnect()

        # Close ZED Camera
        zed.close()

if __name__ == "__main__":
    main()