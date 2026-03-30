from checkpoint8 import CubePoseDetector

import cv2, numpy, time
from xarm.wrapper import XArmAPI

from utils.vis_utils import draw_pose_axes
from utils.zed_camera import ZedCamera
from checkpoint1 import grasp_cube, place_cube, GRIPPER_LENGTH
from checkpoint4 import STACK_HEIGHT
from checkpoint6 import CUBE_SIZE

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
        # Get Observation
        cv_image = zed.image
        point_cloud = zed.point_cloud

        # Detect red and green cube poses
        result_red   = cube_pose_detector.get_transforms([cv_image, point_cloud], 'red cube')
        result_green = cube_pose_detector.get_transforms([cv_image, point_cloud], 'green cube')

        if result_red is None or result_green is None:
            print('One or more cubes not detected. Aborting.')
            return

        t_robot_red,   t_cam_red   = result_red
        t_robot_green, t_cam_green = result_green

        # Compute target pose: red goes on top of green
        green_top = numpy.eye(4)
        green_top[:3, :3] = t_robot_green[:3, :3]
        green_top[0, 3] = t_robot_green[0, 3]
        green_top[1, 3] = t_robot_green[1, 3]
        green_top[2, 3] = t_robot_green[2, 3] + STACK_HEIGHT

        # Visualization: draw both cube poses on the image
        draw_pose_axes(cv_image, camera_intrinsic, t_cam_red)
        draw_pose_axes(cv_image, camera_intrinsic, t_cam_green)
        cv2.namedWindow('Verifying Cube Poses', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Verifying Cube Poses', 1280, 720)
        cv2.imshow('Verifying Cube Poses', cv_image)
        key = cv2.waitKey(0)

        if key == ord('k'):
            cv2.destroyAllWindows()

            # Grasp red cube and stack it on the green cube
            grasp_cube(arm, t_robot_red)
            place_cube(arm, green_top)

    finally:
        # Close Lite6 Robot
        arm.move_gohome(wait=True)
        time.sleep(0.5)
        arm.disconnect()

        # Close ZED Camera
        zed.close()

if __name__ == "__main__":
    main()
