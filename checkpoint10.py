from checkpoint8 import CubePoseDetector

import cv2, time
from xarm.wrapper import XArmAPI

from utils.vis_utils import draw_pose_axes
from utils.zed_camera import ZedCamera
from checkpoint1 import grasp_cube, place_cube, GRIPPER_LENGTH
from checkpoint4 import STACK_HEIGHT

stacking_order = ['red cube', 'green cube', 'blue cube']   # From top to bottom
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

        # Detect all three cubes from the initial observation
        robot_poses = {}
        cam_poses = {}
        for prompt in stacking_order:
            result = cube_pose_detector.get_transforms([cv_image, point_cloud], prompt)
            if result is None:
                print(f'Could not detect {prompt}. Aborting.')
                return
            robot_poses[prompt], cam_poses[prompt] = result

        # Visualization: draw all detected cube poses
        for prompt in stacking_order:
            draw_pose_axes(cv_image, camera_intrinsic, cam_poses[prompt])
        cv2.namedWindow('Verifying Cube Poses', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Verifying Cube Poses', 1280, 720)
        cv2.imshow('Verifying Cube Poses', cv_image)
        key = cv2.waitKey(0)

        if key != ord('k'):
            return
        cv2.destroyAllWindows()

        # Build the stacking sequence: bottom cube is the base, stack the rest above it
        base_pose = robot_poses[stacking_order[-1]]
        
        cubes_to_stack = list(reversed(stacking_order[:-1]))

        for level, prompt in enumerate(cubes_to_stack, start=1):
            target_pose = base_pose.copy()
            target_pose[2, 3] += level * STACK_HEIGHT

            grasp_cube(arm, robot_poses[prompt])
            place_cube(arm, target_pose)

            # Re-acquire fresh observation before next detection
            cv_image = zed.image
            point_cloud = zed.point_cloud

    finally:
        # Close Lite6 Robot
        arm.move_gohome(wait=True)
        time.sleep(0.5)
        arm.disconnect()

        # Close ZED Camera
        zed.close()

if __name__ == "__main__":
    main()
