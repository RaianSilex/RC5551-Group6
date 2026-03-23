import cv2, numpy, time
from pupil_apriltags import Detector
from xarm.wrapper import XArmAPI
from scipy.spatial.transform import Rotation

from utils.vis_utils import draw_pose_axes
from utils.zed_camera import ZedCamera
from checkpoint0 import get_transform_camera_robot

GRIPPER_LENGTH = 0.067 * 1000
CUBE_TAG_FAMILY = 'tag36h11'
CUBE_TAG_ID = 4
CUBE_TAG_SIZE = 0.0207

robot_ip = ''

def grasp_cube(arm, cube_pose):
    """
    Execute a pick sequence to grasp a cube at a specified pose.

    Parameters
    ----------
    arm : xarm.wrapper.XArmAPI
        The initialized XArm API object controlling the Lite6 robot.
    cube_pose : numpy.ndarray
        A 4x4 transformation matrix representing the cube's pose in the robot base frame.
        All translational units in this matrix are in meters.
    """
    # Extract position in mm (robot API uses mm)
    x = cube_pose[0, 3] * 1000
    y = cube_pose[1, 3] * 1000
    z = cube_pose[2, 3] * 1000

    # Extract yaw from cube rotation: project cube x-axis onto robot XY plane
    x_axis = cube_pose[:3, 0]
    yaw = numpy.degrees(numpy.arctan2(x_axis[1], x_axis[0]))

    # Top-down grasp orientation: roll=180 flips end-effector to point down
    grasp_roll, grasp_pitch, grasp_yaw = 180, 0, yaw

    PRE_GRASP_OFFSET = 120  # mm above tag face before descending
    GRASP_DEPTH = 10        # mm below tag face to grip cube sides

    # Open gripper before approaching
    arm.open_lite6_gripper()
    time.sleep(0.5)
    arm.stop_lite6_gripper()

    # Move to pre-grasp position above cube
    arm.set_position(x, y, z + PRE_GRASP_OFFSET, grasp_roll, grasp_pitch, grasp_yaw, wait=True)

    # Descend to grasp position
    arm.set_position(x, y, z - GRASP_DEPTH, grasp_roll, grasp_pitch, grasp_yaw, wait=True)

    # Close gripper to grasp cube
    arm.close_lite6_gripper()
    time.sleep(0.8)
    arm.stop_lite6_gripper()

    # Lift cube back to safe height
    arm.set_position(x, y, z + PRE_GRASP_OFFSET, grasp_roll, grasp_pitch, grasp_yaw, wait=True)

def place_cube(arm, cube_pose):
    """
    Execute a place sequence to release a cube at a specified pose.

    Parameters
    ----------
    arm : xarm.wrapper.XArmAPI
        The initialized XArm API object controlling the Lite6 robot.
    cube_pose : numpy.ndarray
        A 4x4 transformation matrix representing the target placement pose in the robot base frame.
        All translational units in this matrix are in meters.
    """
    # Extract position in mm
    x = cube_pose[0, 3] * 1000
    y = cube_pose[1, 3] * 1000
    z = cube_pose[2, 3] * 1000

    # Extract yaw from rotation matrix
    x_axis = cube_pose[:3, 0]
    yaw = numpy.degrees(numpy.arctan2(x_axis[1], x_axis[0]))

    place_roll, place_pitch, place_yaw = 180, 0, yaw

    PRE_PLACE_OFFSET = 120  # mm above target before descending
    PLACE_DEPTH = 10        # mm below tag face to set cube down

    # Move to pre-place position above target
    arm.set_position(x, y, z + PRE_PLACE_OFFSET, place_roll, place_pitch, place_yaw, wait=True)

    # Descend to place position
    arm.set_position(x, y, z - PLACE_DEPTH, place_roll, place_pitch, place_yaw, wait=True)

    # Open gripper to release cube
    arm.open_lite6_gripper()
    time.sleep(0.5)
    arm.stop_lite6_gripper()

    # Lift back to safe height
    arm.set_position(x, y, z + PRE_PLACE_OFFSET, place_roll, place_pitch, place_yaw, wait=True)

def get_transform_cube(observation, camera_intrinsic, camera_pose):
    """
    Calculate the transformation matrix for the cube relative to the robot base frame, 
    as well as relative to the camera frame.

    This function uses visual fiducial detection to find the cube's pose in the camera's view, 
    then transforms that pose into the robot's global coordinate system. 

    Parameters
    ----------
    observation : numpy.ndarray
        The input image from the camera. Can be a color (BGRA/BGR) or grayscale image.
    camera_intrinsic : numpy.ndarray
        The 3x3 intrinsic camera matrix.
    camera_pose : numpy.ndarray
        A 4x4 transformation matrix representing the camera's pose in the robot base frame (t_cam_robot).
        All translations are in meters.

    Returns
    -------
    tuple or None
        If successful, returns a tuple (t_robot_cube, t_cam_cube) where both 
        are 4x4 transformation matrices with translations in meters. 
        If no cube tag is detected, returns None.
    """
    # Initialize AprilTag detector
    detector = Detector(families=CUBE_TAG_FAMILY)

    # Convert image to grayscale (required by pupil_apriltags)
    if len(observation.shape) > 2:
        gray = cv2.cvtColor(observation, cv2.COLOR_BGRA2GRAY)
    else:
        gray = observation

    # Extract camera intrinsic parameters for pose estimation
    fx = camera_intrinsic[0, 0]
    fy = camera_intrinsic[1, 1]
    cx = camera_intrinsic[0, 2]
    cy = camera_intrinsic[1, 2]

    # Detect AprilTags with pose estimation enabled
    tags = detector.detect(gray, estimate_tag_pose=True,
                           camera_params=[fx, fy, cx, cy],
                           tag_size=CUBE_TAG_SIZE)

    for tag in tags:
        if tag.tag_id == CUBE_TAG_ID:
            # Build 4x4 pose of cube tag in camera frame (t_cam_cube)
            t_cam_cube = numpy.eye(4)
            t_cam_cube[:3, :3] = tag.pose_R
            t_cam_cube[:3, 3] = tag.pose_t.flatten()

            # Transform to robot base frame: t_robot_cube = inv(t_cam_robot) @ t_cam_cube
            # camera_pose (t_cam_robot) maps robot frame -> camera frame, so inv maps back
            t_robot_cube = numpy.linalg.inv(camera_pose) @ t_cam_cube

            return t_robot_cube, t_cam_cube

    return None

def main():

    # Initialize ZED Camera
    zed = ZedCamera()
    camera_intrinsic = zed.camera_intrinsic

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

        # Get Transformation
        t_cam_robot = get_transform_camera_robot(cv_image, camera_intrinsic)
        if t_cam_robot is None:
            return
        
        t_cam_cube = None
        result = get_transform_cube(cv_image, camera_intrinsic, t_cam_robot)
        if result is None:
            print('Cube tag not detected.')
            return
        t_robot_cube, t_cam_cube = result
        
        # Visualization
        draw_pose_axes(cv_image, camera_intrinsic, t_cam_cube)
        cv2.namedWindow('Verifying Cube Pose', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Verifying Cube Pose', 1280, 720)
        cv2.imshow('Verifying Cube Pose', cv_image)
        key = cv2.waitKey(0)

        if key == ord('k'):
            cv2.destroyAllWindows()

            grasp_cube(arm, t_robot_cube)
            place_cube(arm, t_robot_cube)
    
    finally:
        # Close Lite6 Robot
        arm.move_gohome(wait=True)
        time.sleep(0.5)
        arm.disconnect()

        # Close ZED Camera
        zed.close()

if __name__ == "__main__":
    main()
