import cv2, numpy, time, torch
import open3d as o3d
from scipy.spatial.transform import Rotation
from xarm.wrapper import XArmAPI

from utils.vis_utils import draw_pose_axes
from utils.zed_camera import ZedCamera
from checkpoint0 import get_transform_camera_robot
from checkpoint1 import grasp_cube, place_cube, GRIPPER_LENGTH

CUBE_SIZE = 0.025

robot_ip = ''

def get_transform_cube(observation, camera_intrinsic, camera_pose):
    """
    Calculate the transformation matrix for the cube relative to the robot base frame, 
    as well as relative to the camera frame.

    This function leverages text prompts to semantically segment a specific 
    cube (e.g., 'red cube') and determines the cube's pose using its 3D point cloud.

    Parameters
    ----------
    observation : list or tuple
        A collection containing [image, point_cloud], where image is the 
        RGB/BGRA array and point_cloud is the registered 3D point cloud.
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
        If no matching object is segmented, returns None.
    """
    image, point_cloud = observation

    # TODO
    if len(image.shape) > 2 and image.shape[2] == 4:
        bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    else:
        bgr = image

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, numpy.array([0, 60, 50]), numpy.array([180, 255, 255]))

    pts = point_cloud[mask > 0, :3].astype(numpy.float64)
    valid = numpy.isfinite(pts).all(axis=1)
    pts = pts[valid]

    if len(pts) < 10:
        return None

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    if len(pcd.points) < 10:
        return None

    obb = pcd.get_oriented_bounding_box()
    center = numpy.array(obb.center)
    R = numpy.array(obb.R)

    t_cam_cube = numpy.eye(4)
    t_cam_cube[:3, :3] = R
    t_cam_cube[:3, 3] = center

    t_robot_cube = numpy.linalg.inv(camera_pose) @ t_cam_cube

    return t_robot_cube, t_cam_cube

    

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
        point_cloud = zed.point_cloud

        t_cam_cube = None
        # TODO
        t_cam_robot = get_transform_camera_robot(cv_image, camera_intrinsic)
        if t_cam_robot is None:
            return
        result = get_transform_cube([cv_image, point_cloud], camera_intrinsic, t_cam_robot)
        if result is None:
            print('Cube not detected.')
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

            # TODO
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
