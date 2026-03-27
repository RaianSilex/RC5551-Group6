import cv2, numpy, time
import open3d as o3d
from scipy.spatial.transform import Rotation
from xarm.wrapper import XArmAPI

from utils.vis_utils import draw_pose_axes
from utils.zed_camera import ZedCamera
from checkpoint0 import get_transform_camera_robot
from checkpoint1 import grasp_cube, place_cube, GRIPPER_LENGTH
from checkpoint6 import CUBE_SIZE, get_transform_cube

cube_prompt = 'red cube'
robot_ip = '192.168.1.183'

class CubePoseDetector:
    """
    A detector to robustly identify and locate a specific cube in the scene.

    This class leverages text prompts to semantically segment a specific cube (e.g., 
    'blue cube') and determine the cube's pose by its 3D point cloud.
    """

    COLOR_RANGES = {
        'red':   [((0,   80, 50), (10,  255, 255)), ((160, 80, 50), (180, 255, 255))],
        'green': [((40,  60, 50), (80,  255, 255))],
        'blue':  [((100, 80, 50), (130, 255, 255))],
    }

    def __init__(self, camera_intrinsic):
        """
        Initialize the CubePoseDetector with camera parameters.

        Parameters
        ----------
        camera_intrinsic : numpy.ndarray
            The 3x3 intrinsic camera matrix.
        """
        self.camera_intrinsic = camera_intrinsic
        #self.detector = Detector(families=CUBE_TAG_FAMILY)
        


    def get_transforms(self, observation, cube_prompt):
        """
        Calculate the transformation matrix for a specific prompted cube relative to the robot base frame,
        as well as relative to the camera frame.

        Parameters
        ----------
        observation : numpy.ndarray
            The input image from the camera. Can be a color (BGRA/BGR) or grayscale image.
        cube_prompt : str
            The text prompt used to segment the target object (e.g., 'blue cube').

        Returns
        -------
        tuple or None
            If successful, returns a tuple (t_robot_cube, t_cam_cube) where both
            are 4x4 transformation matrices with translations in meters.
            If no matching object or tag is found, returns None.
        """
        image, point_cloud = observation

        # TODO
        if len(image.shape) > 2 and image.shape[2] == 4:
            bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        else:
            bgr = image

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        red1  = cv2.inRange(hsv, numpy.array([0,   80,  50]), numpy.array([10,  255, 255]))
        red2  = cv2.inRange(hsv, numpy.array([160, 80,  50]), numpy.array([180, 255, 255]))
        green = cv2.inRange(hsv, numpy.array([40,  80,  50]), numpy.array([85,  255, 255]))
        blue  = cv2.inRange(hsv, numpy.array([95,  80,  50]), numpy.array([130, 255, 255]))
        mask  = red1 | red2 | green | blue

        h_img = mask.shape[0]
        mask[:int(h_img * 0.3), :] = 0

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print('No colored contours found.')
            return None
        valid_contours = [c for c in contours if cv2.contourArea(c) > 500]
        if not valid_contours:
            print('No contours large enough.')
            return None
        largest = max(valid_contours, key=cv2.contourArea)
        print(f'Detected contour: area={cv2.contourArea(largest):.0f}, bounds={cv2.boundingRect(largest)}')

        cv2.drawContours(image, [largest], -1, (0, 255, 255), 3)

        contour_mask = numpy.zeros(mask.shape, dtype=numpy.uint8)
        cv2.drawContours(contour_mask, [largest], -1, 255, cv2.FILLED)

        pix_ys, pix_xs = numpy.where(contour_mask > 0)
        depths = point_cloud[pix_ys, pix_xs, 2].astype(numpy.float64) / 1000.0
        valid = numpy.isfinite(depths) & (depths > 0.3) & (depths < 1.5)
        pix_xs, pix_ys, depths = pix_xs[valid], pix_ys[valid], depths[valid]

        if len(depths) < 10:
            print('Not enough valid depth points.')
            return None

        fx = self.camera_intrinsic[0, 0]
        fy = self.camera_intrinsic[1, 1]
        cx = self.camera_intrinsic[0, 2]
        cy = self.camera_intrinsic[1, 2]

        X = (pix_xs - cx) * depths / fx
        Y = (pix_ys - cy) * depths / fy
        Z = depths
        pts = numpy.stack([X, Y, Z], axis=1)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

        if len(pcd.points) < 10:
            print('Not enough points after outlier removal.')
            return None

        obb = pcd.get_oriented_bounding_box()
        center = numpy.array(obb.center) 
        #center[0] = center[0] - 0.0125
        center[1] = center[1] - CUBE_SIZE/2
        center[2] = center[2] + CUBE_SIZE/2
        R = numpy.array(obb.R)
        print(f'OBB center in camera frame (m): {numpy.round(center, 3)}')

        t_cam_cube = numpy.eye(4)
        t_cam_cube[:3, :3] = R
        t_cam_cube[:3, 3] = center 

        t_robot_cube = numpy.linalg.inv(camera_pose) @ t_cam_cube

        return t_robot_cube, t_cam_cube

        
        

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

        t_cam_cube = None
        result = cube_pose_detector.get_transforms([cv_image, point_cloud], cube_prompt)
        if result is None:
            print('Target cube not detected.')
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
