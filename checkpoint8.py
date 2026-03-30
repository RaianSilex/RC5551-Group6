import cv2, numpy, time
import open3d as o3d
from scipy.spatial.transform import Rotation
from xarm.wrapper import XArmAPI

from utils.vis_utils import draw_pose_axes
from utils.zed_camera import ZedCamera
from checkpoint0 import get_transform_camera_robot
from checkpoint1 import grasp_cube, place_cube, GRIPPER_LENGTH
from checkpoint6 import CUBE_SIZE

# Threshold (meters) to select top-face points from the highest Z in robot frame
TOP_FACE_THRESHOLD = 0.006

cube_prompt = 'green cube'
robot_ip = '192.168.1.182'

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

    def get_transforms(self, observation, cube_prompt):
        """
        Calculate the transformation matrix for a specific prompted cube relative to the robot base frame,
        as well as relative to the camera frame.

        Parameters
        ----------
        observation : list or tuple
            A collection containing [image, point_cloud], where image is the
            RGB/BGRA array and point_cloud is the registered 3D point cloud.
        cube_prompt : str
            The text prompt used to segment the target object (e.g., 'blue cube').

        Returns
        -------
        tuple or None
            If successful, returns a tuple (t_robot_cube, t_cam_cube) where both
            are 4x4 transformation matrices with translations in meters.
            If no matching object is segmented, returns None.
        """
        image, point_cloud = observation

        # Get camera-to-robot transform from checkpoint0


        camera_pose = get_transform_camera_robot(image, self.camera_intrinsic)
        if camera_pose is None:
            print('Could not determine camera-to-robot transform.')
            return None



        # Parse target color from prompt (Reused from checkpoint3)


        
        target_color = None
        for color in self.COLOR_RANGES:
            if color in cube_prompt.lower():
                target_color = color
                break
        if target_color is None:
            print(f'No recognized color in prompt: {cube_prompt}')
            return None

        # Convert to BGR for HSV processing
        if len(image.shape) > 2 and image.shape[2] == 4:
            bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        else:
            bgr = image.copy()

        # Build color mask using HSV thresholding
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask = numpy.zeros(hsv.shape[:2], dtype=numpy.uint8)
        for (lo, hi) in self.COLOR_RANGES[target_color]:
            mask |= cv2.inRange(hsv, numpy.array(lo), numpy.array(hi))



        # Reused from checkpoint6 onwards




        # Remove top 30% of image to suppress background noise
        h_img = mask.shape[0]
        mask[:int(h_img * 0.3), :] = 0

        # Find the largest valid contour
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print(f'No {target_color} contours found.')
            return None
        valid_contours = [c for c in contours if cv2.contourArea(c) > 200]
        if not valid_contours:
            print(f'No {target_color} contours large enough.')
            return None
        largest = max(valid_contours, key=cv2.contourArea)
        print(f'Detected {target_color} contour: area={cv2.contourArea(largest):.0f}')

        # Create filled mask of the largest contour
        contour_mask = numpy.zeros(mask.shape, dtype=numpy.uint8)
        cv2.drawContours(contour_mask, [largest], -1, 255, cv2.FILLED)

        # Extract depth values from point cloud (ZED returns mm, convert to meters)
        pix_ys, pix_xs = numpy.where(contour_mask > 0)
        depths = point_cloud[pix_ys, pix_xs, 2].astype(numpy.float64) / 1000.0
        valid = numpy.isfinite(depths) & (depths > 0.3) & (depths < 1.5)
        pix_xs, pix_ys, depths = pix_xs[valid], pix_ys[valid], depths[valid]

        if len(depths) < 10:
            print(f'Not enough valid depth points for {target_color} cube.')
            return None

        # Back-project to 3D using camera intrinsics
        fx = self.camera_intrinsic[0, 0]
        fy = self.camera_intrinsic[1, 1]
        cx = self.camera_intrinsic[0, 2]
        cy = self.camera_intrinsic[1, 2]

        X = (pix_xs - cx) * depths / fx
        Y = (pix_ys - cy) * depths / fy
        Z = depths
        pts_cam = numpy.stack([X, Y, Z], axis=1)

        # Remove outliers
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_cam)
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        if len(pcd.points) < 10:
            print('Not enough points after outlier removal.')
            return None

        pts_cam_clean = numpy.asarray(pcd.points)

        # Transform points to robot frame to find the true cube center
        t_cam2robot = numpy.linalg.inv(camera_pose)  # 4x4
        ones = numpy.ones((pts_cam_clean.shape[0], 1))
        pts_cam_h = numpy.hstack([pts_cam_clean, ones])  # Nx4
        pts_robot = (t_cam2robot @ pts_cam_h.T).T[:, :3]  # Nx3

        # In robot frame, Z is up. Find the top face of the cube.
        max_z = numpy.max(pts_robot[:, 2])
        top_mask = pts_robot[:, 2] > (max_z - TOP_FACE_THRESHOLD)
        top_pts = pts_robot[top_mask]

        if len(top_pts) < 5:
            # Fallback: use all points
            top_pts = pts_robot

        # The top face centroid gives the correct X, Y of the cube center.
        # The cube center Z is half a cube size below the top surface.
        center_robot = numpy.array([
            numpy.median(top_pts[:, 0]),
            numpy.median(top_pts[:, 1]),
            max_z - CUBE_SIZE / 2.0,
        ])
        print(f'Cube center in robot frame (m): {numpy.round(center_robot, 4)}')

        # Use OBB for rotation (yaw estimation)
        obb = pcd.get_oriented_bounding_box()
        R_obb = numpy.array(obb.R)

        # Build robot-frame transform
        t_robot_cube = numpy.eye(4)
        t_robot_cube[:3, :3] = R_obb
        t_robot_cube[:3, 3] = center_robot

        # Compute camera-frame transform for visualization
        t_cam_cube = camera_pose @ t_robot_cube

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
