import cv2, numpy, time
from pupil_apriltags import Detector
from xarm.wrapper import XArmAPI

from utils.vis_utils import draw_pose_axes
from utils.zed_camera import ZedCamera
from checkpoint0 import get_transform_camera_robot
from checkpoint1 import grasp_cube, place_cube, GRIPPER_LENGTH, CUBE_TAG_FAMILY, CUBE_TAG_ID, CUBE_TAG_SIZE

cube_prompt = 'blue cube'
robot_ip = ''

class CubePoseDetector:
    """
    A detector to robustly identify and locate a specific cube in the scene.

    This class leverages text prompts to semantically segment a specific cube (e.g., 
    'blue cube') and determine the cube's pose by the AprilTags.
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
        self.detector = Detector(families=CUBE_TAG_FAMILY)

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
        t_cam_robot = get_transform_camera_robot(observation, self.camera_intrinsic)
        if t_cam_robot is None:
            return None

        target_color = None
        for color in self.COLOR_RANGES:
            if color in cube_prompt.lower():
                target_color = color
                break
        if target_color is None:
            return None

        if len(observation.shape) > 2:
            bgr = cv2.cvtColor(observation, cv2.COLOR_BGRA2BGR)
        else:
            bgr = observation

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask = numpy.zeros(hsv.shape[:2], dtype=numpy.uint8)
        for (lo, hi) in self.COLOR_RANGES[target_color]:
            mask |= cv2.inRange(hsv, numpy.array(lo), numpy.array(hi))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest)
        if M['m00'] == 0:
            return None
        color_cx = M['m10'] / M['m00']
        color_cy = M['m01'] / M['m00']

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        fx = self.camera_intrinsic[0, 0]
        fy = self.camera_intrinsic[1, 1]
        px = self.camera_intrinsic[0, 2]
        py = self.camera_intrinsic[1, 2]
        tags = self.detector.detect(gray, estimate_tag_pose=True,
                                    camera_params=[fx, fy, px, py],
                                    tag_size=CUBE_TAG_SIZE)

        best_tag = None
        best_dist = float('inf')
        for tag in tags:
            if tag.tag_id <= 3:
                continue
            dist = numpy.hypot(tag.center[0] - color_cx, tag.center[1] - color_cy)
            if dist < best_dist:
                best_dist = dist
                best_tag = tag

        if best_tag is None:
            return None

        t_cam_cube = numpy.eye(4)
        t_cam_cube[:3, :3] = best_tag.pose_R
        t_cam_cube[:3, 3] = best_tag.pose_t.flatten()

        t_robot_cube = numpy.linalg.inv(t_cam_robot) @ t_cam_cube

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

        t_cam_cube = None
        result = cube_pose_detector.get_transforms(cv_image, cube_prompt)
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
