import cv2, numpy, time
import open3d as o3d
from xarm.wrapper import XArmAPI

from utils.vis_utils import draw_pose_axes
from utils.zed_camera import ZedCamera
from checkpoint0 import get_transform_camera_robot
from checkpoint1 import grasp_cube, place_cube, GRIPPER_LENGTH
from checkpoint4 import STACK_HEIGHT
from checkpoint6 import CUBE_SIZE

robot_ip = '192.168.1.168'

# Minimum pixel area for a contour to be considered a cube
MIN_CONTOUR_AREA = 300

STACK_XY_TOLERANCE = 0.04


class ChallengeDetector:

    COLOR_RANGES = {
        'red':   [((0,   80, 50), (10,  255, 255)), ((160, 80, 50), (180, 255, 255))],
        'green': [((40,  60, 50), (80,  255, 255))],
        'blue':  [((95,  80, 50), (130, 255, 255))],
    }

    def __init__(self, camera_intrinsic):
        self.camera_intrinsic = camera_intrinsic

    def _pose_from_mask(self, contour_mask, point_cloud, t_cam_robot):
        """Extract a (t_robot_cube, t_cam_cube) pose from a filled contour mask and point cloud.
        Returns None if insufficient valid depth data."""
        TOP_FACE_THRESHOLD = 0.006

        pix_ys, pix_xs = numpy.where(contour_mask > 0)
        depths = point_cloud[pix_ys, pix_xs, 2].astype(numpy.float64) / 1000.0
        valid = numpy.isfinite(depths) & (depths > 0.3) & (depths < 1.5)
        pix_xs_v = pix_xs[valid]
        pix_ys_v = pix_ys[valid]
        depths_v  = depths[valid]

        if len(depths_v) < 10:
            return None

        fx = self.camera_intrinsic[0, 0]
        fy = self.camera_intrinsic[1, 1]
        cx = self.camera_intrinsic[0, 2]
        cy = self.camera_intrinsic[1, 2]

        X = (pix_xs_v - cx) * depths_v / fx
        Y = (pix_ys_v - cy) * depths_v / fy
        Z = depths_v
        pts_cam = numpy.stack([X, Y, Z], axis=1)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_cam)
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

        if len(pcd.points) < 10:
            return None

        pts_cam_clean = numpy.asarray(pcd.points)

        # Transform to robot frame to find the true cube center via top face
        t_cam2robot = numpy.linalg.inv(t_cam_robot)
        ones = numpy.ones((pts_cam_clean.shape[0], 1))
        pts_robot = (t_cam2robot @ numpy.hstack([pts_cam_clean, ones]).T).T[:, :3]

        max_z = numpy.max(pts_robot[:, 2])
        top_mask = pts_robot[:, 2] > (max_z - TOP_FACE_THRESHOLD)
        top_pts = pts_robot[top_mask]
        if len(top_pts) < 5:
            top_pts = pts_robot

        top_xy = top_pts[:, :2].astype(numpy.float32)
        rect = cv2.minAreaRect(top_xy)
        rect_center = rect[0]

        center_robot = numpy.array([
            rect_center[0],
            rect_center[1],
            max_z - CUBE_SIZE / 2.0,
        ])
        yaw_rad = numpy.radians(rect[2])
        cos_y = numpy.cos(yaw_rad)
        sin_y = numpy.sin(yaw_rad)
        R_robot = numpy.array([
            [cos_y, -sin_y, 0],
            [sin_y,  cos_y, 0],
            [0,      0,     1],
        ])

        t_robot_cube = numpy.eye(4)
        t_robot_cube[:3, :3] = R_robot
        t_robot_cube[:3, 3] = center_robot

        t_cam_cube = t_cam_robot @ t_robot_cube
        return t_robot_cube, t_cam_cube

    def detect_all_cubes(self, image, point_cloud, t_cam_robot,
                         exclude_xy=None):
        """
        Detect all cubes visible in the scene.

        Parameters
        ----------
        image : numpy.ndarray
            BGRA or BGR image from the camera.
        point_cloud : numpy.ndarray
            Registered point cloud (H x W x 4), values in mm.
        t_cam_robot : numpy.ndarray
            4x4 camera-to-robot transform.
        exclude_xy : tuple or None
            (x, y) in robot frame (meters). Cubes within STACK_XY_TOLERANCE
            of this position are skipped (already on the stack).

        Returns
        -------
        list of (t_robot_cube, t_cam_cube) tuples, one per detected cube.
        """
        if len(image.shape) > 2 and image.shape[2] == 4:
            bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        else:
            bgr = image.copy()

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        # Build combined mask for all cube colors
        mask = numpy.zeros(hsv.shape[:2], dtype=numpy.uint8)
        for ranges in self.COLOR_RANGES.values():
            for lo, hi in ranges:
                mask |= cv2.inRange(hsv, numpy.array(lo), numpy.array(hi))

        # Suppress top 30% (background, walls)
        h_img = mask.shape[0]
        mask[:int(h_img * 0.3), :] = 0

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = [c for c in contours if cv2.contourArea(c) > MIN_CONTOUR_AREA]

        results = []
        for contour in valid_contours:
            contour_mask = numpy.zeros(mask.shape, dtype=numpy.uint8)
            cv2.drawContours(contour_mask, [contour], -1, 255, cv2.FILLED)

            pose_result = self._pose_from_mask(contour_mask, point_cloud, t_cam_robot)
            if pose_result is None:
                continue

            t_robot_cube, t_cam_cube = pose_result

            # Skip cubes that are already part of the stack
            if exclude_xy is not None:
                dx = t_robot_cube[0, 3] - exclude_xy[0]
                dy = t_robot_cube[1, 3] - exclude_xy[1]
                if numpy.hypot(dx, dy) < STACK_XY_TOLERANCE:
                    continue

            results.append((t_robot_cube, t_cam_cube))

        return results


def main():

    # Initialize ZED Camera
    zed = ZedCamera()
    camera_intrinsic = zed.camera_intrinsic

    detector = ChallengeDetector(camera_intrinsic)

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
        # ── Initial scan ──────────────────────────────────────────────────────
        cv_image    = zed.image
        point_cloud = zed.point_cloud

        t_cam_robot = get_transform_camera_robot(cv_image, camera_intrinsic)
        if t_cam_robot is None:
            print('Could not determine camera-to-robot transform. Aborting.')
            return

        cubes = detector.detect_all_cubes(cv_image, point_cloud, t_cam_robot)
        n_cubes = len(cubes)
        print(f'Detected {n_cubes} cube(s).')

        if n_cubes == 0:
            print('No cubes detected. Aborting.')
            return
        if n_cubes == 1:
            print('Only one cube detected — nothing to stack.')
            return

        # ── Visualize all detected cubes ──────────────────────────────────────
        vis_image = cv_image.copy()
        for _, t_cam_cube in cubes:
            draw_pose_axes(vis_image, camera_intrinsic, t_cam_cube)
        cv2.namedWindow('Detected Cubes — press k to start stacking', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Detected Cubes — press k to start stacking', 1280, 720)
        cv2.imshow('Detected Cubes — press k to start stacking', vis_image)
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()

        if key != ord('k'):
            print('Aborted by user.')
            return

        # ── Choose the base cube ──────────────────────────────────────────────
        # Use the cube with the largest Y in robot frame as the base
        # (furthest from the robot, typically most stable to build on)
        base_idx = int(numpy.argmax([c[0][1, 3] for c in cubes]))
        base_pose = cubes[base_idx][0]
        stack_xy  = (base_pose[0, 3], base_pose[1, 3])
        print(f'Base cube at robot XY = ({stack_xy[0]*1000:.1f}, {stack_xy[1]*1000:.1f}) mm')

        # ── Stack remaining cubes one by one ──────────────────────────────────
        for stack_level in range(1, n_cubes):
            print(f'\n── Stacking level {stack_level} ──')

            # Re-detect cubes each iteration for fresh poses, excluding the stack
            cv_image    = zed.image
            point_cloud = zed.point_cloud
            t_cam_robot = get_transform_camera_robot(cv_image, camera_intrinsic)
            if t_cam_robot is None:
                print('Lost camera-to-robot transform. Aborting.')
                break

            remaining = detector.detect_all_cubes(
                cv_image, point_cloud, t_cam_robot, exclude_xy=stack_xy
            )

            if not remaining:
                print('No more cubes found on the table. Done.')
                break

            # Pick the cube closest to the robot (smallest Y) — easier to reach
            pick_idx  = int(numpy.argmin([c[0][1, 3] for c in remaining]))
            t_robot_pick = remaining[pick_idx][0]
            print(f'Picking cube at robot XY = '
                  f'({t_robot_pick[0,3]*1000:.1f}, {t_robot_pick[1,3]*1000:.1f}) mm')

            # Target: base XY, rotation from picked cube, Z = base + level * STACK_HEIGHT
            target_pose = t_robot_pick.copy()
            target_pose[0, 3] = base_pose[0, 3]
            target_pose[1, 3] = base_pose[1, 3]
            target_pose[2, 3] = base_pose[2, 3] + stack_level * STACK_HEIGHT

            grasp_cube(arm, t_robot_pick)
            place_cube(arm, target_pose)

        print('\nChallenge 1 complete.')

    finally:
        arm.move_gohome(wait=True)
        time.sleep(0.5)
        arm.disconnect()
        zed.close()


if __name__ == "__main__":
    main()
