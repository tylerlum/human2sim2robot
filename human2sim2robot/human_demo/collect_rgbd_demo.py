import signal
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import rospy
import tyro
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo, Image


@dataclass
class Args:
    output_dir: Path
    rgb_topic: str = "/zed/zed_node/rgb/image_rect_color"
    depth_topic: str = "/zed/zed_node/depth/depth_registered"
    camera_info_topic: str = "/zed/zed_node/rgb/camera_info"


class ImageSaver:
    def __init__(
        self, output_dir: Path, rgb_topic: str, depth_topic: str, camera_info_topic: str
    ):
        rospy.init_node("image_saver", anonymous=True)

        self.output_dir = output_dir
        self.rgb_topic = rgb_topic
        self.depth_topic = depth_topic
        self.camera_info_topic = camera_info_topic

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rgb_save_dir = self.output_dir / "rgb"
        self.depth_save_dir = self.output_dir / "depth"
        self.rgb_save_dir.mkdir(parents=True, exist_ok=True)
        self.depth_save_dir.mkdir(parents=True, exist_ok=True)

        self.bridge = CvBridge()
        self.rgb_images = []
        self.depth_images = []
        self.latest_rgb_image = None
        self.latest_depth_image = None
        self.latest_camera_info = None

        self.is_in_progress_saving_to_file = False

        # Signal handling to save on shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

        RATE_HZ = 30
        self.save_rate = rospy.Rate(RATE_HZ)

        rospy.Subscriber(rgb_topic, Image, self.color_callback)
        rospy.Subscriber(depth_topic, Image, self.depth_callback)
        rospy.Subscriber(camera_info_topic, CameraInfo, self.camera_info_callback)

        rospy.sleep(2)

        print("ImageSaver initialized.")

    def color_callback(self, msg: Image):
        if self.is_in_progress_saving_to_file:
            return
        try:
            self.latest_rgb_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            rospy.logerr(f"Failed to process RGB image: {e}")

    def depth_callback(self, msg: Image):
        if self.is_in_progress_saving_to_file:
            return
        try:
            self.latest_depth_image = self.bridge.imgmsg_to_cv2(msg, "64FC1")
        except Exception as e:
            rospy.logerr(f"Failed to process depth image: {e}")

    def camera_info_callback(self, msg: CameraInfo):
        self.latest_camera_info = np.array(msg.K).reshape(3, 3)

    def save_to_disk(self):
        rospy.loginfo("Saving images...")
        for idx, (rgb_image, depth_image) in enumerate(
            zip(self.rgb_images, self.depth_images)
        ):
            # process depth image
            depth_image[np.isnan(depth_image)] = 0
            depth_image[np.isinf(depth_image)] = 0
            depth_image = np.array(depth_image, dtype=np.uint16)

            rgb_file_path = self.rgb_save_dir / f"{idx:05d}.png"
            depth_file_path = self.depth_save_dir / f"{idx:05d}.png"
            cv2.imwrite(rgb_file_path, rgb_image)
            cv2.imwrite(depth_file_path, depth_image)
            rospy.loginfo(
                f"Saved RGB image to {rgb_file_path} and depth image to {depth_file_path}"
            )
        rospy.loginfo(
            f"Saved {len(self.rgb_images)} RGB images and {len(self.depth_images)} depth images."
        )

        rospy.loginfo("Saving camera info...")
        assert self.latest_camera_info is not None
        camera_info_file_path = self.output_dir / "cam_K.txt"
        np.savetxt(camera_info_file_path, self.latest_camera_info)
        rospy.loginfo(f"Saved camera info to {camera_info_file_path}")

    def run(self):
        while not rospy.is_shutdown():
            if (
                self.latest_rgb_image is None
                or self.latest_depth_image is None
                or self.latest_camera_info is None
            ):
                rospy.logwarn("Waiting for RGB, depth, and camera info...")
                rospy.logwarn(
                    f"RGB: {self.rgb_topic}: have info? {self.latest_rgb_image is not None}"
                )
                rospy.logwarn(
                    f"Depth: {self.depth_topic}: have info? {self.latest_depth_image is not None}"
                )
                rospy.logwarn(
                    f"Camera Info: {self.camera_info_topic}: have info? {self.latest_camera_info is not None}"
                )
                self.save_rate.sleep()
                continue

            rospy.loginfo(
                f"Saving data, have {len(self.rgb_images)} RGB images and {len(self.depth_images)} depth images and {len(self.latest_camera_info)} camera info"
            )
            rgb_image = self.latest_rgb_image.copy()
            depth_image = self.latest_depth_image.copy()
            self.rgb_images.append(rgb_image)
            self.depth_images.append(depth_image)
            self.save_rate.sleep()

    def signal_handler(self, signum, frame):
        if self.is_in_progress_saving_to_file:
            return

        self.is_in_progress_saving_to_file = True
        rospy.loginfo(
            f"Signal {signum} received. Saving all collected data and shutting down."
        )
        self.save_to_disk()
        rospy.signal_shutdown("Image and Pose Saver node stopped.")
        exit()


def main() -> None:
    args = tyro.cli(Args)
    print("=" * 80)
    print(args)
    print("=" * 80 + "\n")

    saver = ImageSaver(
        output_dir=args.output_dir,
        rgb_topic=args.rgb_topic,
        depth_topic=args.depth_topic,
        camera_info_topic=args.camera_info_topic,
    )
    saver.run()


if __name__ == "__main__":
    main()
