"""Non-blocking ROS image visualization for detections and both maps."""

import queue
import threading

import cv2
import numpy as np
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, Float32MultiArray


class DebugVisualizer:
    def __init__(self, topic_prefix, scale=5):
        self.bridge = CvBridge()
        self.scale = max(1, int(scale))
        self.queue = queue.Queue(maxsize=1)
        prefix = topic_prefix.rstrip("/")
        self.detection_pub = rospy.Publisher(
            prefix + "/detections_image", Image, queue_size=1)
        self.global_pub = rospy.Publisher(
            prefix + "/global_map", Image, queue_size=1)
        self.local_pub = rospy.Publisher(
            prefix + "/local_map", Image, queue_size=1)
        self.comparison_pub = rospy.Publisher(
            prefix + "/map_comparison", Image, queue_size=1)
        self.probability_pub = rospy.Publisher(
            prefix + "/action_probabilities", Float32MultiArray, queue_size=1)
        self.timing_pub = rospy.Publisher(
            prefix + "/inference_time", Float32, queue_size=1)
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def submit(self, stamp, step_id, rgb, result, global_map, local_map):
        item = (
            stamp,
            int(step_id),
            np.array(rgb, copy=True),
            dict(result),
            np.array(global_map, copy=True),
            np.array(local_map, copy=True),
        )
        try:
            self.queue.put_nowait(item)
        except queue.Full:
            try:
                self.queue.get_nowait()
                self.queue.task_done()
            except queue.Empty:
                pass
            self.queue.put_nowait(item)

    @staticmethod
    def _global_color(value, result):
        value = np.asarray(value, dtype=np.float32)
        height, width = value.shape[1:]
        image = np.zeros((height, width, 3), dtype=np.float32)
        explored = value[1]
        image += explored[..., None] * 55.0
        image[..., 2] = np.maximum(image[..., 2], value[0] * 255.0)
        image[..., 1] = np.maximum(image[..., 1], value[2] * 180.0)
        robot = value[3] > 0
        image[robot] = (255, 255, 255)
        echinus = value[4]
        image[..., 0] = np.maximum(image[..., 0], echinus * 220.0)
        image[..., 2] = np.maximum(image[..., 2], echinus * 255.0)
        image[..., 0] = np.maximum(image[..., 0], value[5] * 255.0)
        # np.flipud returns a negative-stride view.  OpenCV drawing functions
        # require writable, C-contiguous image storage.
        image = np.ascontiguousarray(
            np.flipud(np.clip(image, 0, 255).astype(np.uint8)))
        pose = result["robot_pose"]
        spec = result["global_map_spec"]
        gx = int((pose["x"] - spec["origin_x"]) / spec["resolution_m"])
        gy = height - 1 - int(
            (pose["y"] - spec["origin_y"]) / spec["resolution_m"])
        if 0 <= gx < width and 0 <= gy < height:
            length = 7
            end = (
                int(round(gx + length * np.cos(pose["yaw"]))),
                int(round(gy - length * np.sin(pose["yaw"]))),
            )
            cv2.arrowedLine(
                image, (gx, gy), end, (255, 255, 255), 1, tipLength=0.4)
        return image

    @staticmethod
    def _local_color(value):
        value = np.asarray(value, dtype=np.float32)
        height, width = value.shape[1:]
        image = np.zeros((height, width, 3), dtype=np.float32)
        image[..., 2] = np.maximum(image[..., 2], value[0] * 255.0)
        image[..., 0] = np.maximum(image[..., 0], value[1] * 255.0)
        image[..., 0] = np.maximum(image[..., 0], value[2] * 220.0)
        image[..., 2] = np.maximum(image[..., 2], value[2] * 255.0)
        image[..., 1] = np.maximum(image[..., 1], value[3] * 180.0)
        image = np.clip(image, 0, 255).astype(np.uint8)
        center = (width // 2, height // 2)
        cv2.arrowedLine(image, center, (center[0], max(0, center[1] - 8)),
                        (255, 255, 255), 1, tipLength=0.4)
        return image

    @staticmethod
    def _detections_image(rgb, result):
        image = np.array(rgb, copy=True)
        colors = {"echinus": (255, 0, 255), "rock": (255, 100, 0)}
        for item in result["detections"]:
            x1, y1, x2, y2 = [int(round(value)) for value in item["bbox"]]
            name = item["class_name"]
            color = colors.get(name, (0, 255, 255))
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                image,
                "%s %.2f" % (name, item["score"]),
                (x1, max(16, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
        target = result["target"]
        text = "%s  visible=%d bearing=%.2f distance=%.2f" % (
            result["action"],
            int(target["visible"]),
            target["bearing_rad"],
            target["distance_m"],
        )
        cv2.putText(image, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 255, 0), 2, cv2.LINE_AA)
        return image

    @staticmethod
    def _resize_to_height(image, height):
        width = max(1, int(round(image.shape[1] * height / image.shape[0])))
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_NEAREST)

    def _publish_image(self, publisher, image, stamp):
        message = self.bridge.cv2_to_imgmsg(image, encoding="bgr8")
        message.header.stamp = stamp
        publisher.publish(message)

    def _loop(self):
        while not rospy.is_shutdown():
            try:
                item = self.queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                stamp, step_id, rgb, result, global_map, local_map = item
                detections = self._detections_image(rgb, result)
                global_color = self._global_color(global_map, result)
                local_color = self._local_color(local_map)
                global_large = cv2.resize(
                    global_color, None, fx=self.scale, fy=self.scale,
                    interpolation=cv2.INTER_NEAREST)
                local_large = self._resize_to_height(local_color, global_large.shape[0])
                comparison = np.concatenate((global_large, local_large), axis=1)
                cv2.putText(
                    comparison,
                    "Global Map                 Local Map  step=%d  action=%s" %
                    (step_id, result["action"]),
                    (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                self._publish_image(self.detection_pub, detections, stamp)
                self._publish_image(self.global_pub, global_large, stamp)
                self._publish_image(self.local_pub, local_large, stamp)
                self._publish_image(self.comparison_pub, comparison, stamp)
                self.probability_pub.publish(
                    Float32MultiArray(data=result["probabilities"]))
                self.timing_pub.publish(
                    Float32(data=float(result["inference_time_s"])))
            except Exception as exc:
                rospy.logwarn("ObjectNav visualization failed: %s", exc)
            finally:
                self.queue.task_done()
