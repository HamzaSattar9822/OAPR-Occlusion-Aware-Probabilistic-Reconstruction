"""COCO pose visualization with a virtual neck keypoint.

COCO-17 has no neck joint. For article figures we compute neck as the midpoint
of the shoulders and draw nose -> neck -> shoulders.
"""

import cv2
import numpy as np

NOSE, L_SHOULDER, R_SHOULDER = 0, 5, 6

COCO_BODY_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),           # head: nose-eyes-ears
    (5, 6),                                      # shoulders
    (5, 7), (7, 9), (6, 8), (8, 10),            # arms
    (5, 11), (6, 12),                            # torso
    (11, 12),                                    # hips
    (11, 13), (13, 15), (12, 14), (14, 16),     # legs
]

JOINT_COLORS = [
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0),
    (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 255, 85),
    (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
    (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255), (255, 0, 170),
]

NECK_COLOR = (0, 200, 255)


def neck_point(keypoints):
    """Shoulder midpoint used as the virtual neck location."""
    return (
        (keypoints[L_SHOULDER, 0] + keypoints[R_SHOULDER, 0]) * 0.5,
        (keypoints[L_SHOULDER, 1] + keypoints[R_SHOULDER, 1]) * 0.5,
    )


def _visible(confidences, indices, threshold):
    return all(confidences[i] >= threshold for i in indices)


def draw_pose_with_neck(image, keypoints, confidences, threshold=0.3,
                        line_width=2, joint_radius=4):
    """Draw COCO body skeleton plus nose-neck-shoulder connections."""
    vis = image.copy()
    kps = np.asarray(keypoints, dtype=np.float32)
    conf = np.asarray(confidences, dtype=np.float32).reshape(-1)

    for i, (a, b) in enumerate(COCO_BODY_SKELETON):
        if not _visible(conf, (a, b), threshold):
            continue
        xa, ya = int(kps[a, 0]), int(kps[a, 1])
        xb, yb = int(kps[b, 0]), int(kps[b, 1])
        color = JOINT_COLORS[i % len(JOINT_COLORS)]
        cv2.line(vis, (xa, ya), (xb, yb), color, line_width, cv2.LINE_AA)

    if _visible(conf, (NOSE, L_SHOULDER, R_SHOULDER), threshold):
        nx, ny = int(kps[NOSE, 0]), int(kps[NOSE, 1])
        neck_x, neck_y = neck_point(kps)
        neck_x, neck_y = int(neck_x), int(neck_y)
        ls_x, ls_y = int(kps[L_SHOULDER, 0]), int(kps[L_SHOULDER, 1])
        rs_x, rs_y = int(kps[R_SHOULDER, 0]), int(kps[R_SHOULDER, 1])
        for p1, p2 in [((nx, ny), (neck_x, neck_y)),
                       ((neck_x, neck_y), (ls_x, ls_y)),
                       ((neck_x, neck_y), (rs_x, rs_y))]:
            cv2.line(vis, p1, p2, NECK_COLOR, line_width, cv2.LINE_AA)
        cv2.circle(vis, (neck_x, neck_y), joint_radius, NECK_COLOR, -1, cv2.LINE_AA)
        cv2.circle(vis, (neck_x, neck_y), joint_radius, (255, 255, 255), 1, cv2.LINE_AA)

    for k in range(kps.shape[0]):
        if conf[k] < threshold:
            continue
        x, y = int(kps[k, 0]), int(kps[k, 1])
        color = JOINT_COLORS[k % len(JOINT_COLORS)]
        cv2.circle(vis, (x, y), joint_radius, color, -1, cv2.LINE_AA)
        cv2.circle(vis, (x, y), joint_radius, (255, 255, 255), 1, cv2.LINE_AA)

    return vis
