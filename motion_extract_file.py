import argparse
import csv
import os
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


LANDMARK_COUNT = 33
POSE_CONNECTIONS = mp.solutions.pose.POSE_CONNECTIONS


def calculate_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Calculate the angle between three points in 3D space.
    The angle is calculated at point b.
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    denominator = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denominator == 0:
        return float("nan")

    cosine_angle = np.dot(ba, bc) / denominator
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)


def landmark_to_tuple(landmark) -> Tuple[float, float, float]:
    return (landmark.x, landmark.y, landmark.z)


def draw_pose(image: np.ndarray, landmarks, color: Tuple[int, int, int]):
    h, w, _ = image.shape

    for start_idx, end_idx in POSE_CONNECTIONS:
        if start_idx >= len(landmarks) or end_idx >= len(landmarks):
            continue

        start = landmarks[start_idx]
        end = landmarks[end_idx]
        start_point = (int(start.x * w), int(start.y * h))
        end_point = (int(end.x * w), int(end.y * h))
        cv2.line(image, start_point, end_point, color, 2)

    for landmark in landmarks:
        point = (int(landmark.x * w), int(landmark.y * h))
        cv2.circle(image, point, 3, color, -1)


def draw_pose_tuples(
    image: np.ndarray,
    landmarks: List[Tuple[float, float, float]],
    color: Tuple[int, int, int],
):
    h, w, _ = image.shape

    for start_idx, end_idx in POSE_CONNECTIONS:
        if start_idx >= len(landmarks) or end_idx >= len(landmarks):
            continue

        start = landmarks[start_idx]
        end = landmarks[end_idx]
        start_point = (int(start[0] * w), int(start[1] * h))
        end_point = (int(end[0] * w), int(end[1] * h))
        cv2.line(image, start_point, end_point, color, 2)

    for x, y, _ in landmarks:
        point = (int(x * w), int(y * h))
        cv2.circle(image, point, 3, color, -1)


def get_person_colors(count: int) -> List[Tuple[int, int, int]]:
    base_colors = [
        (0, 255, 0),
        (255, 128, 0),
        (0, 200, 255),
        (255, 0, 255),
        (255, 255, 0),
        (180, 180, 255),
    ]
    return [base_colors[i % len(base_colors)] for i in range(count)]


def create_pose_landmarker(model_path: str, max_poses: int, running_mode):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Pose landmarker model not found: {model_path}\n"
            "Download a MediaPipe pose_landmarker .task model and pass it with --model."
        )

    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=model_path),
        running_mode=running_mode,
        num_poses=max_poses,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )
    return vision.PoseLandmarker.create_from_options(options)


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def has_box_overlap(box: np.ndarray, boxes: np.ndarray, threshold: float) -> bool:
    return any(
        box_iou(box, other) > threshold
        for other in boxes
        if not np.array_equal(box, other)
    )


def crop_person(
    image: np.ndarray,
    box: np.ndarray,
    padding: float,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    h, w, _ = image.shape
    x1, y1, x2, y2 = box
    box_width = x2 - x1
    box_height = y2 - y1
    pad_x = box_width * padding
    pad_y = box_height * padding

    crop_x1 = max(0, int(x1 - pad_x))
    crop_y1 = max(0, int(y1 - pad_y))
    crop_x2 = min(w, int(x2 + pad_x))
    crop_y2 = min(h, int(y2 + pad_y))

    return image[crop_y1:crop_y2, crop_x1:crop_x2], (
        crop_x1,
        crop_y1,
        crop_x2 - crop_x1,
        crop_y2 - crop_y1,
    )


def convert_crop_landmarks_to_full_frame(
    landmarks,
    crop_bounds: Tuple[int, int, int, int],
    frame_width: int,
    frame_height: int,
) -> List[Tuple[float, float, float]]:
    crop_x, crop_y, crop_width, crop_height = crop_bounds
    z_scale = crop_width / frame_width if frame_width else 1.0

    full_frame_landmarks = []
    for landmark in landmarks:
        full_x = (crop_x + landmark.x * crop_width) / frame_width
        full_y = (crop_y + landmark.y * crop_height) / frame_height
        full_z = landmark.z * z_scale
        full_frame_landmarks.append((full_x, full_y, full_z))

    return full_frame_landmarks


def clamp_landmarks_to_box(
    landmarks: List[Tuple[float, float, float]],
    box: np.ndarray,
    frame_width: int,
    frame_height: int,
) -> List[Tuple[float, float, float]]:
    x1, y1, x2, y2 = box
    min_x = x1 / frame_width
    max_x = x2 / frame_width
    min_y = y1 / frame_height
    max_y = y2 / frame_height

    return [
        (
            float(np.clip(x, min_x, max_x)),
            float(np.clip(y, min_y, max_y)),
            z,
        )
        for x, y, z in landmarks
    ]


def smooth_landmarks(
    previous_landmarks: Optional[List[Tuple[float, float, float]]],
    current_landmarks: List[Tuple[float, float, float]],
    smoothing: float,
) -> List[Tuple[float, float, float]]:
    if previous_landmarks is None or smoothing <= 0:
        return current_landmarks

    if len(previous_landmarks) != len(current_landmarks):
        return current_landmarks

    return [
        (
            smoothing * previous[0] + (1 - smoothing) * current[0],
            smoothing * previous[1] + (1 - smoothing) * current[1],
            smoothing * previous[2] + (1 - smoothing) * current[2],
        )
        for previous, current in zip(previous_landmarks, current_landmarks)
    ]


def create_yolo_model(model_name: str):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "YOLO person gating requires Ultralytics. Install it with: pip install ultralytics"
        ) from exc

    return YOLO(model_name)


def detect_people_with_yolo(
    yolo_model,
    image: np.ndarray,
    confidence: float,
    iou_threshold: float,
) -> List[dict]:
    results = yolo_model.track(
        image,
        classes=[0],
        persist=True,
        conf=confidence,
        iou=iou_threshold,
        verbose=False,
    )

    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        return []

    boxes = results[0].boxes
    xyxy = boxes.xyxy.cpu().numpy()
    confidences = boxes.conf.cpu().numpy()
    track_ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None

    detections = []
    for idx, box in enumerate(xyxy):
        detections.append(
            {
                "box": box,
                "confidence": float(confidences[idx]),
                "track_id": int(track_ids[idx]) if track_ids is not None else idx,
            }
        )
    return detections


def save_to_csv(
    landmarks_history: List[
        Tuple[int, int, Optional[float], bool, List[Tuple[float, float, float]]]
    ],
    filename: str = "pose_data.csv",
    output_person_id: Optional[int] = 1,
):
    """
    Save pose landmarks history to a CSV file.
    Each row represents one detected person in one frame.
    """
    if output_person_id is not None:
        landmarks_history = [
            row for row in landmarks_history if row[1] == output_person_id
        ]

    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        header = ["frame", "person_id", "yolo_confidence", "overlap_warning"]
        for i in range(LANDMARK_COUNT):
            header.extend([f"landmark_{i}_x", f"landmark_{i}_y", f"landmark_{i}_z"])
        writer.writerow(header)

        for frame_idx, person_id, yolo_confidence, overlap_warning, landmarks in landmarks_history:
            row = [
                frame_idx,
                person_id,
                "" if yolo_confidence is None else yolo_confidence,
                int(overlap_warning),
            ]
            for landmark in landmarks:
                row.extend(landmark)
            writer.writerow(row)

    if output_person_id is None:
        print(f"Data saved to {filename}")
    else:
        print(f"Data saved to {filename} for person ID {output_person_id}")


def default_annotated_video_filename(output_file: str) -> str:
    base, _ = os.path.splitext(output_file)
    return f"{base}_annotated.mp4"


def process_video(
    input_file: str,
    output_file: str,
    annotated_video_file: Optional[str],
    model_path: str,
    max_poses: int,
    use_yolo_gate: bool,
    yolo_model_name: str,
    yolo_confidence: float,
    yolo_iou: float,
    overlap_iou: float,
    crop_padding: float,
    keep_overlaps: bool,
    landmark_smoothing: float,
    clamp_pose_to_yolo_box: bool,
    priority_person_id: int,
    output_person_id: Optional[int],
):
    """
    Process a video file and extract pose data for up to max_poses people.
    """
    cap = cv2.VideoCapture(input_file)

    if not cap.isOpened():
        print(f"Error: Could not open video file {input_file}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    timestamp_step_ms = int(1000 / fps) if fps and fps > 0 else 33
    writer_fps = fps if fps and fps > 0 else 30.0

    print(f"Processing video: {input_file}")
    print(f"Resolution: {frame_width}x{frame_height}, FPS: {fps}, Total frames: {total_frames}")
    print(f"Detecting up to {max_poses} people per frame")

    annotated_writer = None
    if annotated_video_file:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        annotated_writer = cv2.VideoWriter(
            annotated_video_file,
            fourcc,
            writer_fps,
            (frame_width, frame_height),
        )
        if annotated_writer.isOpened():
            print(f"Annotated overlay video will be saved to {annotated_video_file}")
        else:
            print(f"Warning: Could not open annotated video output: {annotated_video_file}")
            annotated_writer = None

    yolo_model = create_yolo_model(yolo_model_name) if use_yolo_gate else None
    running_mode = vision.RunningMode.IMAGE if use_yolo_gate else vision.RunningMode.VIDEO
    pose_max_count = 1 if use_yolo_gate else max_poses

    if use_yolo_gate:
        print(f"YOLO person gate enabled with model: {yolo_model_name}")
        print(f"Skipping overlapped people above IoU {overlap_iou} unless --keep-overlaps is set")
        if output_person_id is not None:
            print(f"Only person ID {output_person_id} will be written to the CSV")

    with create_pose_landmarker(model_path, pose_max_count, running_mode) as pose:
        landmarks_history = []
        smoothed_landmarks_by_person = {}
        frame_count = 0

        cv2.namedWindow("MediaPipe Multi-Person Pose Estimation", cv2.WINDOW_NORMAL)

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("End of video or error reading frame.")
                break

            frame_count += 1
            h, _, _ = image.shape
            clean_image = image.copy()

            if use_yolo_gate:
                detections = detect_people_with_yolo(yolo_model, clean_image, yolo_confidence, yolo_iou)
                detections = detections[:max_poses]
                boxes = np.array([detection["box"] for detection in detections])
                colors = get_person_colors(len(detections))

                for detection_idx, detection in enumerate(detections):
                    box = detection["box"]
                    person_id = detection["track_id"]
                    overlap_warning = has_box_overlap(box, boxes, overlap_iou)
                    color = (0, 0, 255) if overlap_warning else colors[detection_idx]

                    x1, y1, x2, y2 = box.astype(int)
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        image,
                        f"ID {person_id} {detection['confidence']:.2f}",
                        (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        color,
                        2,
                    )

                    if output_person_id is not None and person_id != output_person_id:
                        continue

                    should_skip_overlap = (
                        overlap_warning
                        and not keep_overlaps
                        and person_id != priority_person_id
                    )

                    if should_skip_overlap:
                        cv2.putText(
                            image,
                            "Skipped: overlap",
                            (x1, min(frame_height - 10, y2 + 20)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            color,
                            2,
                        )
                        continue
                    elif overlap_warning and person_id == priority_person_id:
                        cv2.putText(
                            image,
                            "Kept: priority overlap",
                            (x1, min(frame_height - 10, y2 + 20)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            color,
                            2,
                        )

                    crop, crop_bounds = crop_person(clean_image, box, crop_padding)
                    if crop.size == 0:
                        continue

                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_rgb)
                    results = pose.detect(mp_image)
                    pose_landmarks = results.pose_landmarks or []
                    if not pose_landmarks:
                        continue

                    frame_landmarks = convert_crop_landmarks_to_full_frame(
                        pose_landmarks[0],
                        crop_bounds,
                        frame_width,
                        frame_height,
                    )
                    if clamp_pose_to_yolo_box:
                        frame_landmarks = clamp_landmarks_to_box(
                            frame_landmarks,
                            box,
                            frame_width,
                            frame_height,
                        )
                    frame_landmarks = smooth_landmarks(
                        smoothed_landmarks_by_person.get(person_id),
                        frame_landmarks,
                        landmark_smoothing,
                    )
                    if clamp_pose_to_yolo_box:
                        frame_landmarks = clamp_landmarks_to_box(
                            frame_landmarks,
                            box,
                            frame_width,
                            frame_height,
                        )
                    smoothed_landmarks_by_person[person_id] = frame_landmarks
                    landmarks_history.append(
                        (
                            frame_count,
                            person_id,
                            detection["confidence"],
                            overlap_warning,
                            frame_landmarks,
                        )
                    )

                    draw_pose_tuples(image, frame_landmarks, color)
                    draw_knee_angles(image, frame_landmarks, person_id, color, detection_idx)
            else:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
                timestamp_ms = (frame_count - 1) * timestamp_step_ms
                results = pose.detect_for_video(mp_image, timestamp_ms)

                pose_landmarks = results.pose_landmarks or []
                colors = get_person_colors(len(pose_landmarks))

                for person_id, person_landmarks in enumerate(pose_landmarks):
                    if output_person_id is not None and person_id != output_person_id:
                        continue

                    frame_landmarks = [landmark_to_tuple(landmark) for landmark in person_landmarks]
                    frame_landmarks = smooth_landmarks(
                        smoothed_landmarks_by_person.get(person_id),
                        frame_landmarks,
                        landmark_smoothing,
                    )
                    smoothed_landmarks_by_person[person_id] = frame_landmarks
                    landmarks_history.append((frame_count, person_id, None, False, frame_landmarks))

                    draw_pose(image, person_landmarks, colors[person_id])
                    draw_knee_angles(image, frame_landmarks, person_id, colors[person_id], person_id)

            progress = frame_count / total_frames * 100 if total_frames else 0
            cv2.putText(
                image,
                f"Progress: {progress:.1f}% (Frame {frame_count}/{total_frames})",
                (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

            if annotated_writer is not None:
                annotated_writer.write(image)

            cv2.imshow("MediaPipe Multi-Person Pose Estimation", image)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            if frame_count % 100 == 0:
                print(f"Processed {frame_count}/{total_frames} frames ({progress:.1f}%)")

        cap.release()
        if annotated_writer is not None:
            annotated_writer.release()
        cv2.destroyAllWindows()

        if landmarks_history:
            save_to_csv(landmarks_history, output_file, output_person_id)
            print(f"Processing complete. Data saved to {output_file}")
        else:
            print("No pose landmarks detected in the video.")


def draw_knee_angles(
    image: np.ndarray,
    frame_landmarks: List[Tuple[float, float, float]],
    person_id: int,
    color: Tuple[int, int, int],
    display_index: int,
):
    left_knee_angle = calculate_angle(
        np.array(frame_landmarks[23]),
        np.array(frame_landmarks[25]),
        np.array(frame_landmarks[27]),
    )
    right_knee_angle = calculate_angle(
        np.array(frame_landmarks[24]),
        np.array(frame_landmarks[26]),
        np.array(frame_landmarks[28]),
    )

    y_offset = 30 + display_index * 55
    cv2.putText(
        image,
        f"Person {person_id} L knee: {left_knee_angle:.1f} deg",
        (10, y_offset),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
    )
    cv2.putText(
        image,
        f"Person {person_id} R knee: {right_knee_angle:.1f} deg",
        (10, y_offset + 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Extract multi-person pose data from a video file using MediaPipe."
    )
    parser.add_argument("-i", "--input", required=True, help="Input video file path")
    parser.add_argument("-o", "--output", required=True, help="Output CSV file path")
    parser.add_argument(
        "-m",
        "--model",
        default="pose_landmarker_full.task",
        help="Path to a MediaPipe pose_landmarker .task model file",
    )
    parser.add_argument(
        "--max-poses",
        type=int,
        default=2,
        help="Maximum number of people to detect per frame",
    )
    parser.add_argument(
        "--use-yolo-gate",
        action="store_true",
        help="Use YOLO person boxes/tracking before running MediaPipe pose on each person crop",
    )
    parser.add_argument(
        "--yolo-model",
        default="yolo11s.pt",
        help="Ultralytics YOLO detection model to use for person tracking",
    )
    parser.add_argument(
        "--yolo-confidence",
        type=float,
        default=0.45,
        help="Minimum YOLO confidence for person detections",
    )
    parser.add_argument(
        "--yolo-iou",
        type=float,
        default=0.45,
        help="YOLO NMS IoU threshold",
    )
    parser.add_argument(
        "--overlap-iou",
        type=float,
        default=0.20,
        help="Person boxes with IoU above this value are treated as overlapping",
    )
    parser.add_argument(
        "--crop-padding",
        type=float,
        default=0.35,
        help="Padding added around each YOLO person box before MediaPipe pose detection",
    )
    parser.add_argument(
        "--keep-overlaps",
        action="store_true",
        help="Keep overlapped people instead of skipping them",
    )
    parser.add_argument(
        "--landmark-smoothing",
        type=float,
        default=0.80,
        help="Temporal landmark smoothing from 0.0 to 0.95; higher is smoother but more delayed",
    )
    parser.add_argument(
        "--allow-pose-outside-yolo-box",
        action="store_true",
        help="Allow MediaPipe landmarks from YOLO crops to extend outside the YOLO person box",
    )
    parser.add_argument(
        "--priority-person-id",
        type=int,
        default=1,
        help="YOLO track ID that should still be recorded during overlap while other overlapped people are skipped",
    )
    parser.add_argument(
        "--output-person-id",
        type=int,
        default=1,
        help="Only write this person/track ID to the output CSV. Use -1 to write all IDs.",
    )
    parser.add_argument(
        "--annotated-video",
        default=None,
        help="Path for the skeleton/overlay output video. Defaults to '<output>_annotated.mp4'.",
    )
    parser.add_argument(
        "--no-annotated-video",
        action="store_true",
        help="Do not save an annotated skeleton/overlay video.",
    )

    args = parser.parse_args()
    args.landmark_smoothing = min(max(args.landmark_smoothing, 0.0), 0.95)
    output_person_id = None if args.output_person_id < 0 else args.output_person_id
    annotated_video_file = None
    if not args.no_annotated_video:
        annotated_video_file = args.annotated_video or default_annotated_video_filename(args.output)

    process_video(
        args.input,
        args.output,
        annotated_video_file,
        args.model,
        args.max_poses,
        args.use_yolo_gate,
        args.yolo_model,
        args.yolo_confidence,
        args.yolo_iou,
        args.overlap_iou,
        args.crop_padding,
        args.keep_overlaps,
        args.landmark_smoothing,
        not args.allow_pose_outside_yolo_box,
        args.priority_person_id,
        output_person_id,
    )


if __name__ == "__main__":
    main()
