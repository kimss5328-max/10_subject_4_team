"""추론 파이프라인 (담당: kjh2)

영상 경로 → 프레임 → 관절 17개 → 특징 1,140개 → RF 예측 → 응답 dict

main.py(API 담당)는 analyze_video(video_path)만 부르고,
돌려받은 dict를 그대로 응답으로 보낸다.

[현재] 함수 4개(read_video, extract_keypoints, make_features, predict) 모두 진짜로 동작한다.
       analyze_video와 반환 모양은 1단계 그대로다.
"""
import math
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

from ai_service.models_loader import get_bundle, load_model_bundle

STEPS = 30          # 모델 입력: 30 Step
FEATURES = 38       # Step당 특징 38개 (좌표 34 + 관절 각도 4)

POSE_MODEL = Path(__file__).resolve().parent / "models" / "pose_landmarker_heavy.task"
ASSUMED_FPS = 30    # VIDEO 모드 시간값용 (read_video가 FPS를 돌려주지 않아서 가정)

# COCO 17개 순서대로, 각 관절이 MediaPipe 33개 중 몇 번인지
# nose, left/right_eye, left/right_ear, left/right_shoulder, left/right_elbow,
# left/right_wrist, left/right_hip, left/right_knee, left/right_ankle
MP_TO_COCO = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]


def read_video(video_path):
    """영상에서 프레임 목록을 꺼낸다. OpenCV가 읽은 그대로(BGR) 돌려준다."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"영상을 열 수 없습니다: {video_path}")

    frames = []
    while True:
        ok, frame = cap.read()   # 한 장씩 읽는다. 끝나면 ok가 False
        if not ok:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        raise ValueError(f"영상에서 프레임을 하나도 읽지 못했습니다: {video_path}")
    return frames


def extract_keypoints(frames):
    """프레임마다 COCO 17개 관절의 픽셀 좌표 [(x, y), ...]를 뽑는다. 사람이 없으면 None."""
    options = vision.PoseLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=str(POSE_MODEL)),
        running_mode=vision.RunningMode.VIDEO,   # 앞뒤 Frame을 이어서 본다 (학습 CSV와 가장 비슷)
        num_poses=1,                             # 한 사람만
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)   # 영상마다 새로 → 시간값 0부터

    keypoints = []
    for i, frame in enumerate(frames):
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)              # MediaPipe는 RGB
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(image, int(i * 1000 / ASSUMED_FPS))   # 시간값(ms)

        if not result.pose_landmarks:
            keypoints.append(None)                                # 사람을 못 찾은 Frame
            continue
        points = result.pose_landmarks[0]                         # 33개, 0~1 비율 좌표
        keypoints.append([(points[j].x * w, points[j].y * h) for j in MP_TO_COCO])   # 픽셀로

    landmarker.close()
    return keypoints


def make_features(keypoints):
    """Frame별 관절 → 30 Step × 38 = 1,140개 특징.
    학습 코드(sequence_utils.py, feature_utils.py)와 같은 순서·같은 식을 numpy로 옮겼다.
    """
    # 각도를 잴 세 점 (끝점 A, 가운데 B, 끝점 C). B에서 꺾인 각도를 잰다. 숫자는 COCO 번호
    angle_joints = [
        (5, 7, 9),     # 왼 팔꿈치: 왼 어깨 - 왼 팔꿈치 - 왼 손목
        (6, 8, 10),    # 오른 팔꿈치: 오른 어깨 - 오른 팔꿈치 - 오른 손목
        (11, 13, 15),  # 왼 무릎: 왼 골반 - 왼 무릎 - 왼 발목
        (12, 14, 16),  # 오른 무릎: 오른 골반 - 오른 무릎 - 오른 발목
    ]

    rows = []        # 쓸 수 있는 Frame의 38개 특징
    frame_ids = []   # 그 Frame의 번호 (30 Step으로 나눌 때 시간 위치로 쓴다)
    for i, kp in enumerate(keypoints):
        # 1) Frame 고르기
        if kp is None:                                   # 사람을 못 찾은 Frame은 버린다
            continue
        points = np.asarray(kp, dtype=np.float32)        # 17 × 2 픽셀 좌표
        if not np.all(np.isfinite(points)):              # 숫자가 아닌 값이 있으면 버린다
            continue

        # 2) 정규화: 골반 중점을 (0, 0)으로 옮기고, 어깨 폭으로 나눈다
        center = (points[11] + points[12]) / 2.0                 # 왼·오른 골반의 중점 = 몸 중심
        scale = float(np.linalg.norm(points[5] - points[6]))     # 왼·오른 어깨 사이 거리
        if scale < 1e-8:                                         # 어깨 폭을 못 재면 버린다
            continue
        normalized = (points - center) / scale                   # 사람 크기·위치와 상관없는 좌표

        # 3) 관절 각도 4개 (단위: 도). 정규화 전 픽셀 좌표로 잰다
        angles = []
        for a, b, c in angle_joints:
            ba = points[a] - points[b]                   # 가운데 점 → 끝점 A 방향
            bc = points[c] - points[b]                   # 가운데 점 → 끝점 C 방향
            norm_ba = np.linalg.norm(ba)                 # 두 방향의 길이
            norm_bc = np.linalg.norm(bc)
            if norm_ba < 1e-8 or norm_bc < 1e-8:         # 점이 겹치면 각도를 못 잰다
                angles.append(float("nan"))
                continue
            cosine = np.dot(ba, bc) / (norm_ba * norm_bc)    # 두 방향 사이 각도의 cos
            cosine = np.clip(cosine, -1.0, 1.0)              # 계산 오차로 1을 넘지 않게
            angles.append(math.degrees(math.acos(float(cosine))))   # cos → 각도(도)

        # 4) Frame당 38개 = 정규화 좌표 34개 (관절마다 x, y) + 각도 4개
        row = np.asarray(list(normalized.reshape(-1)) + angles, dtype=np.float32)
        if not np.all(np.isfinite(row)):                 # 각도를 못 잰 Frame은 버린다
            continue
        rows.append(row)
        frame_ids.append(i)

    # 쓸 수 있는 Frame이 너무 적으면 예측하지 않는다 (학습 때도 이런 영상은 뺐다)
    min_valid = get_bundle().min_valid_frames
    if len(rows) < min_valid:
        raise ValueError(f"사람이 보이는 Frame이 {len(rows)}개입니다. {min_valid}개 이상 필요합니다.")

    # 5) 30 Step으로 맞추기: Frame 번호를 0~1 위치로 바꾸고, 30칸 위치의 값을 선형 보간
    matrix = np.asarray(rows, dtype=np.float32)                # (유효 Frame 수) × 38
    ids = np.asarray(frame_ids, dtype=np.float32)
    original_x = (ids - ids[0]) / (ids[-1] - ids[0])           # 첫 Frame 0, 마지막 Frame 1
    target_x = np.linspace(0.0, 1.0, STEPS)                    # 0~1을 같은 간격 30칸으로
    sequence = np.empty((STEPS, FEATURES), dtype=np.float32)   # 30 × 38
    for f in range(FEATURES):                                  # 특징 하나씩
        sequence[:, f] = np.interp(target_x, original_x, matrix[:, f])   # 사이 값은 직선으로 채움
    expected = get_bundle().feature_count
    if sequence.size != expected:
        raise ValueError (f"입력 길이 {sequence.size} 기대 길이 {expected}")
    # 6) Step 순서대로 한 줄로 펴면 1,140개 (rf_feature_columns.json 순서와 같다)
    return sequence.reshape(-1).tolist()


def predict(features):
    """(action_label, confidence)를 돌려준다. 모델은 models_loader가 불러 둔 것을 쓴다."""
    bundle = get_bundle()
    model = bundle.model

    if len(features) != bundle.feature_count:
        raise ValueError(f"특징 개수가 {len(features)}개입니다. {bundle.feature_count}개여야 합니다.")

    x = np.array(features, dtype=float).reshape(1, -1)   # 1행 × 1,140열
    proba = model.predict_proba(x)[0]                     # 클래스 3개의 확률
    best = int(np.argmax(proba))                          # 가장 큰 확률의 위치

    label = str(model.classes_[best])
    confidence = float(proba[best])
    return label, confidence


def analyze_video(video_path):
    """위 4개를 순서대로 부르고, 계약 모양의 dict를 돌려준다."""
    started = time.time()

    frames = read_video(video_path)
    keypoints = extract_keypoints(frames)
    features = make_features(keypoints)
    label, confidence = predict(features)

    return {
        "status": "ok",
        "message": "1명의 동작을 예측했습니다.",
        "predictions": [
            {"track_id": 1, "action_label": label, "confidence": round(confidence, 4)},
        ],
        "processing_time_sec": round(time.time() - started, 3),
    }


if __name__ == "__main__":
    # 직접 확인용: python pipeline.py
    load_model_bundle()   # 서버에서는 main.py가 켤 때 부른다
    print(analyze_video("/home/user1/ai_vision/test_videos/S03_A006_T01.mp4"))
