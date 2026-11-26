"""추론 파이프라인 (담당: kjh2)

영상 경로 → 프레임 → 관절 17개 → 특징 1,140개 → RF 예측 → 응답 dict

main.py(API 담당)는 analyze_video(video_path)만 부르고,
돌려받은 dict를 그대로 응답으로 보낸다.

[1단계] 아래 함수 4개는 모두 가짜 값을 돌려준다.
        하나씩 진짜로 바꿔 가도 analyze_video와 반환 모양은 그대로 둔다.
"""
import time

import cv2
import numpy as np

from models_loader import get_bundle, load_model_bundle

STEPS = 30          # 모델 입력: 30 Step
FEATURES = 38       # Step당 특징 38개 (좌표 34 + 관절 각도 4)


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
    """프레임마다 관절 17개를 뽑는다. [가짜] 빈 목록. (나중에: MediaPipe heavy → COCO 17개)"""
    return []


def make_features(keypoints):
    """30 Step × 38 = 1,140개 특징을 만든다. [가짜] 0으로 채움. (나중에: 학습과 같은 계산)"""
    return [0.0] * (STEPS * FEATURES)


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
