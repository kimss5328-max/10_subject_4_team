"""추론 파이프라인 (담당: kjh2)

영상 경로 → 프레임 → 관절 17개 → 특징 1,140개 → RF 예측 → 응답 dict

main.py(API 담당)는 analyze_video(video_path)만 부르고,
돌려받은 dict를 그대로 응답으로 보낸다.

[1단계] 아래 함수 4개는 모두 가짜 값을 돌려준다.
        하나씩 진짜로 바꿔 가도 analyze_video와 반환 모양은 그대로 둔다.
"""
import time

STEPS = 30          # 모델 입력: 30 Step
FEATURES = 38       # Step당 특징 38개 (좌표 34 + 관절 각도 4)


def read_video(video_path):
    """영상에서 프레임 목록을 꺼낸다. [가짜] 빈 목록. (나중에: OpenCV)"""
    return []


def extract_keypoints(frames):
    """프레임마다 관절 17개를 뽑는다. [가짜] 빈 목록. (나중에: MediaPipe heavy → COCO 17개)"""
    return []


def make_features(keypoints):
    """30 Step × 38 = 1,140개 특징을 만든다. [가짜] 0으로 채움. (나중에: 학습과 같은 계산)"""
    return [0.0] * (STEPS * FEATURES)


def predict(features):
    """(action_label, confidence)를 돌려준다. [가짜] 고정 값. (나중에: models/rf_action_team.joblib)"""
    return "side_lying_leg_raise", 0.8


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
    print(analyze_video("test.mp4"))
