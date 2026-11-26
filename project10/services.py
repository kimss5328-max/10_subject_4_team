"""[B] Django → FastAPI 연결 (동기 방식).

A의 views.analyze()가 저장한 job을 받아서
FastAPI에 영상을 보내고, 결과를 job의 네 칸에 채워 저장한다.
    action_label · confidence · status(SUCCESS / FAILURE) · processing_time
"""
import time
from pathlib import Path

import requests

from .models import AnalysisJob

FASTAPI_URL = "http://localhost:8001/predict/video"  # FastAPI(C·D) 주소
TIMEOUT_SEC = 300                                    # 분석이 길어도 최대 5분까지 기다림


def run_prediction(job):
    """영상 1개를 FastAPI로 보내고, 결과를 job에 채워 저장한다."""
    started = time.time()                                   # 1. 시작 시간

    try:
        with job.video.open("rb") as f:                     # 2. 저장된 영상 열기
            res = requests.post(                            #    FastAPI로 전송 (필드 이름 "file")
                FASTAPI_URL,
                files={"file": (Path(job.video.name).name, f)},
                timeout=TIMEOUT_SEC,
            )
        res.raise_for_status()                              #    400·500 같은 오류 응답이면 except로

        first = _pick_first(res.json())                     #    첫 번째 사람 결과
        if first:                                           #    사람이 있으면 두 칸 채움
            job.action_label = first.get("action_label", "")
            job.confidence = first.get("confidence")
        job.status = AnalysisJob.Status.SUCCESS             #    사람이 없어도 SUCCESS (칸은 "-")

    except (requests.RequestException, ValueError):
        # FastAPI 꺼짐 · 시간 초과 · 오류 응답 · JSON이 아님 → 실패로 기록
        job.status = AnalysisJob.Status.FAILURE

    job.processing_time = time.time() - started             # 3. 처리 시간(초)
    job.save()                                              # 4. 저장
    return job                                              # 5. 돌려줌


def _pick_first(data):
    """FastAPI 응답에서 첫 번째 사람 결과를 꺼낸다. 없으면 None."""
    predictions = data.get("predictions") or []
    return predictions[0] if predictions else None