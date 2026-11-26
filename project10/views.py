from pathlib import Path

from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_POST

from .models import AnalysisJob
from .services import run_prediction  # [B] FastAPI 호출

ALLOWED_TYPES = {"mp4", "mov", "avi"}  # 템플릿 accept=".mp4,.mov,.avi" 와 맞춤


def index(request):
    """GET /  : 처음 화면. 저장된 작업 기록을 템플릿에 넘긴다."""
    jobs = AnalysisJob.objects.all()  # 최신순 (models.py Meta.ordering)
    return render(request, "project10/index.html", {"jobs": jobs})


@require_POST  # GET으로 오면 405
def analyze(request):
    """POST /analyze/ : analyze.js가 보낸 영상을 받아 작업을 저장하고 JSON으로 돌려준다."""
    video = request.FILES.get("video")  # 템플릿 name="video"
    file_type = Path(video.name).suffix.lstrip(".").lower() if video else ""

    if not video:
        return JsonResponse({"error": "영상을 선택하세요."}, status=400)
    if file_type not in ALLOWED_TYPES:
        return JsonResponse({"error": f"MP4 · MOV · AVI 영상만 올릴 수 있어요. (받은 형식: {file_type or '없음'})"},
                            status=400)
    if video.size == 0:
        return JsonResponse({"error": "빈 파일이에요."}, status=400)

    job = AnalysisJob.objects.create(video=video, file_type=file_type)  # status = PENDING
    # [B] 여기서 FastAPI를 호출해 action_label·confidence·status·processing_time을 채운다.
    run_prediction(job)

    # analyze.js가 표에 한 줄 추가할 때 쓰는 키 6개
    return JsonResponse({
        "id": job.id,
        "action_label": job.action_label,
        "confidence": job.confidence,
        "file_type": job.file_type,
        "status": job.status,
        "processing_time": job.processing_time,
    })