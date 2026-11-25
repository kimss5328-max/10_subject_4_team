from pathlib import Path

from django.shortcuts import redirect, render

from .models import AnalysisJob

ALLOWED_TYPES = {"mp4", "mov", "avi"}  # 템플릿 accept=".mp4,.mov,.avi" 와 맞춤


def index(request):
    """GET: 작업 기록 보여주기 / POST: 영상 받아서 작업 저장 → 목록으로 이동."""
    if request.method == "POST":
        video = request.FILES.get("video")  # 템플릿 name="video"
        file_type = Path(video.name).suffix.lstrip(".").lower() if video else ""

        error = None
        if not video:
            error = "영상을 선택하세요."
        elif file_type not in ALLOWED_TYPES:
            error = f"MP4 · MOV · AVI 영상만 올릴 수 있어요. (받은 형식: {file_type or '없음'})"
        elif video.size == 0:
            error = "빈 파일이에요."

        if error:
            jobs = AnalysisJob.objects.all()
            return render(request, "project10/index.html", {"jobs": jobs, "error": error})

        AnalysisJob.objects.create(video=video, file_type=file_type)  # status = PENDING
        # [B] 여기서 FastAPI를 호출해 action_label·confidence·status·processing_time을 채운다.

        return redirect("index")  # 새로고침해도 같은 영상이 다시 올라가지 않게

    jobs = AnalysisJob.objects.all()
    return render(request, "project10/index.html", {"jobs": jobs})