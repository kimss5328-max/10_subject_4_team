from django.db import models

# Create your models here.
from django.db import models


class AnalysisJob(models.Model):
    """분석 작업 1건 = 업로드한 영상 1개.

    A가 채우는 칸: video, file_type, status(PENDING), created_at
    B가 채우는 칸: action_label, confidence, status(SUCCESS/FAILURE), processing_time
    """

    class Status(models.TextChoices):
        PENDING = "PENDING", "대기"
        SUCCESS = "SUCCESS", "완료"
        FAILURE = "FAILURE", "실패"

    video = models.FileField("영상", upload_to="uploads/")              # 파일은 media/ 폴더, DB에는 경로만
    file_type = models.CharField("파일 유형", max_length=10, blank=True)  # mp4, mov, avi
    action_label = models.CharField("자세", max_length=40, blank=True)
    confidence = models.FloatField("신뢰도", null=True, blank=True)
    status = models.CharField("결과", max_length=10, choices=Status.choices, default=Status.PENDING)
    processing_time = models.FloatField("처리 시간(초)", null=True, blank=True)
    created_at = models.DateTimeField("생성", auto_now_add=True)

    class Meta:
        ordering = ["-id"]  # 최신 작업이 위로

    def __str__(self):
        return f"#{self.pk} {self.file_type} {self.status}"