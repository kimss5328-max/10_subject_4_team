import shutil
import tempfile

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings

from .models import AnalysisJob

TMP_MEDIA = tempfile.mkdtemp()


def video(name="clip.mp4", content=b"0" * 1024):
    return SimpleUploadedFile(name, content, content_type="video/mp4")


@override_settings(MEDIA_ROOT=TMP_MEDIA)
class IndexTests(TestCase):
    """GET / : 처음 화면"""

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        shutil.rmtree(TMP_MEDIA, ignore_errors=True)

    def test_empty_list(self):
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertContains(res, "아직 작업 기록이 없어요.")

    def test_list_shows_job(self):
        AnalysisJob.objects.create(video="uploads/a.mp4", file_type="mp4", status="SUCCESS",
                                   action_label="wave", confidence=0.9, processing_time=3.2)
        res = self.client.get("/")
        for text in ["wave", "0.90", "mp4", "SUCCESS", "3.2초"]:
            self.assertContains(res, text)


@override_settings(MEDIA_ROOT=TMP_MEDIA)
class AnalyzeTests(TestCase):
    """POST /analyze/ : analyze.js가 부르는 주소"""

    def test_upload_returns_json_with_6_keys(self):
        res = self.client.post("/analyze/", {"video": video()})
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertEqual(set(body), {"id", "action_label", "confidence", "file_type", "status", "processing_time"})
        self.assertEqual((body["file_type"], body["status"]), ("mp4", "PENDING"))
        self.assertTrue(AnalysisJob.objects.filter(pk=body["id"]).exists())

    def test_errors_return_400_with_error_key(self):
        cases = {"영상을 선택하세요": {}, "MP4 · MOV · AVI": {"video": video("memo.txt")},
                 "빈 파일": {"video": video(content=b"")}}
        for message, data in cases.items():
            with self.subTest(message):
                res = self.client.post("/analyze/", data)
                self.assertEqual(res.status_code, 400)
                self.assertIn(message, res.json()["error"])
        self.assertFalse(AnalysisJob.objects.exists())

    def test_get_not_allowed(self):
        self.assertEqual(self.client.get("/analyze/").status_code, 405)

    def test_csrf_required(self):
        """프론트가 X-CSRFToken 헤더를 보내야 통과 (토큰 없으면 403)."""
        from django.test import Client
        strict = Client(enforce_csrf_checks=True)
        self.assertEqual(strict.post("/analyze/", {"video": video()}).status_code, 403)