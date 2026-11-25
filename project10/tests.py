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
    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        shutil.rmtree(TMP_MEDIA, ignore_errors=True)

    def test_empty_list(self):
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertContains(res, "아직 작업 기록이 없어요.")

    def test_upload_saves_pending_and_redirects(self):
        res = self.client.post("/", {"video": video()})
        self.assertRedirects(res, "/")
        job = AnalysisJob.objects.get()
        self.assertEqual((job.file_type, job.status), ("mp4", "PENDING"))

    def test_list_shows_job(self):
        AnalysisJob.objects.create(video="uploads/a.mp4", file_type="mp4", status="SUCCESS",
                                   action_label="wave", confidence=0.9, processing_time=3.2)
        res = self.client.get("/")
        for text in ["wave", "0.90", "mp4", "SUCCESS", "3.2초"]:
            self.assertContains(res, text)

    def test_errors(self):
        cases = {"영상을 선택하세요": {}, "MP4 · MOV · AVI": {"video": video("memo.txt")},
                 "빈 파일": {"video": video(content=b"")}}
        for message, data in cases.items():
            with self.subTest(message):
                res = self.client.post("/", data)
                self.assertEqual(res.status_code, 200)
                self.assertContains(res, message)
        self.assertFalse(AnalysisJob.objects.exists())