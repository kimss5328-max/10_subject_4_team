from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('project10.urls')),
]

if settings.DEBUG:  # 개발할 때만 업로드 영상을 Django가 직접 보여줌
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)