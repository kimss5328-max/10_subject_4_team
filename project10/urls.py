from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('analyze/', views.analyze, name='analyze'),  # analyze.js가 영상을 보내는 주소 (POST)
]