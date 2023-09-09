from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
     path("camera/", views.lives, name="live_camera"),
     path("execute_function/", views.text_to_speech, name="sound")
]