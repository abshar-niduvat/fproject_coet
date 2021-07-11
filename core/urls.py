from django.urls import path

from core.views import info

urlpatterns = [
    path('', info),
]