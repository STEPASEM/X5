from django.urls import path

from . import views

app_name = 'search'

urlpatterns = [
    path('', views.search, name='sea'),
    path('api/predict', views.api_predict, name='api_predict'),
    path('health', views.health_check, name='health_check'),
]