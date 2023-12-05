from django.urls import path
from .views import *


urlpatterns = [
    path('', menu),
    path('fir', fir),
    path('sec', sec),
    path('sis', sis)


]

