# pylint: disable=C0103
"""URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.urls import path
from qlpdb.experiment.views import ExperimentView, ExperimentSummaryView

app_name = "experiment"
urlpatterns = [
    path("summary/", ExperimentSummaryView.as_view(), name="Experiment Summary"),
    path("detail/<int:pk>/", ExperimentView.as_view(), name="Experiment Detail"),
]
