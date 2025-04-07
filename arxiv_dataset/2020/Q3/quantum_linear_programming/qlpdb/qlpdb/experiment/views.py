from django.views.generic import DetailView
from django.views.generic.list import ListView

from qlpdb.experiment.models import Experiment

# Create your views here.


class ExperimentView(DetailView):
    """View for visualizing experiments
    """

    template_name = "experiments.html"
    model = Experiment

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        experiment = context["experiment"]  # this is the instance of experiment 1
        # experiment.objects.filter()
        return context


### Summary table
### Graph: tag,
### experiment table of parameters sorted by lowest eigenvalue and
## number of data which has eigenvalue and if constraints are violated.


class ExperimentSummaryView(ListView):
    """
    """

    template_name = "experiment-summary.html"
    model = Experiment
    ordering = ["-tag"]
