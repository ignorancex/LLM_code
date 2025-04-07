"""Models of data
"""

# Note: if you want your models to use espressodb features, they must inherit from Base

from django.db import models
from django.core.validators import MaxValueValidator, MinValueValidator
from espressodb.base.models import Base
from django.contrib.postgres.fields import ArrayField

class Data(Base):
    experiment = models.ForeignKey(
        "experiment.Experiment",
        on_delete=models.CASCADE,
        help_text=r"Foreign Key to `experiment`",
    )
    measurement = models.PositiveIntegerField(
        null=False,
        help_text="Increasing integer field labeling measurement number"
    )
    spin_config = ArrayField(
        models.PositiveSmallIntegerField(
            validators=[
                MaxValueValidator(1),
                MinValueValidator(0)
            ]
        ),
        help_text="Spin configuration of solution, limited to 0, 1"
    )
    chain_break_fraction = models.FloatField(
        null=False,
        help_text="Chain break fraction"
    )
    energy = models.FloatField(
        null=False,
        help_text="Energy corresponding to spin_config and QUBO"
    )
    constraint_satisfaction = models.BooleanField(
        null=False,
        help_text="Are the inequality constraints satisfied by the slacks?"
    )
    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["experiment", "measurement"], name="unique_data"
            )
        ]