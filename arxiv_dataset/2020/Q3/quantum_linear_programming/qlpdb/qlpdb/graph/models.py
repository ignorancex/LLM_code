"""Models of graph
"""

# Note: if you want your models to use espressodb features, they must inherit from Base

from django.db import models
from espressodb.base.models import Base
from django.contrib.postgres.fields import ArrayField


class Graph(Base):
    tag = models.TextField(
        null=False,
        blank=False,
        help_text="Tag for graph type (e.g. Hamming(n,m) or K(n,m))",
    )
    total_vertices = models.PositiveSmallIntegerField(
        null=False, help_text="Total number of vertices in graph"
    )
    total_edges = models.PositiveSmallIntegerField(
        null=False, help_text="Total number of edges in graph"
    )
    max_edges = models.PositiveSmallIntegerField(
        null=False, help_text="Maximum number of edges per vertex"
    )
    adjacency = ArrayField(
        ArrayField(models.PositiveSmallIntegerField(null=False), size=2),
        help_text="Sorted adjacency matrix of dimension [N, 2]"
    )
    adjacency_hash = models.TextField(
        null=False,
        blank=False,
        help_text="md5 hash of adjacency list used for unique constraint"
    )
    class Meta:
        constraints = [
            models.UniqueConstraint(
                fields=["adjacency_hash", "tag"], name="unique_graph"
            )
        ]
