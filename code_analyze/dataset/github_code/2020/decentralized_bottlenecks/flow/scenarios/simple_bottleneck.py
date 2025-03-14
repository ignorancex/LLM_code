"""Contains the bottleneck scenario class."""

from flow.core.params import InitialConfig
from flow.core.params import TrafficLightParams
from flow.scenarios.base_scenario import Scenario
import numpy as np

ADDITIONAL_NET_PARAMS = {
    # the factor multiplying number of lanes.
    "scaling": 1,
    # edge speed limit
    'speed_limit': 23
}


class SimpleBottleneckScenario(Scenario):
    """Scenario class for bottleneck simulations.

    This network acts as a scalable representation of the Bay Bridge. It
    consists of a two-stage lane-drop bottleneck where 4n lanes reduce to 2n
    and then to n, where n is the scaling value. The length of the bottleneck
    is fixed.

    Requires from net_params:

    * **scaling** : the factor multiplying number of lanes
    * **speed_limit** : edge speed limit

    In order for right-of-way dynamics to take place at the intersection,
    set *no_internal_links* in net_params to False.

    Usage
    -----
    >>> from flow.core.params import NetParams
    >>> from flow.core.params import VehicleParams
    >>> from flow.core.params import InitialConfig
    >>> from flow.scenarios import BottleneckScenario
    >>>
    >>> scenario = BottleneckScenario(
    >>>     name='bottleneck',
    >>>     vehicles=VehicleParams(),
    >>>     net_params=NetParams(
    >>>         additional_params={
    >>>             'scaling': 1,
    >>>             'speed_limit': 1,
    >>>         },
    >>>         no_internal_links=False  # we want junctions
    >>>     )
    >>> )
    """

    def __init__(self,
                 name,
                 vehicles,
                 net_params,
                 initial_config=InitialConfig(),
                 traffic_lights=TrafficLightParams()):
        """Instantiate the scenario class."""
        for p in ADDITIONAL_NET_PARAMS.keys():
            if p not in net_params.additional_params:
                raise KeyError('Network parameter "{}" not supplied'.format(p))

        super().__init__(name, vehicles, net_params, initial_config,
                         traffic_lights)

    def specify_nodes(self, net_params):
        """See parent class."""
        nodes = [
            {
                "id": "1",
                "x": 0,
                "y": 0
            },  # pre-toll
            {
                "id": "2",
                "x": 100,
                "y": 0
            },  # toll

        ]  # post-merge2
        return nodes

    def specify_edges(self, net_params):
        """See parent class."""
        scaling = net_params.additional_params.get("scaling", 1)
        speed = net_params.additional_params['speed_limit']
        assert (isinstance(scaling, int)), "Scaling must be an int"

        edges = [
            {
                "id": "1",
                "from": "1",
                "to": "2",
                "length": 100,
                "spreadType": "center",
                "numLanes": 2 * scaling,
                "speed": speed
            },
        ]

        return edges

    def specify_connections(self, net_params):
        """See parent class."""
        conn_dic = {}
        conn = []
        for i in range(2):
            conn += [{
                "from": "1",
                "to": "2",
                "fromLane": i,
                "toLane": int(np.floor(i / 2))
            }]
        conn_dic["2"] = conn
        return conn_dic

    def specify_routes(self, net_params):
        """See parent class."""
        rts = {
            "1": ["1", "2"],
            "2": ["2"],
        }

        return rts

    def specify_edge_starts(self):
        """See parent class."""
        return [("1", 0), ("2", 100)]
