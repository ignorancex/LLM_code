
# external imports
from __future__ import annotations
import pickle
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi
import numpy as np
import math

# internal imports
from .World import *
from .Agent import *


class Experiment:
    def __init__(self, name : str = "", domain : Domain = Domain()) -> None:
        self._vc = []                                                           # Voronoi centers
        self._voronoi = None                                                    # Voronoi object
        self._world : World = World()                                           # world object
        self._agents : List[Agent] = []                                         # agent object
        self._homogeneous_agents = False                                        # all agents have the same sensor model
        self._domain = domain                                                   # domain object
        self._name = name                                                       # name of the experiment

    # getters
    def world(self) -> World:
        return self._world

    def agent(self, idx : int = 0) -> Agent:
        if idx >= len(self._agents) or idx < 0:
            raise IndexError("Agent index out of bounds.")
        return self._agents[idx]
        
    def agents(self) -> List[Agent]:
        return self._agents

    def randomRegion(self) -> Region:
        idx = np.random.randint(0, self._world.nRegions())
        return self._world.regions()[idx]

    def voronoi(self) -> Voronoi:
        return self._voronoi
    
    def nAgents(self) -> int:
        return len(self._agents)
    
    def nTargets(self) -> int:
        return self._world.nTargets()
    
    def nRegions(self) -> int:
        return self._world.nRegions()

    # modifiers
    def addRandomVoronoiPoints(self, M : int, min_dist=0.0) -> None:
        self._vc = []
        counter = 0
        
        if (M < 0):
            raise ValueError("Number of Voronoi points must be a nonnegative number.")
        
        while len(self._vc) < M:

            # prevent infinite loop
            counter += 1
            if counter > 1000 * M:
                raise Exception(
                    "Could not generate enough Voronoi points. " + 
                    "Try decreasing the minimum distance between points."
                )

            # sample new point
            x = np.random.uniform(self._domain.xmin(), self._domain.xmax())
            y = np.random.uniform(self._domain.ymin(), self._domain.ymax())
            p = np.array([x, y])
            
            # always add first point
            if len(self._vc) == 0:
                self._vc.append(p)
                continue
            
            # check if point is too close to existing points
            dist = np.linalg.norm(self._vc - p, axis=1)
            if np.min(dist) < min_dist:
                continue

            self._vc.append(p)

        self._M = len(self._vc)
        self._vc = np.array(self._vc)

    def generatePartitioning(self, n_obstacles=0) -> None:
        
        if self._vc.shape[0] > 1:
            self._voronoi = Voronoi(self._vc)

        regions = []
        for i in range(self._M):
            g = {}
            b = {}
            for j in range(self._M):
                if i == j:
                    continue

                a = self._vc[j] - self._vc[i]
                a = a / np.linalg.norm(a)
                g[j] = a 
                b[j] = a @ (self._vc[i] + self._vc[j]) / 2
            
            if i < self._M-n_obstacles:
                dyn = ConstantDynamics(2,0,0,np.random.uniform(-0.5,0.5,2))
                regions.append(ConstantDCPRegion(
                    g,
                    b,
                    self._vc[i], 
                    domain=self._domain, 
                    dynamics=dyn)
                    )      
            else:
                regions.append(ObstacleCPRegion(
                    g,
                    b,
                    self._vc[i], 
                    domain=self._domain)
                    )
        self._world.setRegions(regions)
    
    def addAgent(self, agent : Agent) -> None:
        if not isinstance(agent, Agent):
            raise ValueError("Argument must be of type Agent.")
        self._agents.append(agent)

    def addRandomAgent(
            self, 
            gpp : GlobalPathPlanner = None,
            sensor : Sensor = None,
            name : str = ""
            ) -> None:
        if sensor is None:
            sensor = Sensor()
            for target in self._world.targets():
                if target.name == '3':
                    sensor.setTargetQualityFunction(
                        target, 
                        SinusoidalQualityFunction(
                            c1=np.random.uniform(3,20),
                            c2=np.random.uniform(3,20)
                            )
                        )
                else:
                    sensor.setTargetQualityFunction(
                        target, 
                        GaussianQualityFunction()
                        )

                sensor.setNoiseMatrix(target, np.eye(1))
                sensor.setMeasurementMatrix(target, np.eye(1))
        self.addAgent(Agent(self._world, sensor=sensor, gpp=gpp, name=name))

    def addRandomTargets(
            self, 
            n : int = None, 
            fraction : float = 0.5,
            min_dist_to_boundary : float = 0.005
            ) -> None:
        target_counter = 0
        if fraction < 0 or float(fraction) > 1:
            raise ValueError("Fraction must be in [0,1].")
        if n is None:
            if fraction is None:
                raise ValueError("Either n or fraction must be specified.")
            n = self._world.nRegions() * fraction
        n = math.floor(n)

        if n > self._world.nRegions() - self._world.nObstacles():
            raise ValueError("Number of targets exceeds number of regions.")

        for region in self._world.regions():
            if target_counter >= n:
                break
           
            if region.isObstacle():
                continue

            cntr = 0
            while True:
                pos = region.randomPoint()
                if region.distToBoundary(pos) > min_dist_to_boundary:
                    break
                cntr += 1
                if cntr > 1000:
                    raise Exception("Could not add target. Try decreasing minimum distance to boundary.")
            phi0 = np.array([1.0])
            Q = np.array([0.8])
            A = np.array([0.001])
            target = Target(pos=pos, region=region, phi0=phi0, Q=Q, A=A)
            target.name = str(target_counter+1)
            self.addTarget(target)
            target_counter += 1

        if target_counter < n:
            raise Exception("Could not add all targets.")

    def addTarget(self, target : Target) -> None:
        if not isinstance(target, Target):
            raise ValueError("Argument must be of type Target.")
        self._world.addTarget(target)
        self._M += 1

    # plotters
    def plotWorld(
            self, 
            with_sensor_quality=False, 
            add_target_labels=True, 
            fill_empty_regions=True,
            plot_partition=True,
            plot_targets=True,
            plot_vector_field=True
            ) -> Tuple[plt.Figure, plt.Axes]:
        fig, ax = plt.subplots()
        ax.set_aspect('equal', 'box')
        fig.tight_layout()
        ax.axis('off')
        ax.set_xlim(self._domain.xmin()*1.01, self._domain.xmax()*1.01)
        ax.set_ylim(self._domain.ymin()*1.01, self._domain.ymax()*1.01)

        if with_sensor_quality and len(self._agents) > 0:
            if len(self._agents) == 1 or self._homogeneous_agents:
                self.agent(0).plotSensorQuality(ax=ax)
            else:
                warnings.warn("Adding the sensor quality tot he world plot is only supported for a single agent.")

        self._world.plotMissionSpace(
            ax=ax, 
            add_target_labels=add_target_labels, 
            fill_empty_regions=fill_empty_regions,
            plot_partition=plot_partition,
            plot_targets=plot_targets,
            plot_vector_field=plot_vector_field
            )

        return fig, ax

    def zoomToTargetRegion(self, ax : plt.Axes, name : str):
        target = self._world.getTargetByName(name)
        region = target.region()
        xrange = [np.inf, -np.inf]
        yrange = [np.inf, -np.inf]
        i = 0
        while i < 100:
            i += 1
            p = region.randomBoundaryPoint()
            xrange[0] = min(xrange[0], p[0])
            xrange[1] = max(xrange[1], p[0])
            yrange[0] = min(yrange[0], p[1])
            yrange[1] = max(yrange[1], p[1])

        ax.set_xlim(xrange[0] - 0.01, xrange[1] + 0.01)
        ax.set_ylim(yrange[0] - 0.01, yrange[1] + 0.01)

    def serialize(self, filename : str) -> None:
        if not str.endswith(filename, ".pkl") and not str.endswith(filename, ".pickle"):
            raise ValueError("File must be have a pickle file extension (.pkl or .pickle).")
        plt.close()
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "wb") as f:
            pickle.dump(self, f)
    
    # static methods
    @staticmethod
    def deserialize(filename : str):
        if not str.endswith(filename, ".pkl") and not str.endswith(filename, ".pickle"):
            raise ValueError("File must be have a pickle file extension (.pkl or .pickle).")
        if not os.path.exists(filename):
            return None
        with open(filename, "rb") as f:
            return pickle.load(f)
    
    @staticmethod
    def generate(
            n_sets=15, 
            fraction=0.5, 
            seed=None, 
            min_dist=0.0,
            n_agents=1,
            homogeneous_agents=True,
            n_obstacles=0,
            domain=Domain()
            ) -> Experiment:
        '''
        generate: Generate a random experiment.
        
        Parameters:
        n_sets: int
            Number of sets for the partition.
        fraction: float
            Fraction of regions that will contain targets.
        seed: int
            Seed for the random number generator.
        min_dist: float
            Minimum distance between Voronoi points.
        n_agents: int
            Number of agents (multi-agent scenarios are experimental!).
        homogenoeous_agents: bool
            If True, all agents will have the same sensor model.
        '''
        if seed is not None:
            np.random.seed(seed)
        try:
            ex = Experiment(domain=domain)
            ex.addRandomVoronoiPoints(n_sets, min_dist=min_dist)
            ex.generatePartitioning(n_obstacles)
            ex.addRandomTargets(fraction=fraction)
            gpp = GlobalPathPlanner(ex.world())
            sensor = None
            ex._homogeneous_agents = homogeneous_agents or n_agents == 1
            for i in range(n_agents):
                ex.addRandomAgent(gpp=gpp, sensor=sensor, name=str(i))
                if homogeneous_agents:
                    sensor = ex.agent().sensor()
            return ex
        except Exception as e:  
            print(e)
            return None
