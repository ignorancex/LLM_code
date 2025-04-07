# pylint: disable=C0103, R0903, R0913

"""Functions and classes which define TDSE anneal schedule setups
"""
from typing import Optional

from numpy import linspace, exp, array, ndarray
from scipy.interpolate import interp1d
from pandas import read_excel

from matplotlib.pyplot import subplots, draw, show, savefig

from qlp.mds.mds_qlpdb import AnnealOffset, find_offset


class s_to_offset:
    """Class which provides anneal curves for A(s) (original Hamiltonian coefficient),
    B(s) (final Hamiltonian coefficient) and C(s) (anneal time) over normalized
    anneal time s in GHz.

    Attributes:
        maxB: float
            Max Dwave magnet field coefficient in GHz.
        anneal_schedule: Dict[str, List[int, int]]
            Boundaries for anneal parameters.
        interpA: np.ndarray
            Interpolated curve for A(s) in GHz.
        interpB: np.ndarray
            Interpolated curve for B(s) in GHz.
        interpC: np.ndarray
            Interpolated curve for C(s).
    """

    maxB = 11.8604700

    def __init__(self, fill_value: str, anneal_curve: str):
        """Allocates offset curves A(s), B(s), C(s) over normalized anneal time s in GHz.

        Arguments:
            fill_value: If normalized time (s) is extended beyond (0, 1),
                this option decides whether to extrapolate the anneal schedule,
                or truncate it at the nearest value.
                Options are "extrapolate" or "truncate".
            anneal_curve: Curve normalized time follows.
                Options are "linear", "logistic" or "dwave" (reads in excel file).

        Raises:
            KeyError: If one of the inputs is not reckognized
        """
        if anneal_curve == "linear":

            self.anneal_schedule = {
                "s": [0, 1],
                "C (normalized)": [0, 1],
                "A(s) (GHz)": [self.maxB, 0],
                "B(s) (GHz)": [0, self.maxB],
            }
        elif anneal_curve == "logistic":
            # DWave approx 10 GHz
            s = linspace(0, 1)
            self.anneal_schedule = {
                "s": s,
                "C (normalized)": s,
                "A(s) (GHz)": self.maxB / (1.0 + exp(10.0 * (s - 0.5))),
                "B(s) (GHz)": self.maxB / (1.0 + exp(-10.0 * (s - 0.5))),
            }
        elif anneal_curve == "constantA":
            # DWave approx 10 GHz
            self.anneal_schedule = {
                "s": [0, 1],
                "C (normalized)": [0, 1],
                "A(s) (GHz)": [self.maxB, self.maxB],
                "B(s) (GHz)": [0, 0],
            }
        elif anneal_curve == "constantB":
            # DWave approx 10 GHz
            self.anneal_schedule = {
                "s": [0, 1],
                "C (normalized)": [0, 1],
                "A(s) (GHz)": [0, 0],
                "B(s) (GHz)": [self.maxB, self.maxB],
            }
        elif anneal_curve == "dwave":
            #io = "./09-1212A-B_DW_2000Q_5_anneal_schedule.xlsx"
            io = "./09-1216A-A_DW_2000Q_6_annealing_schedule.xlsx"
            print(f"anneal schedule from {io}")
            anneal_schedule = read_excel(
                io=io, sheet_name=1
            )
            self.anneal_schedule = {
                key: anneal_schedule[key].values for key in anneal_schedule.columns
            }
            #for key in self.anneal_schedule:
            #    if key in ["A(s) (GHz)", "B(s) (GHz)"]:
            #        self.anneal_schedule[key] = [round(i, 3) for i in self.anneal_schedule[key]]
        else:
            raise KeyError(f"Anneal curve {anneal_curve} not reckognized")

        if fill_value == "extrapolate":
            fill_valueA = "extrapolate"
            fill_valueB = "extrapolate"
        elif fill_value == "truncate":
            fill_valueA = (
                self.anneal_schedule["A(s) (GHz)"][0],
                self.anneal_schedule["A(s) (GHz)"][-1],
            )
            fill_valueB = (
                self.anneal_schedule["B(s) (GHz)"][0],
                self.anneal_schedule["B(s) (GHz)"][-1],
            )
        else:
            raise KeyError(f"Fill value {fill_value} not reckognized")

        paramsA = {
            "kind": "linear",
            "bounds_error": False,
            "fill_value": fill_valueA,
        }  # linear makes for more sensible extrapolation.
        paramsB = {
            "kind": "linear",
            "bounds_error": False,
            "fill_value": fill_valueB,
        }  # linear makes for more sensible extrapolation.
        paramsC = {"kind": "linear", "fill_value": "extrapolate"}
        self.interpC = interp1d(
            self.anneal_schedule["s"], self.anneal_schedule["C (normalized)"], **paramsC
        )
        self.interpA = interp1d(
            self.anneal_schedule["C (normalized)"],
            self.anneal_schedule["A(s) (GHz)"],
            **paramsA,
        )
        self.interpB = interp1d(
            self.anneal_schedule["C (normalized)"],
            self.anneal_schedule["B(s) (GHz)"],
            **paramsB,
        )

    def sanity_check(self):
        """Plots anneal schedules for A, B and C

        Sanity check: The data and interpolation should match

        Todo:
            Provide better naming for method.
        """
        # Interpolate C
        _, ax = subplots()
        x = linspace(0, 1)
        ax.errorbar(
            x=self.anneal_schedule["s"], y=self.anneal_schedule["C (normalized)"]
        )
        ax.errorbar(x=x, y=self.interpC(x))
        draw()
        show()
        # Interpolate A and B
        _, ax = subplots()
        x = linspace(-0.05, 1.05)
        ax.errorbar(
            x=self.anneal_schedule["s"],
            y=self.anneal_schedule["A(s) (GHz)"]
            / self.anneal_schedule["A(s) (GHz)"].max(),
        )
        ax.errorbar(x=x, y=self.interpA(self.interpC(x)))
        ax.errorbar(
            x=self.anneal_schedule["s"],
            y=self.anneal_schedule["B(s) (GHz)"]
            / self.anneal_schedule["B(s) (GHz)"].max(),
        )
        ax.errorbar(x=x, y=self.interpB(self.interpC(x)))
        draw()
        show()


class AnnealSchedule:
    """Class for obtaining anneal parameters for given anneal schedule parameters.
    """

    def __init__(  # pylint: disable=W0613
        self,
        offset,
        hi_for_offset,
        embedding,
        offset_min,
        offset_range,
        fill_value: str = "extrapolate",
        anneal_curve: str = "linear",
        graph_params={},
        **kwargs,
    ):
        """Initializes offset curves

        Arguments:
            offset, hi_for_offset, offset_min, offset_range: Parameters for AnnealOffset
            fill_value, anneal_curve: Parameters for s_to_offset
        """
        AO = AnnealOffset(offset, graph_params)
        self.offset_list, self.offset_tag = AO.fcn(
            hi_for_offset, offset_min, offset_range
        )
        #_, self.offset_tag, self.offset_list = find_offset(hi_for_offset, AO.fcn, embedding, offset_min, offset_range)
        print("From find offset")
        print(self.offset_list)
        self.s2o = s_to_offset(fill_value, anneal_curve)

    def C(self, s: float) -> ndarray:
        """Returns anneal time for given normalized time including offset
        """
        C = self.s2o.interpC(s)
        C_offset = C + self.offset_list
        return C_offset

    def A(self, s: float) -> ndarray:
        """Converts normalized anneal time to anneal time and returns corresponding
        value for initial Hamiltonian parameter.
        """
        C = self.C(s)
        return self.s2o.interpA(C)

    def B(self, s: float) -> ndarray:
        """Converts normalized anneal time to anneal time and returns corresponding
        value for final Hamiltonian parameter.
        """
        C = self.C(s)
        return self.s2o.interpB(C)

    def plot(
        self,
        normalized_time: float,
        ax: Optional["Axes"] = None,
        outfile: Optional[str] = "./coefficient.pdf",
        **kwargs,
    ):
        """Plots the anneal schedule for A(s) and B(s).

        Arguments:
            normalized_time: End value for s
            ax: Axes to plot in. If not given, creates new one.
            outfile: Store result to file if given.
            kwargs: For errorbar plot.
            """
        if ax is None:
            _, ax = subplots()

        X = linspace(*normalized_time)
        yA = array([self.A(Xi) for Xi in X])
        yB = array([self.B(Xi) for Xi in X])
        for qubit in range(len(yA[0])):
            ax.errorbar(x=X, y=yA[:, qubit], **kwargs)
            ax.errorbar(x=X, y=yB[:, qubit], ls="--", **kwargs)
        ax.set_xlabel("normalized time")
        ax.set_ylabel("energy/h [GHz]")

        if outfile is not None:
            savefig(outfile, bbox_inches="tight")

        draw()
        show()
        return X, yA, yB
