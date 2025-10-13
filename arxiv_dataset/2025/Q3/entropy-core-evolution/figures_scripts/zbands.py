import attr
import numpy as np
from scipy.optimize import bisect
from astropy.cosmology import Planck18


@attr.s(auto_attribs=True)
class RedshiftBand:
    name: str
    z_min: float
    z_max: float
    color: str
    z_label: str
    time_label: str
    label_midpoint_log: float


z_bands_group = [
    RedshiftBand(
        name="A",
        z_min=1.5,
        z_max=3.0,
        color="#003f5c",
        z_label=r"$1.5 \leq z < 3$" + "\n" + r"(9.5 $-$ 11.6) Gyr",
        time_label="",  # r'(9.5 $-$ 11.6) Gyr',
        label_midpoint_log=10 ** (np.log10(2.5 * 4.0) * 0.5),
    ),
    RedshiftBand(
        name="B",
        z_min=0.7,
        z_max=1.3,
        color="#7a5195",
        z_label=r"$0.7 \leq z < 1.3$",
        time_label=r"(6.5 $-$ 9.0) Gyr",
        label_midpoint_log=10 ** (np.log10(1.7 * 2.3) * 0.5),
    ),
    RedshiftBand(
        name="C",
        z_min=0.0,
        z_max=0.3,
        color="#ffa600",
        z_label=r"$0 \leq z < 0.3$",
        time_label=r"(0 $-$ 3.5) Gyr",
        label_midpoint_log=10 ** (np.log10(1 * 1.3) * 0.5),
    ),
]

z_bands_cluster = [
    RedshiftBand(
        name="A",
        z_min=2.2,
        z_max=4.0,
        color="#003f5c",
        z_label=r"$2.2 \leq z < 4$" + "\n" + r"(10.8 $-$ 12.3) Gyr",
        time_label="",  # r'(10.8 $-$ 12.3) Gyr',
        label_midpoint_log=10 ** (np.log10(3.2 * 5.0) * 0.5),
    ),
    RedshiftBand(
        name="B",
        z_min=1.5,
        z_max=2.0,
        color="#7a5195",
        z_label=r"$1.5 \leq z < 2$",
        time_label=r"(9.5 $-$ 10.5) Gyr",
        label_midpoint_log=10 ** (np.log10(2.5 * 3.0) * 0.5),
    ),
    RedshiftBand(
        name="C",
        z_min=0.0,
        z_max=0.3,
        color="#ffa600",
        z_label=r"$0 \leq z < 0.3$",
        time_label=r"(0 $-$ 3.5) Gyr",
        label_midpoint_log=10 ** (np.log10(1 * 1.3) * 0.5),
    ),
]


def redshift_from_lookback_time(lookback_time: float) -> float:

    def lookback_time_func(z):
        return Planck18.lookback_time(z).value - lookback_time

    return bisect(lookback_time_func, 0, 20)


import numpy as np
from bisect import bisect
from matplotlib.text import Annotation
from matplotlib.transforms import Affine2D


class LineAnnotation(Annotation):
    """A sloped annotation to *line* at position *x* with *text*
    Optionally an arrow pointing from the text to the graph at *x* can be drawn.
    Usage
    -----
    fig, ax = subplots()
    x = linspace(0, 2*pi)
    line, = ax.plot(x, sin(x))
    ax.add_artist(LineAnnotation("text", line, 1.5))
    """

    def __init__(
        self, text, line, x, xytext=(0, 5), textcoords="offset points", **kwargs
    ):
        """Annotate the point at *x* of the graph *line* with text *text*.
        By default, the text is displayed with the same rotation as the slope of the
        graph at a relative position *xytext* above it (perpendicularly above).
        An arrow pointing from the text to the annotated point *xy* can
        be added by defining *arrowprops*.
        Parameters
        ----------
        text : str
            The text of the annotation.
        line : Line2D
            Matplotlib line object to annotate
        x : float
            The point *x* to annotate. y is calculated from the points on the line.
        xytext : (float, float), default: (0, 5)
            The position *(x, y)* relative to the point *x* on the *line* to place the
            text at. The coordinate system is determined by *textcoords*.
        **kwargs
            Additional keyword arguments are passed on to `Annotation`.
        See also
        --------
        `Annotation`
        `line_annotate`
        """
        assert textcoords.startswith(
            "offset "
        ), "*textcoords* must be 'offset points' or 'offset pixels'"

        self.line = line
        self.xytext = xytext

        # Determine points of line immediately to the left and right of x
        xs, ys = line.get_data()

        assert (
            np.diff(xs) >= 0
        ).all(), "*line* must be a graph with datapoints in increasing x order"

        i = np.clip(bisect(xs, x), 1, len(xs) - 1)
        self.neighbours = n1, n2 = np.asarray([(xs[i - 1], ys[i - 1]), (xs[i], ys[i])])

        # Calculate y by interpolating neighbouring points
        y = n1[1] + ((x - n1[0]) * (n2[1] - n1[1]) / (n2[0] - n1[0]))

        kwargs = {
            "horizontalalignment": "center",
            "rotation_mode": "anchor",
            **kwargs,
        }
        super().__init__(text, (x, y), xytext=xytext, textcoords=textcoords, **kwargs)

    def get_rotation(self):
        """Determines angle of the slope of the neighbours in display coordinate system"""
        transData = self.line.get_transform()
        dx, dy = np.diff(transData.transform(self.neighbours), axis=0).squeeze()
        return np.rad2deg(np.arctan2(dy, dx))

    def update_positions(self, renderer):
        """Updates relative position of annotation text
        Note
        ----
        Called during annotation `draw` call
        """
        xytext = Affine2D().rotate_deg(self.get_rotation()).transform(self.xytext)
        self.set_position(xytext)
        super().update_positions(renderer)


def line_annotate(text, line, x, *args, **kwargs):
    """Add a sloped annotation to *line* at position *x* with *text*
    Optionally an arrow pointing from the text to the graph at *x* can be drawn.
    Usage
    -----
    x = linspace(0, 2*pi)
    line, = ax.plot(x, sin(x))
    line_annotate("sin(x)", line, 1.5)
    See also
    --------
    `plt.annotate`
    """
    ax = line.axes
    a = LineAnnotation(text, line, x, *args, **kwargs)
    if "clip_on" in kwargs:
        a.set_clip_path(ax.patch)
    ax.add_artist(a)
    return a
