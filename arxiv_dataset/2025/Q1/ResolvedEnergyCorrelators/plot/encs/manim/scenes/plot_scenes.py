from manim import *

class PlotFunctionGraph(Scene):
    def construct(self):
        axes = Axes(
            x_range=[-3, 3],
            y_range=[-1, 1],
            axis_config={"color": BLUE},
        )

        sin_graph = axes.plot(lambda x: np.sin(x), color=YELLOW)
        graph_label = axes.get_graph_label(sin_graph,
                                           label="\\sin(x)")

        self.play(Create(axes), Create(sin_graph), Write(graph_label))
        self.wait(2)

