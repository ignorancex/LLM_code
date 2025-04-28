from manim import *

import os
import pickle
import numpy as np
from numpy.linalg import norm
from math import atan2

from functools import lru_cache

from plotter import manim_enc_dir
from encs.manim.scenes.text_scenes import *


# Plot definitions
s_color  = WHITE
i1_color = XKCD.CORNFLOWERBLUE
i2_color = XKCD.BRICKRED
i3_color = XKCD.DARKLILAC
i4_color = GOLD

static_scale = 12.0
dynamic_scale = 12.0


# CMS Open Data
with open(os.path.join(manim_enc_dir/"bullseye_3part",
                   'od_img_dict_3part.pkl'), 'rb') as f:
    od3_dict = pickle.load(f)
with open(os.path.join(manim_enc_dir/"bullseye_4part",
                   'od_img_dict_4part.pkl'), 'rb') as f:
    od4_dict = pickle.load(f)
for key, val in od4_dict.copy().items():
    if val is None:
        od4_dict.pop(key)

def re3c_img(R1, re3c_dict):
    """Find the nearest R1 key in the re3c_dict."""
    nearest_R1 = min(re3c_dict.keys(), key=lambda k: abs(k - R1))
    return re3c_dict[nearest_R1]

empty_bullseye_file = re3c_img(0, od3_dict)

def re4c_img(R1, R2, phi2, re4c_dict):
    """Find the nearest (R1, R2, phi2) keys in the re4c_dict."""
    nearest_keys = min(re4c_dict.keys(),
                       key=lambda k: (abs(k[0] - R1)
                                      + abs(k[1] - R2)
                                      + abs(k[2] - phi2)))
    return re4c_dict[nearest_keys]


class RENC_Intro(ThreeDScene):
    # ==================================
    # Intro Slide
    # ==================================
    def introslide(self):
        # Render the intro animations
        title = Text("Resolved Energy Correlators", font_size=64)
        self.play(Write(title))
        self.wait(1)
        # Transition to next animations
        self.play(Unwrite(title))


    # ==================================
    # Jet Cartoon
    # ==================================
    def jetcartoon(self):
        # --------------------------------------
        # 0. Set up camera in a weird way
        # --------------------------------------
        self.move_camera(phi=-90*DEGREES, theta=-90*DEGREES,
                         run_time=2)  # Transition to top-down view

        # --------------------------------------
        # 1. Quark represented by a moving dot
        # --------------------------------------
        quark_text = Text("Toy \"Jet\" Introducing Coordinates")
        quark_text.to_corner(UL)
        self.add_fixed_in_frame_mobjects(quark_text)

        quark_start = OUT * 3
        quark_end   = IN  * 2
        q_pnt3d = Dot3D(color=s_color).move_to(quark_start)
        quark_path = Line(quark_start, quark_end, color=s_color)

        # --------------------------------------
        # 2. Arrow representing quark
        # --------------------------------------
        quark_arrow = Arrow3D(start=quark_start,
                              end=quark_end,
                              color=s_color)

        self.play(FadeIn(q_pnt3d), Write(quark_text))
        self.play(Create(quark_arrow),
                  run_time=1)
        self.wait(1)

        # --------------------------------------
        # 3. Emit 3 gluon lines
        # --------------------------------------
        gluon1_phi, gluon1_eta = 3.5,  0.0
        gluon2_phi, gluon2_eta = -1.8, -1.5
        gluon3_phi, gluon3_eta = -0.9, +0.5
        gluon1_end = quark_end + RIGHT*gluon1_phi + UP*gluon1_eta
        gluon2_end = quark_end + RIGHT*gluon2_phi + UP*gluon2_eta
        gluon3_end = quark_end + RIGHT*gluon3_phi + UP*gluon3_eta

        # Create spring-like gluon lines using
        # ParametricFunction (for a spiral effect)
        def point_along_arrow(t):
            return quark_arrow.get_start() + \
                t*(quark_arrow.get_end() -
                   quark_arrow.get_start())

        def spring_curve_func(t, start, end):
            vec = end-start

            perp1 = np.cross((1,0,0), vec)
            perp1 = perp1/norm(perp1)
            perp2 = np.cross(perp1, vec)
            perp2 = perp2/norm(perp2)

            phi = 20*t*norm(vec)
            perp_loc = 0.4 * (perp1*np.sin(phi) + perp2*np.cos(phi))

            return start + (end - start) * t + perp_loc

        g1_line = ParametricFunction(
            lambda t: spring_curve_func(t,
                            quark_arrow.get_start(),
                            gluon1_end),
            t_range=[0, 1], color=i1_color)
        g2_line = ParametricFunction(
            lambda t: spring_curve_func(t,
                            point_along_arrow(0.4),
                            gluon2_end),
            t_range=[0, 1], color=i2_color)
        gluon3 = ParametricFunction(
            lambda t: spring_curve_func(t,
                            point_along_arrow(0.6),
                            gluon3_end),
            t_range=[0, 1], color=i3_color)


        # Play the gluon emission animation
        self.play(MoveAlongPath(q_pnt3d, quark_path,
                                run_time=1),
                  AnimationGroup(
                      Create(g1_line),
                      Create(g2_line),
                      Create(gluon3),
                      lag_ratio=0.2,
                      rate_func=smooth
                  ))

        # ---------------------------------
        # 4. Map to the eta-phi plane
        # ---------------------------------
        self.play(FadeOut(quark_text), run_time=0.5)
        self.move_camera(phi=0*DEGREES, theta=-135*DEGREES,
                         run_time=2)

        # Create the points associated with each particle
        q_point  = Dot(color=s_color).move_to(quark_end)
        g1_point = Dot(color=i1_color).move_to(gluon1_end)
        g2_point = Dot(color=i2_color).move_to(gluon2_end)
        g3_point = Dot(color=i3_color).move_to(gluon3_end)

        # Create the eta-phi plane (Axes)
        axes = Axes(
            x_range=[-3, 3],
            y_range=[-3, 3],
            axis_config={"color": GREY_BROWN},
        )
        ylabel1 = Text("Azimuthal Angle")
        ylabel2 = Tex(r"$(\phi)$")
        xlabel1 = Text("Rapidity")
        xlabel2 = Tex(r"$(\eta)$")
        ylabel1.move_to(RIGHT*2.5+UP*2.5)
        ylabel2.move_to(RIGHT*2.8+UP*2.2)
        xlabel1.move_to(RIGHT*3.0+DOWN*2.8)
        xlabel2.move_to(RIGHT*2.8+DOWN*3.2)

        axes.rotate(-PI/4, axis=OUT)
        xlabel1.rotate(-PI/4, axis=OUT)
        xlabel2.rotate(-PI/4, axis=OUT)
        ylabel1.rotate(-PI/4, axis=OUT)
        ylabel2.rotate(-PI/4, axis=OUT)

        # self.play(Create(axes))
        self.play(Create(axes),
                  Write(xlabel1), Write(ylabel1))
                  # Write(xlabel2), Write(ylabel2),)

        # Animate their appearance on the plane
        self.play(FadeOut(quark_path),
                  ReplacementTransform(q_pnt3d, q_point),
                  ReplacementTransform(g1_line, g1_point),
                  ReplacementTransform(g2_line, g2_point),
                  ReplacementTransform(gluon3, g3_point),
                  run_time=1)
        self.play(FadeOut(axes), FadeOut(quark_arrow),
                  FadeOut(xlabel1), FadeOut(ylabel1),)
                  # FadeOut(xlabel2), FadeOut(ylabel2))

        # Compute the angle between g1_point and the x-axis
        self.move_camera(phi=0*DEGREES, theta=-90*DEGREES,
                         run_time=2)

        # Remove axes and draw R1, R2, R3
        R1_line = always_redraw(lambda:
                        Line(q_point.get_center(),
                             g1_point.get_center(),
                             color=i1_color))
        R2_line = always_redraw(lambda:
                        Line(q_point.get_center(),
                             g2_point.get_center(),
                             color=i2_color))
        R3_line = always_redraw(lambda:
                        Line(q_point.get_center(),
                             g3_point.get_center(),
                             color=i3_color))

        # Create the labels R_1, R_2, and R_3
        R1_label = Tex(r"$R_1$", color=i1_color).\
                            next_to(R1_line, UP, buff=0.1)
        R2_label = Tex(r"$R_2$", color=i2_color).\
                            next_to(R2_line, buff=0.1).\
                            shift(LEFT*0.8+DOWN*0.2)
        R3_label = Tex(r"$R_3$", color=i3_color).\
                            next_to(R3_line, buff=0.1).\
                            shift(LEFT*0.45+UP*0.2)

        ordered_text = Tex(r"$R_1 > R_2 > R_3 > \cdots$").\
                            move_to(DOWN*3.5)

        # Play animations to create the lines and labels
        self.play(Create(R1_line), Create(R2_line),
                  Create(R3_line))
        self.play(Write(R1_label), Write(R2_label),
                  Write(R3_label))

        self.play(Write(ordered_text))
        self.wait(1)
        self.play(FadeOut(ordered_text))

        # Save particles
        self.cartoon_axes = axes
        self.particles = [q_point, g1_point,
                          g2_point, g3_point]
        self.R_lines = [None, R1_line, R2_line, R3_line]

        # --------------------------------------
        # Move the camera to the right
        # --------------------------------------
        # Preparing to show pdfs of CMS open data
        self.move_camera(frame_center=RIGHT*6, zoom=0.8,
                         run_time=2)

        self.play(Unwrite(R1_label), Unwrite(R2_label),
                  Unwrite(R3_label))
        self.play(FadeOut(g3_point), FadeOut(R3_line),)


    def static_re3c(self):
        # Get existing points
        q_point = self.particles[0]
        g1_point = self.particles[1]
        g2_point = self.particles[2]

        # Setting up image
        dist = norm(q_point.get_center()-g1_point.get_center())
        image_path = re3c_img(dist/static_scale, od3_dict)
        self.bullseye = ImageMobject(image_path).\
                                move_to(RIGHT*18).set(width=8)

        # Add the static image to the scene
        self.play(FadeIn(self.bullseye))
        STATIC_CENTER = 9.8

        # Set up the particle (g2_right) and the circular path
        right_rad = 2
        unit_start = g2_point.get_center() - q_point.get_center()
        unit_start = unit_start / norm(unit_start)

        g2_right = Dot(color=i2_color).move_to(RIGHT*STATIC_CENTER
                                               +right_rad*unit_start)
        g2_right.z_index=1

        # Simultaneously circular motion for g2_right, g2_point
        self.play(
            Transform(g2_point, g2_point,
                      path_func=utils.paths.path_along_circles(
                          2*PI, q_point.get_center()),
                      ),
            Transform(g2_right, g2_right,
                      path_func=utils.paths.path_along_circles(
                          2*PI, RIGHT*STATIC_CENTER),
                      ),
            run_time=2
        )

        self.wait(2)  # Keep the image displayed

        # Clean up
        self.play(FadeOut(g2_right))


    def dynamic_re3c(self, img_dict):
        q_point = self.particles[0]
        g1_point = self.particles[1]

        # Create a ValueTracker for R1
        R1_tracker = ValueTracker(0)

        # Define the always_redraw ImageMobject
        empty_bullseye = ImageMobject(empty_bullseye_file).\
                            move_to(RIGHT*18).set(width=8)
        self.play(FadeOut(self.bullseye), FadeIn(empty_bullseye))

        self.remove(self.bullseye)
        self.bullseye = always_redraw(lambda: ImageMobject(
            re3c_img(R1_tracker.get_value(), od3_dict)
        ).move_to(RIGHT*18).set(width=8))

        self.remove(empty_bullseye)
        self.add(self.bullseye)

        # Update method to adjust g1_point and update R1_tracker
        def update_radius(mobject):
            dist = norm(q_point.get_center() - mobject.get_center())
            R1_tracker.set_value(dist / dynamic_scale)

        # Animation 1: Move g1_point quickly to the origin
        path0 = Line(g1_point.get_center(), q_point.get_center())
        self.play(MoveAlongPath(g1_point, path0),
                  run_time=1.5, rate_func=smooth)
        update_radius(g1_point)

        # Animation 2: Move g1_point slowly outward
        g1_point.add_updater(update_radius)
        path1 = Line(q_point.get_center(),
                     q_point.get_center() + RIGHT * 3.5)
        self.play(MoveAlongPath(g1_point, path1),
                  run_time=15, rate_func=smooth)

        # Remove the updater after the animation completes
        g1_point.remove_updater(update_radius)


    def dynamic_re4c(self, re4c_dict):
        q_point = self.particles[0]
        g1_point = self.particles[1]
        g2_point = self.particles[2]

        # Create ValueTrackers for R1, R2, and phi2
        R1_tracker = ValueTracker(0)
        R2_tracker = ValueTracker(0)
        phi2_tracker = ValueTracker(0)

        # Set R1 and R2 to constants
        R1_const  = norm(q_point.get_center()-g1_point.get_center())
        R1_const /= dynamic_scale
        R2_const  = 0.95*R1_const  # Slightly smaller than R1_const

        # Position g2_point at a fixed distance R2_const from q_point
        # Initially aligned with g1_point (phi2 = 0)
        g2_point.move_to(q_point.get_center()
                         + R2_const*RIGHT*dynamic_scale)
        self.R_lines[2].put_start_and_end_on(q_point.get_center(),
                                             g1_point.get_center())

        # Update R1_tracker and R2_tracker to the constants
        R1_tracker.set_value(R1_const)
        R2_tracker.set_value(R2_const)

        # Define the always_redraw ImageMobject for RE4C
        self.bullseye = always_redraw(lambda: ImageMobject(
            re4c_img(
                R1_tracker.get_value(),
                R2_tracker.get_value(),
                phi2_tracker.get_value(),
                re4c_dict
            )
        ).move_to(RIGHT*18).set(width=8))

        self.play(FadeIn(g2_point), FadeIn(self.R_lines[2]),
                  FadeIn(self.bullseye))

        def angle_between_vectors(v1, v2):
            angle = atan2(v2[1], v2[0]) - atan2(v1[1], v1[0])
            while angle >= PI:
                angle -= 2*PI
            while angle < -PI:
                angle += 2*PI
            return angle

        # Define an updater for phi2 based on the position of g2_point
        def update_g2(mobject):
            # Vector from q_point to g1_point
            vec1 = g1_point.get_center() - q_point.get_center()
            # Vector from q_point to g2_point
            vec2 = g2_point.get_center() - q_point.get_center()

            # Compute the distance R2
            R2 = norm(q_point.get_center()-g2_point.get_center())
            R2 /= dynamic_scale

            # Compute the angle between vec1 and vec2
            angle = angle_between_vectors(vec1, vec2)

            R2_tracker.set_value(R2)
            phi2_tracker.set_value(angle)

        # Add the updater to g2_point
        g2_point.add_updater(update_g2)

        # Animation 1: R1 and R2 constant, phi2 varies
        # -------------------------------------------------
        # Define a circular path for g2_point around q_point
        circle_radius = R1_const * dynamic_scale
        circle_center = q_point.get_center()

        # Create a ParametricFunction for the path
        def parametric_circle(t):
            return circle_center + circle_radius * np.array([
                np.cos(t),
                -np.sin(t),
                0
            ])

        # Animate g1_point moving around the circle
        self.play(
            MoveAlongPath(g1_point, ParametricFunction(
                parametric_circle, t_range=[0, TAU]
            )),
            run_time=15,
            rate_func=smooth
        )

        self.wait(1)

        # Animation 2: R1 constant, R2 and phi2 spiral inward
        # -------------------------------------------------
        # Animate the spiral inward movement
        self.play(
            MoveAlongPath(g1_point, ParametricFunction(
                parametric_circle, t_range=[0, TAU]
            )),
            MoveAlongPath(g2_point, Line(g2_point, q_point)),
            run_time=15,
            rate_func=smooth
        )

        # Remove the updater after the animation completes
        g2_point.remove_updater(update_g2)


    def opendata_re3c(self):
        # RE3C: Static
        re3c = Tex(
            r"$\text{RE3C} \propto \langle E_1 E_2 E_3 \rangle$")
        re3c.move_to(DOWN*3.5+RIGHT+1.0)
        self.play(Write(re3c))
        self.static_re3c()

        # RE3C: Dynamic
        self.play(FadeOut(self.particles[2]),
                  FadeOut(self.R_lines[2]))

        self.dynamic_re3c(od3_dict)
        self.wait()
        self.play(FadeOut(re3c), FadeOut(self.bullseye))


    def opendata_re4c(self):
        re4c = Tex(
            r"$\text{RE4C} \propto \langle E_1 E_2 E_3 E_4 \rangle$")
        re4c.move_to(DOWN*3.5+RIGHT+1.0)

        self.play(Write(re4c))
        self.dynamic_re4c(od4_dict)
        self.wait()

        self.play(FadeOut(re4c), FadeOut(self.bullseye))
        self.wait(2)


    def final(self):
        q_point = self.particles[0]
        g2_point = self.particles[2]
        g3_point = self.particles[3]

        g2_point_start = -1.8*RIGHT - 1.5*UP

        self.R_lines[2].put_start_and_end_on(q_point.get_center(),
                                             g2_point.get_center())

        self.play(
            MoveAlongPath(g2_point, Line(g2_point.get_center(),
                                         g2_point_start)),
            FadeIn(g3_point), FadeIn(self.R_lines[3])
        )

        # Create the labels R_1, R_2, and R_3
        R1_label = Tex(r"$R_1$", color=i1_color).\
                            next_to(self.R_lines[1], UP, buff=0.1)
        R2_label = Tex(r"$R_2$", color=i2_color).\
                            next_to(self.R_lines[2], buff=0.1).\
                            shift(LEFT*0.8+DOWN*0.2)
        R3_label = Tex(r"$R_3$", color=i3_color).\
                            next_to(self.R_lines[3], buff=0.1).\
                            shift(LEFT*0.45+UP*0.2)

        ordered_text = Tex(r"$R_1 > R_2 > R_3 > \cdots$").\
                            move_to(UP*2.5)

        # Play animations to create the lines and labels
        self.play(Write(R1_label), Write(R2_label),
                  Write(R3_label))

        self.move_camera(frame_center=DOWN, zoom=1.0, run_time=2)

        self.play(Write(ordered_text))

        title = Text("Resolved Energy Correlators", font_size=50).\
            shift(3.5*DOWN)
        self.play(Write(title))

        self.wait(3)


    def fade_all(self, unwrite=True):
        animations = []

        for mob in self.mobjects:
            if unwrite and isinstance(mob, (Text, Tex)):
                animations.append(Unwrite(mob))
            else:
                animations.append(FadeOut(mob))

        self.play(*animations)


    def construct(self):
        self.bullseye = None

        # Intro Slide
        self.next_section("Intro")
        self.introslide()

        # Jet cartoon
        self.next_section("Jet Cartoon")
        self.jetcartoon()

        # Open Data visualization
        self.next_section("OpenData RE3C")
        title = Text("CMS Open Data:",
                     font_size=72).move_to(RIGHT*6.5+UP*4.0)
        self.play(Write(title))

        # RE3C
        self.opendata_re3c()

        # RE4C
        self.next_section("OpenData RE4C")
        self.opendata_re4c()

        self.play(Unwrite(title))

        # End "slide"
        self.next_section("Final")
        self.final()
        self.fade_all()
        self.wait()
