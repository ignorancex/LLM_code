import re
import argparse
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.interpolate import BSpline
import numpy as np
from math import atan2, degrees

ETS = 1  # elementary tangle size
GAP = 0.35
LINE_WIDTH= 6.0
ENDPOINT_SIZE = 80
SUBTANGLE_DISTANCE = 0.75  # distance between subtangles when adding/multiplying

ANGLE_LIMIT = 10  # if three points differ by this angle, remove the middle point
LENGTH_MIN = 3 * 1.4142


FIT_INTO_SQUARE = True

SHOW_GRID = False
SHOW_AXIS = False


class TangleExpr:
    """General expression for a tangle, either Sum or Product of tangles stored in an array"""
    def __init__(self, term1, term2):
        self.terms = (term1, term2)

    def __add__(self, other):
        return TangleSum(self, other)

    def __mul__(self, other):
        return TangleProduct(self, other)

    def __iter__(self):
        return iter(self.terms)


class TangleSum(TangleExpr):
    """ Formal sum of tangles """
    def __repr__(self):
        return f"({self.terms[0]} + {self.terms[1]})"


class TangleProduct(TangleExpr):
    """Formal product of tangles"""
    def __repr__(self):
        return f"({self.terms[0]} * {self.terms[1]})"

def integral(n):
    """Integral tangle, return a sum of 1's or -1's"""
    if -1 <= n <= 1:
        return n
    if n > 1:
        return TangleSum(1, integral(n-1))
    if n < 1:
        return TangleSum(-1, integral(n+1))


def angle_between_points(z1, z2, z3):
    """ return angle between three complex points, 0 if parallel"""
    v1 = z2 - z1
    v2 = z3 - z2
    # dot & cross product
    dot = (v1.real * v2.real) + (v1.imag * v2.imag)
    cross = (v1.real * v2.imag) - (v1.imag * v2.real)
    return degrees(atan2(cross, dot))


class ZigZag:
    """Class to draw a zig-zag line. A zig-zag line is a collection of lines, which are sequences of points."""

    def __init__(self):
        self.lines = []
        self.set_compass(0j, 0j, 0j, 0j)

    # bounding box properties
    @property
    def N(self):  # north endpoint position
        return 1j * max(self.NE.imag, self.NW.imag)

    @property
    def S(self):  # south bounding box position
        return 1j * min(self.SW.imag, self.SE.imag)

    @property
    def W(self):  # west bounding box position
        return min(self.NW.real, self.SW.real)

    @property
    def E(self):  # east bounding box position
        return max(self.NE.real, self.SE.real)

    @property
    def height(self):
        return (self.N - self.S).imag

    @property
    def width(self):
        return (self.E - self.W).real



    def bounding_box(self, compass):
        # real bounding box computed from all coords
        if compass == "N":
            return 1j * max(z.imag for line in self.lines for z in line)
        if compass == "S":
            return 1j * min(z.imag for line in self.lines for z in line)
        if compass == "W":
            return min(z.real for line in self.lines for z in line)
        if compass == "E":
            return max(z.real for line in self.lines for z in line)
        raise ValueError(f"Unknown direction {compass}")

    def __bool__(self):
        return bool(self.lines)

    def add_line(self, line: list):
        if len(line) > 1 and not all(z == line[0] for z in line):
            self.lines.append(line if isinstance(line, list) else list(line))
            self.join()

    def set_compass(self, NW, SW, SE, NE):
        self.NW, self.SW, self.SE, self.NE = NW, SW, SE, NE

    def __add__(self, other):
        # append a zig-zag to another zig-zag
        z = ZigZag()
        z.lines = self.lines + other.lines
        z.join()
        z.NW, z.SW, z.SE, z.NE = self.NW, self.SW, other.SE, other.NE
        return z

    def mirror(self):
        """ Mirror through y-axis"""
        self.lines = [[complex(-z.real, z.imag) for z in line]for line in self.lines]
        self.set_compass(
            NW=complex(-self.NE.real, self.NE.imag),
            SW=complex(-self.SE.real, self.SE.imag),
            SE=complex(-self.SW.real, self.SW.imag),
            NE=complex(-self.NW.real, self.NW.imag))

    # def set_height(self, height):
    #     """ scale whole zig-zag, so it matches height"""
    #     if height <= 0:
    #         raise ValueError(f"Can only scale with positive number, not {height}")
    #
    #     factor = height / self.height
    #
    #     self.lines = [[complex(z.real, z.imag * factor) for z in line] for line in self.lines]
    #     self.set_compass(NW=complex(self.NW.real, self.NW.imag * factor),
    #                      SW=complex(self.SW.real, self.SW.imag * factor),
    #                      SE=complex(self.SE.real, self.SE.imag * factor),
    #                      NE=complex(self.NE.real, self.NE.imag * factor))

    def rotate(self):
        """ Rotate once CCW"""
        self.lines = [[z * 1j for z in line] for line in self.lines]
        self.set_compass(
            NW=self.NE * 1j, SW=self.NW * 1j, SE=self.SW * 1j, NE=self.SE * 1j
        )

    def reflect(self):
        self.mirror()
        self.rotate()

    def move(self, dz: complex):
        self.lines = [[z + dz for z in line] for line in self.lines]
        self.set_compass(self.NW + dz, self.SW + dz, self.SE + dz, self.NE + dz)

    def join(self):
        """ If there are zig-zag lines that share an endpoint point, join them together into one zig-zag,"""
        changes = True
        while changes:
            changes = False
            length = len(self.lines)
            for i, j in combinations(range(length), 2):
                if self.lines[i][-1] == self.lines[j][0]:
                    self.lines[i] = self.lines[i] + self.lines[j][1:]
                    del self.lines[j]
                elif self.lines[i][0] == self.lines[j][-1]:
                    self.lines[i] = self.lines[j] + self.lines[i][1:]
                    del self.lines[j]
                elif self.lines[i][0] == self.lines[j][0]:
                    self.lines[i] = list(reversed(self.lines[i][1:])) + self.lines[j]
                    del self.lines[j]
                elif self.lines[i][-1] == self.lines[j][-1]:
                    self.lines[i] = self.lines[i] + list(reversed(self.lines[j][:-1]))
                    del self.lines[j]

                if len(self.lines) != length:
                    changes = True
                    break

    def smoothen(self):
        """ remove points that continue in the same direction"""
        return
        new_lines = []
        for line in self.lines:
            new_line = [line[0]]
            for i in range(len(line) - 2):
                z0, z1, z2 = line[i:i + 3]
                if not (-ANGLE_LIMIT <= angle_between_points(z0, z1, z2) <= ANGLE_LIMIT) and abs(z1 - z0) < LENGTH_MIN and abs(z2 - z1) < LENGTH_MIN:
                    new_line.append(z1)
            new_line.append(line[-1])
            new_lines.append(new_line)
        self.lines = new_lines

    def split(self):
        """if zig-zag lines are too long, split them"""

        new_lines = []
        for line in self.lines:
            new_line = [line[0]]
            for i in range(len(line) - 2):
                z0, z1, z2 = line[i:i + 3]
                if not (-ANGLE_LIMIT <= angle_between_points(z0, z1, z2) <= ANGLE_LIMIT) and abs(
                        z1 - z0) < LENGTH_MIN and abs(z2 - z1) < LENGTH_MIN:
                    new_line.append(z1)
            new_line.append(line[-1])
            new_lines.append(new_line)
        self.lines = new_lines

    def __iter__(self):
        return iter(self.lines)

    def __repr__(self):
        return f"Zig-zag {self.NW}, {self.SW}, {self.SE}, {self.NE}"


# def remove_consecutive_duplicates(points):
#     # Remove consecutive duplicates
#     unique_points = [points[0]]  # Start with the first point
#     for i in range(1, len(points)):
#         if points[i] != points[i - 1]:  # Check if the current point is different from the previous one
#             unique_points.append(points[i])
#     return unique_points


def connect(z, w, pos):
    """Make a connection line with vertical, horizontal or diagonal lines between points z (left) and w (right)
    pos can be "N" or "S"
    """

    # if they are the same, no connection
    if z == w:
        return []

    def sign(i):
        return 1 if i > 0 else (-1 if i < 0 else 0)

    d = w - z
    dx, dy = d.real, d.imag

    # if they are horizontally/vertically aligned or if they are diagonal, make a simple segment
    if z.real == w.real or z.imag == w.imag or abs(dx) == abs(dy):
        return [z, w]

    result = []
    # add diagonal line + horizontal line
    if abs(dx) > abs(dy):
        # diagonal line starting from z
        if pos == "N" and dy > 0 or pos == "S" and dy < 0:
            result = [z, z + abs(dy) + 1j * dy, w]
        # diagonal line starting from w
        else:
            result = [z, w - abs(dy) - 1j * dy, w]

    # add diagonal line + vertical line
    if abs(dx) < abs(dy):
        # diagonal line starting from z
        if pos == "N" and dy < 0 or pos == "S" and dy > 0:
            result = [z, z + dx + 1j * sign(dy) * dx, w]
        # diagonal line starting from w
        else:
            result = [z, w - dx - 1j * sign(dy) * dx, w]

    if abs(result[1] - result[0]) < 1 or abs(result[2] - result[1]) < 1:
        return [result[0], result[2]]
    return result


def crossings(expr):
    """ return the number of crossings"""
    if isinstance(expr, int):
        return abs(expr)
    return sum(crossings(t) for t in expr)


def to_zigzag(expr):
    """ Convert a algebraic tangle into a geometric zig-zag line"""

    if isinstance(expr, int):
        # elementary zig-zag 1, -1, or 0
        zz = ZigZag()  # left zig-zag
        if expr == 1 or expr == -1:
            zz.add_line([1j, 0.5 + 0.5j + (-0.5 + 0.5j) * GAP])
            zz.add_line([1,0.5 + 0.5j - (-0.5 + 0.5j) * GAP])
            zz.add_line([0j, 1 + 1j])
            # zz.add_line([0j, 0.5+0.5j])
            # zz.add_line([0.5 + 0.5j, 1 + 1j])
            zz.set_compass(NW=1j, SW=0j, SE=1+0j, NE=1 +1j)
            if expr == -1:
                zz.mirror()
        elif expr == 0:
            # assume the curve will be smoothened
            zz.add_line([1j, 0.5 + 0.5j, 1 + 1j])
            zz.add_line([0j, 0.5 + 0.5j, 1 + 0j])
            zz.set_compass(NW=1j, SW=0j, SE=1+0j, NE=1 +1j)
        else:
            raise NotImplementedError("Can only layout 1 or -1 tangles")

    elif isinstance(expr, TangleExpr) and isinstance(expr.terms[1], int) and expr.terms[1] == 0:
        # addition or multiplication by 0
        zz = to_zigzag(expr.terms[0])  # just draw the left tangle, no need for the right one
        if isinstance(expr, TangleProduct):
            zz.reflect()

    elif isinstance(expr, TangleExpr):

        L, R = expr.terms

        zl = to_zigzag(L)
        zr = to_zigzag(R)

        if isinstance(expr, TangleProduct):
            zl.reflect()

        # scale subtangle
        # if zl.height > zr.height:
        #     zr.set_height(zl.height)
        # elif zl.height < zr.height:
        #     zl.set_height(zr.height)

        # center the two tangles vertically (y-axis)
        zr.move(0.5 * (zl.N + zl.S - zr.N - zr.S))

        # move right tangle to the right
        if zl.height == 1 and zr.height == 1:
            zr.move(zl.bounding_box("E") - zr.bounding_box("W"))
        else:
            zr.move(zl.bounding_box("E") - zr.bounding_box("W") + SUBTANGLE_DISTANCE)

        zz = zl + zr  # add the two zig-zags

        # draw the connecting arcs
        zz.add_line(connect(zl.NE, zr.NW, "N"))
        zz.add_line(connect(zl.SE, zr.SW, "S"))

    else:
        raise TypeError(f"Cannot put to zig-zag instance of type {type(expr)}")

    return zz


def add_corners_and_smoothen(zz: ZigZag):

    """ add corners to the zig-zag line, so it starts and ends in a rectangle (the bounding box)"""
    additional_corner_length = 0.5

    N = zz.bounding_box("N") + additional_corner_length * 1j
    S = zz.bounding_box("S") - additional_corner_length * 1j
    W = zz.bounding_box("W") - additional_corner_length
    E = zz.bounding_box("E") + additional_corner_length

    # scale endpoints so the tangle fits into a square
    if FIT_INTO_SQUARE:
        w, h = E.real - W.real, N.imag - S.imag
        if h < w:
            N += (w-h) * 0.5j
            S -= (w-h) * 0.5j
        elif h > w:
            W -= (h-w) * 0.5
            E += (h-w) * 0.5

    if zz.NW != N + W:
        zz.add_line(connect(N + W, zz.NW, "S"))
    if zz.SW != S + W:
        zz.add_line(connect(S + W, zz.SW, "N"))
    if zz.SE != S + E:
        zz.add_line(connect(zz.SE, S + E, "S"))
    if zz.NE != N + E:
        zz.add_line(connect(zz.NE, N + E, "N"))

    zz.set_compass(NW=N + W, SW=S + W, SE=S + E, NE=N + E)

    zz.smoothen()


def draw(expr, translate=0j):
    """ Convert the tangle expr, convert it to a zig-zag line and plot it."""

    z = to_zigzag(expr)
    add_corners_and_smoothen(z)

    #z.set_height(z.width)

    # plot the zig-zag
    for i, line in enumerate(z.lines):
        x_values = [z.real + translate.real for z in line]
        y_values = [z.imag + translate.imag for z in line]
        plt.plot(x_values, y_values, lw=LINE_WIDTH)

    plt.grid(SHOW_GRID)
    plt.axis("on" if SHOW_AXIS else "off")

    plt.gca().set_aspect('equal', adjustable='box')
    #plt.show()


def draw_smooth(expr):
    """Draws the tangle expr and smoothens it."""

    degree = 2  # degree of the spline (e.g., cubic)
    N = 10  # number of interpolate points

    z = to_zigzag(expr)

    add_corners_and_smoothen(z)

    #z.set_height(z.width)

    for i, line in enumerate(z.lines):

        k = 1 if len(line) == 2 else (2 if len(line) == 3 else degree)

        control_points = np.array(line)
        length = np.sum(np.abs(np.diff(control_points)))  # full length of the zig-zag line

        x = np.real(control_points)
        y = np.imag(control_points)

        # Define the knot vector (uniform knots, ensuring valid spline)
        n = len(control_points)  # Number of control points
        t = np.concatenate(([0] * k, np.linspace(0, 1, n - k + 1), [1] * k))
        # Parameterize the spline
        t_fine = np.linspace(0, 1, int(length * N))  # Fine sampling for smooth curve

        # Create B-spline for real and imaginary parts
        spline_x = BSpline(t, x, k)(t_fine)  # B-spline for x
        spline_y = BSpline(t, y, k)(t_fine)  # B-spline for y

        # Plot the control points and the B-spline
        plt.plot(spline_x, spline_y, label='B-spline Curve', linewidth=LINE_WIDTH, color="tab:blue")  # B-spline

    if ENDPOINT_SIZE > 0:
        endpoints = np.array([z.NW, z.SW, z.SE, z.NE])
        x = np.real(endpoints)
        y = np.imag(endpoints)
        plt.scatter(x, y, color='k', marker='o', s=ENDPOINT_SIZE, zorder=10)

    plt.grid(SHOW_GRID)
    plt.axis("on" if SHOW_AXIS else "off")
    plt.gca().set_aspect('equal', adjustable='box')

    #draw(expr, translate=z.height * 1.5j + 2j)

    #plt.show()


def nested_to_product(t):
    """ convert a nested tuple, e.g. 1,(2,3) to nested product of tangles Product(1, Product(2,3))."""

    if isinstance(t, int):
        return integral(t)

    if isinstance(t, tuple):
        if len(t) == 1:
            return nested_to_product(t)
        # start with first 2 elements
        result = TangleProduct(nested_to_product(t[0]), nested_to_product(t[1]))
        for elem in t[2:]:
            result = TangleProduct(result, nested_to_product(elem))
        return result

    raise ValueError("Input must be a nested tuple of integers.")

def main(tangle_str):
    tangle = nested_to_product(eval(tangle_str))
    draw_smooth(tangle)
    plt.savefig("{}.svg".format(tangle_str.replace(',','_')), format='svg', bbox_inches='tight', pad_inches=0)
    plt.show()

if __name__ == "__main__":
    tangle_str = '2,3,(2,0)'
    main(tangle_str)


