import numpy as np
# Function to find intersection of line and rectangle
def intersect(p1, rect, line_vec):
    rect_min = np.array([rect[0], rect[1]])  # [xmin, ymin]
    rect_max = np.array([rect[2], rect[3]])  # [xmax, ymax]
    

    # Calculate intersection points with the rectangle edges
    # Check for all four edges of the rectangle (left, right, top, bottom)
    
    # Left edge (x = xmin)
    if line_vec[0] != 0:  # Prevent division by zero
        t = (rect_min[0] - p1[0]) / line_vec[0]
        if t >= 0:
            y_intersect = p1[1] + t * line_vec[1]
            if rect_min[1] <= y_intersect <= rect_max[1]:
                return [rect_min[0], y_intersect]

    # Right edge (x = xmax)
    if line_vec[0] != 0:
        t = (rect_max[0] - p1[0]) / line_vec[0]
        if t >= 0:
            y_intersect = p1[1] + t * line_vec[1]
            if rect_min[1] <= y_intersect <= rect_max[1]:
                return [rect_max[0], y_intersect]

    # Bottom edge (y = ymin)
    if line_vec[1] != 0:  # Prevent division by zero
        t = (rect_min[1] - p1[1]) / line_vec[1]
        if t >= 0:
            x_intersect = p1[0] + t * line_vec[0]
            if rect_min[0] <= x_intersect <= rect_max[0]:
                return [x_intersect, rect_min[1]]

    # Top edge (y = ymax)
    if line_vec[1] != 0:
        t = (rect_max[1] - p1[1]) / line_vec[1]
        if t >= 0:
            x_intersect = p1[0] + t * line_vec[0]
            if rect_min[0] <= x_intersect <= rect_max[0]:
                return [x_intersect, rect_max[1]]



rect = (0.125, 0.25862068965517243, 0.10999999999999999, 0.3364705882352941)
start = (57.55541666666666, 26.33347222222222)
vector = np.array([-0.99641961,  0.08454561])
intersect(start, rect, vector)