import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches
import os
import plotly.graph_objects as go
import matplotlib.pyplot as plt

"given a normal vector finds plane equatino  used for visualizng planes after euler angle rotation "
def calculate_plane_from_normal(n, x, y, z_offset=0):
    # For a plane with normal vector (A, B, C) and passing through the origin, 
    # the plane equation is: Ax + By + Cz = 0
    # We can solve for z: z = -(Ax + By) / C
    z = -(n[0]*x + n[1]*y) / n[2] + z_offset
    return z


"make radar plot to quickly glance at euler angle data"
def plot_radar_chart(proximal_angles, distal_angles, seg):
    assert len(proximal_angles) == len(distal_angles)
    
    labels = np.array(['Z Direction', 'Y Direction', 'X Direction'])
    num_vars = len(labels)

    # compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

    # the radar chart will be a circle, so we need to "close the loop"
    # by appending the start value to the end.
    angles += angles[:1]

    # convert the angles from radians to degrees
    proximal_angles = [np.degrees(angle) for angle in proximal_angles]
    distal_angles = [np.degrees(angle) for angle in distal_angles]
    
    # append the start value to the end
    proximal_angles += proximal_angles[:1]
    distal_angles += distal_angles[:1]
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    
    # plot the proximal angles
    ax.fill(angles, proximal_angles, color='red', alpha=0.25, label='Proximal')
    
    # plot the distal angles
    ax.fill(angles, distal_angles, color='blue', alpha=0.25, label='Distal')

    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
    plt.savefig(f'figures/segment_{seg}_radar.png')

"generate delta plots"
def plot_deltas(delta_values, segments):
    # Create the figure and the axes
    fig, ax = plt.subplots()

    # Prepare labels and locations
    labels = ['ΔProximal', 'ΔDistal'] * segments
    segment_labels = [f'Segment {i+1}' for i in range(segments)] * 2
    segment_labels.sort()

    # Set the bar width
    bar_width = 0.35

    # Set the x coordinates of the bars
    index = np.arange(segments*2)

    # # Create a bar plot
    # bars = ax.bar(index, delta_values, color='black', width=bar_width)


   # Define colors for each segment
    colors = ['white', 'black', 'grey']
    
    
    # Create a bar plot
    for i in range(segments):
        bars = ax.bar(index[i*2:i*2+2], delta_values[i*2:i*2+2], color=colors[i], width=bar_width, label=f'Segment {i+1}', edgecolor='black')
    
    
    # Set x-axis labels and title
    ax.set_ylabel('Degrees')
    ax.set_title('Δ Angle and Refrence Plane Plan and PostOp')

    # Set the position of the x ticks
    ax.set_xticks(index)

    # Set the labels for the x ticks
    ax.set_xticklabels(labels)
    
    ax.legend()

    
    plt.savefig(f'figures/delta_analysis.png')

"plot the euler angles in bar graph form"
def euler_bar_chart(prox, dist, seg):
    # Convert to arrays
    proximal = np.array(prox)
    distal = np.array(dist)

    # Convert to degrees
    proximal = np.rad2deg(proximal)
    distal = np.rad2deg(distal)

    # Define the labels for the bars
    labels = ['Z Direction', 'Y Direction', 'X Direction']

    # Define the bar width and positions
    bar_width = 0.35
    r1 = np.arange(len(labels))  # e.g., [0, 1, 2]
    r2 = [x + bar_width for x in r1]  # e.g., [0.35, 1.35, 2.35]

    # Plot the data
    fig, ax = plt.subplots()

    ax.bar(r1, proximal, color='red', width=bar_width, label='Proximal')
    ax.bar(r2, distal, color='blue', width=bar_width, label='Distal')

    # Add xticks on the middle of the group bars
    # plt.xlabel('group', fontweight='bold')
    plt.xticks([r + bar_width / 2 for r in range(len(proximal))], labels)

    ax.set_ylabel('Degrees of Error')
    ax.set_title('Error in Euler Angles')
    ax.legend()
    plt.savefig(f'figures/segment_{seg}_column.png')


"plot rotated, plan, and postop planes from euler angle calculation. saves in html file"
def plot_planes(planeA, planeB, planeC, segment_num, location):
    x = np.linspace(-1, 1, 10)
    y = np.linspace(-1, 1, 10)
    x, y = np.meshgrid(x, y)
    title = f'Segment {segment_num} {location} Plane'

    zA = calculate_plane_from_normal(planeA, x, y)
    zB = calculate_plane_from_normal(planeB, x, y)
    zC = calculate_plane_from_normal(planeC, x, y, z_offset=1)

    fig = go.Figure(data=[
        go.Surface(x=x, y=y, z=zA, name='Pre-op', colorscale='Reds',  showscale=False),
        go.Surface(x=x, y=y, z=zB, name='Post-op', colorscale='Blues', showscale=False),
        go.Surface(x=x, y=y, z=zC, name='Rotated Post-op', colorscale='Greens', showscale=False),


    # Dummy scatter3d plots for legend
        go.Scatter3d(x=[None], y=[None], z=[None], mode='markers',
                        marker=dict(size=0, color='red'), name='Pre-op'),
        go.Scatter3d(x=[None], y=[None], z=[None], mode='markers',
                        marker=dict(size=0, color='blue'), name='Post-op'),
        go.Scatter3d(x=[None], y=[None], z=[None], mode='markers',
                        marker=dict(size=0, color='green'), name='Rotated Post-op'),
        ])

    fig.update_layout(
        title=title,
        autosize=False,
        width=800,
        height=600,
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z'),
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )

    if not os.path.exists('figures'):
        os.makedirs('figures')

    fig.write_html('figures/' + title + '.html')  # Save the figure as an interactive HTML file


"visualize registered point segments and save as html file"
def visualize_data(plan, postop, segment_num):
    plane1, plane2 = plan
    plane3, plane4 = postop

    title = f'Segment {segment_num} Registered'

    fig = go.Figure()

    for i, planes in enumerate([[plane1, plane2], [plane3, plane4]]):
        if i == 0:
            color = 'red'
            name = 'Plan'
        else:
            color = 'blue'
            name = 'Post-op'

        for j, plane in enumerate(planes):
            x = [point[0] for point in plane]
            y = [point[1] for point in plane]
            z = [point[2] for point in plane]

            # Add point at the end to close the triangle
            x.append(plane[0][0])
            y.append(plane[0][1])
            z.append(plane[0][2])

            # Connect the corresponding points with lines
            if j == 0:
                x_next = [point[0] for point in planes[j+1]]
                y_next = [point[1] for point in planes[j+1]]
                z_next = [point[2] for point in planes[j+1]]

                for k in range(len(x)-1):  # Adjust to exclude last point (repeated for closing the triangle)
                    fig.add_trace(go.Scatter3d(
                        x=[x[k], x_next[k]], 
                        y=[y[k], y_next[k]], 
                        z=[z[k], z_next[k]], 
                        mode='lines',
                        line=dict(color=color),
                        name=name,
                        showlegend=(k == 0)  # Show legend only for the first line
                    ))

            # Add the plane as a scatter plot
            fig.add_trace(go.Scatter3d(
                x=x, 
                y=y, 
                z=z, 
                mode='lines+markers',
                line=dict(color=color),
                marker=dict(size=10, color=color),
                name=name,
                showlegend=False
            ))

    # Add the origin point
    fig.add_trace(go.Scatter3d(
        x=[0], 
        y=[0], 
        z=[0], 
        mode='markers',
        marker=dict(size=6, color='black'),
        name='Origin (Proximal End of Model)'
    ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z'),
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )

    if not os.path.exists('figures'):
        os.makedirs('figures')

    fig.write_html('figures/' + title + '.html')  # Save the figure as an interactive HTML file
