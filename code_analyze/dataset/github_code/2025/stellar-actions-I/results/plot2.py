#Second plot of the paper- fig 2- shift in CoM of cold gas
## data not available in public- so code just for reference

import csv
import matplotlib.pyplot as plt
import numpy as np

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 15

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=15)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# Define the function to read shift values from a CSV file
def read_shift_values_from_csv(file_path):
    shifts = []
    with open(file_path, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            shifts.append([float(x) for x in row])
    return shifts

# function to fit a line
def fit_line(x,y):
    # Fit a straight line using np.polyfit (linear regression)
    coefficients = np.polyfit(x, y, 1)  # 1 means it's a first-degree polynomial (straight line)
    slope, intercept = coefficients
    
    # Generate predicted y-values
    y_pred = slope * x + intercept
    
    # Calculate R-squared
    ss_total = np.sum((y - np.mean(y)) ** 2)         # Total sum of squares
    ss_residual = np.sum((y - y_pred) ** 2)          # Residual sum of squares
    r_squared = 1 - (ss_residual / ss_total)
    return slope,intercept,r_squared,y_pred


  
shifts=read_shift_values_from_csv('/g/data/jh2/ax8338/galaxy_origin_shift/chuhan_initstars_com.csv')

plt.figure(figsize=(8,6))
t = np.arange(np.shape(shifts)[0]) #time
x = np.array(shifts)[:,0]
y = np.array(shifts)[:,1]
z = np.array(shifts)[:,2]

mx,cx,r_squaredx,xpred=fit_line(t,x)
my,cy,r_squaredy,ypred=fit_line(t,y)
mz,cz,r_squaredz,zpred=fit_line(t,z)

print(f"Slope (m): {round(float(mx),2),round(float(my),2),round(float(mz),2)}")
print(f"Intercept (c): {float(cx),float(cy),float(cz)}")
print(f"R-squared: {r_squaredx:.2f},{r_squaredy:.2f},{r_squaredz:.2f}")

# Plot the data points
# plt.plot((t-100)[::2], x[::2], '.',markersize=3,label='x')
# plt.plot((t-100)[::2],y[::2],'.',markersize=3,label='y')
plt.plot((t-100)[::2],z[::2],'.',markersize=3,label='$z$')

# Plot the fitted line
# plt.plot(t-100, xpred, c='tab:blue',label=f'x = {mx:.2f}t + ({cx:.2f})')
# plt.plot(t-100, ypred,c='tab:orange', label=f'y = {my:.2f}t + ({cy:.2f})')
plt.plot(t-100, zpred, c='orange',label=f'$z = {mz:.2f}t + {cz:.2f}$')

# Add labels and title
plt.xlabel('Time (Myr)')
plt.ylabel(r'CoM $z$ coordinate (pc)')
# plt.title(f"$R^2$: {r_squaredx:.2f}, {r_squaredy:.2f}, {r_squaredz:.2f}")
plt.legend(markerscale=2)
plt.tight_layout()
# Show plot
plt.savefig('/g/data/jh2/ax8338/galaxy_origin_shift/CoM_chuhan_z.png')
plt.show()

