import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label
from statsmodels.graphics.tukeyplot import results
from statsmodels.tsa.seasonal import seasonal_decompose, STL, MSTL
import pandas as pd
from scipy.signal import savgol_filter
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.interpolate import UnivariateSpline




def water_consumption(t, C_base=1000, C_day=500, A_morning=2000, A_midday= 1000, A_evening=1500,
                      t_morning=7.0, t_midday=12.0, t_evening=18.0, sigma_morning=4.0, sigma_midday=8, sigma_evening=5,
                      S=1.0, D=1.0):
    """
    Calculate hourly water consumption based on a parametric model.

    Parameters:
    t (float): Hour of the day (0 to 23).
    C_base (float): Baseline consumption (m³/h).
    C_day (float): Additional daytime consumption (m³/h).
    A_morning, A_evening (float): Amplitudes of morning and evening peaks (m³/h).
    t_morning, t_evening (float): Centers of morning and evening peaks (hours).
    sigma_morning, sigma_evening (float): Widths of morning and evening peaks (hours).
    S (float): Seasonal scaling factor (e.g., 1.2 for summer).
    D (float): Day-type factor (e.g., 0.8 for weekends).

    Returns:
    float: Water consumption rate at time t (m³/h).
    """
    # # Daytime consumption (simple step function: 1 for 8:00–16:00, 0 otherwise)
    # f_day = 1.0 if 8 <= t < 16 else 0.0

    # Morning peak (Gaussian)
    morning_peak = A_morning * np.exp(-((t - t_morning) ** 2) / (2 * sigma_morning ** 2))

    # Evening peak (Gaussian)
    midday_peak = A_midday * np.exp(-((t - t_midday) ** 2) / (2 * sigma_midday ** 2))

    # Evening peak (Gaussian)
    evening_peak = A_evening * np.exp(-((t - t_evening) ** 2) / (2 * sigma_evening ** 2))

    # Total consumption
    C = S * D * (C_base + C_day  + morning_peak + midday_peak + evening_peak)
    # C = S * D * (C_base + C_day + morning_peak + evening_peak)


    return C


# Simulate consumption over 24 hours
factor = 4
hours_in_a_day = 24
range_  = hours_in_a_day*factor
hours = range(range_)  # Fine-grained time points for smooth plotting
# consumption = [water_consumption(t) for t in hours]

time_range = pd.date_range(start='00:00', periods=24 * 4, freq='15min')

# Format the timestamps to display only the time
time_list = [time.strftime('%H:%M') for time in time_range]


C_base=100
C_day=10
A_morning=1900
A_midday=800
A_evening=1500
t_morning=9.0 * factor
t_midday=12.0 * factor
t_evening=18.0 * factor
sigma_morning=8
sigma_midday=12
sigma_evening=7
S=1.0
D=1.0



consumption = [water_consumption(t , C_base=C_base, C_day=C_day, A_morning=A_morning, A_midday= A_midday, A_evening=A_evening,
                      t_morning=t_morning, t_midday=t_midday, t_evening=t_evening, sigma_morning=sigma_morning, sigma_midday=sigma_midday, sigma_evening=sigma_evening,
                      S=S, D=D)for t in hours]
# Create a DataFrame for seasonal decomposition
df = pd.DataFrame({'demand': consumption, 'hours': hours})

window_size = 3
df['trend_calculated'] = df['demand'].rolling(window=window_size,
                                                                   center=True,
                                                                   min_periods=1).mean()


# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(hours, consumption, label="Water Consumption", color='blue')

plt.xlabel("Hour of the Day")
plt.ylabel("Consumption (m³/h)")
plt.title("Parametric Model of Hourly Water Consumption")
plt.grid(True)
plt.xticks(hours, time_list, rotation=90, ha='right')

plt.legend()
plt.show()


