# The hourly aggregate water consumption model of a typical city varies throughout the day, influenced by human behavior, daily activities, and the city’s characteristics. Below is a verbal description of such a model:
#
# 1. **Nighttime (00:00–05:00)**: Water consumption is very low, as most residents are asleep. Minimal usage occurs, mainly for basic needs like toilets or limited activity from businesses operating at night (e.g., certain factories).
#
# 2. **Early Morning (05:00–08:00)**: Consumption gradually increases as residents wake up. A distinct morning peak occurs around 06:00–08:00, driven by activities such as showering, cooking, watering gardens, or other household needs before heading to work or school.
#
# 3. **Daytime (08:00–16:00)**: Consumption remains relatively stable but higher than at night. During work hours, usage is driven by commercial, industrial, and institutional activities (e.g., schools, offices, restaurants). Slight increases may occur around noon due to food preparation or restroom use.
#
# 4. **Afternoon and Evening (16:00–20:00)**: Consumption rises again, often reaching another peak, as residents return home. Activities like showering, laundry, cooking, and garden watering contribute to this increase. The evening peak may be similar to or slightly lower than the morning peak.
#
# 5. **Early Night (20:00–00:00)**: Consumption gradually declines as household activities decrease. After 22:00, most households reduce water use, and consumption returns to lower levels.
#
# **Additional Characteristics**:
# - **Seasonality**: In summer, consumption may be higher due to garden watering, pool filling, or increased use for cooling (e.g., more frequent showers).
# - **Day of the Week**: Weekdays show a more consistent pattern, while weekends may have later peaks (e.g., late morning) and less defined patterns.
# - **External Factors**: Events like water outages, holidays, or extreme weather (heavy rain or drought) can alter the pattern.
#
# This model is general and may vary based on the city’s size, residents’ culture, water infrastructure, and local consumption habits. If you’d like a tailored description for a specific city or more details, please let me know!


import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import datetime


class DayPattern:
    def __init__(self, day_type=0, date=datetime.date.today()):
        self.date = date
        self.day = date.day
        self.week_number = date.isocalendar()[1]
        self.year = self.date.year
        self.month = self.date.month
        self.week_day = self.date.weekday()

        self.day_type = day_type
        self.number_high_picks = 1
        self.number_low_picks = self.number_high_picks + 1
        self.max_flow = 1
        self.min_flow = 0
        # self.hourly_time_range = self.create_hourly_time_rage(self.day, self.week_number)
        # self.hourly_time_range = self.create_hourly_time_rage(self.day, self.week_number)

    def update_day(self, day):
        self.day = day

    def update_day_type(self, day_type):
        self.day_type = day_type

    def update_max_flow(self, max_flow):
        self.max_flow = max_flow

    def update_min_flow(self, min_flow):
        self.min_flow = min_flow

    def update_num_picks(self, num_high_picks):
        self.number_high_picks = num_high_picks
        self.number_low_picks = num_high_picks + 1

    def create_daily_demand_pattern(self):
        pass

    @staticmethod
    def create_24_hours_rage():
        return range(0, 24)


if __name__ == "__main__":
    print(datetime.date.today())
    daypa_ttern = DayPattern()
    week = daypa_ttern.date
    print(dir(week))
