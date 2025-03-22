
import datetime
from dhnamlib.pylib.time import get_YmdHMSf


initial_datetime = datetime.datetime.now()
initial_date_str = get_YmdHMSf(initial_datetime)
