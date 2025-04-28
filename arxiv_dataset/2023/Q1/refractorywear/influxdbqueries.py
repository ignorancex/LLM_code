import influxdb_client
from influxdb_client.client.write_api import SYNCHRONOUS
import numpy as np
from datetime import datetime

token = "token"
bucket = "bucket"
org = "org"
url = "https://test.com"
client    = influxdb_client.InfluxDBClient(url=url, token=token, org=org, timeout=20000)
query_api = client.query_api()
delete_api = client.delete_api()


def cyclic(heat_number, field):
    query = f' from(bucket: "{bucket}") \
    |> range(start: 0) \
    |> filter(fn: (r) => r["_measurement"] == "cyclic")\
    |> filter(fn: (r) => r.heatID == "cyclic_{heat_number}")\
    |> filter(fn: (r) => r["_field"] == "{field}") '

    result = query_api.query(org=org, query=query)
    results = []
    for table in result:
        for record in table.records:
            results.append(record.get_value())

    return results


def acyclic(heat_number, ladle_number, use_number, field):
    query = f' from(bucket: "{bucket}") \
    |> range(start: 0) \
    |> filter(fn: (r) => r["_measurement"] == "acyclic")\
    |> filter(fn: (r) => r.heatID == "{ladle_number:02d}_{use_number:04d}_{heat_number}")\
    |> filter(fn: (r) => r["_field"] == "{field}") '

    result = query_api.query(org=org, query=query)
    results = []
    for table in result:
        for record in table.records:
            results.append(record.get_value())

    return results


def db_time(heat_number):

    results = cyclic(heat_number, "FECHAHORA")
    retval = []
    for record in results:
        retval.append((np.datetime64(record)-np.datetime64(results[0]))/np.timedelta64(1, 's'))
    return retval


def db_timep(heat_number):

    return cyclic(heat_number, "FECHAHORA")


def db_temperature(heat_number):

    return [float(x) for x in cyclic(heat_number, "TEMPERATURA")]

def db_amount_steel_alt(heat_number):
    query = f' from(bucket: "{bucket}") \
    |> range(start: 0) \
    |> filter(fn: (r) => r["_measurement"] == "steel_weights")\
    |> filter(fn: (r) => r["heatNumber"] == "{heat_number}")\
    |> filter(fn: (r) => r["_field"] == "LiquidSteel") '
    result = query_api.query(org=org, query=query)
    results = []
    for table in result:
        for record in table.records:
            results.append(record.get_value())
    return results


def db_amount_steel(heat_number, ladle_number, use_number):

    weight = float(acyclic(heat_number, ladle_number, use_number, "Liquid_Steel_Wheight")[0])
    if weight < 1.0:
        weight = float(db_amount_steel_alt(heat_number)[0])

    return weight



def db_tap_temperature(heat_number, ladle_number, use_number):

    return float(acyclic(heat_number, ladle_number, use_number, "Steel_Temperature_at_Tap")[0]) + 273.15


def db_gas(heat_number):

    return [float(x) for x in cyclic(heat_number, "CAUDAL_GAS")]


def db_power(heat_number):

    return [float(x) for x in cyclic(heat_number, "CONSUMO_ELECTRICO")]


def db_vacuum_pressure(heat_number):

    return [float(x) for x in cyclic(heat_number, "PRESION_VACIO")]


def db_temperature_points(heat_number, ladle_number, use_number):
    query = f' from(bucket: "{bucket}") \
    |> range(start: 0) \
    |> filter(fn: (r) => r["_measurement"] == "temperatures")\
    |> filter(fn: (r) => r.heatID == "{ladle_number:02d}_{use_number:04d}_{heat_number}")\
    |> filter(fn: (r) => r["_field"] == "Temperature") '

    result = query_api.query(org=org, query=query)
    results = []
    for table in result:
        for record in table.records:
            results.append(float(record.get_value()))
    return results


def db_temperature_timepoints(heat_number, ladle_number, use_number, t0):
    query = f' from(bucket: "{bucket}") \
    |> range(start: 0) \
    |> filter(fn: (r) => r["_measurement"] == "temperatures")\
    |> filter(fn: (r) => r.heatID == "{ladle_number:02d}_{use_number:04d}_{heat_number}")\
    |> filter(fn: (r) => r["_field"] == "Sample_DateTime") '

    result = query_api.query(org=org, query=query)
    results = []
    for table in result:
        for record in table.records:
            results.append(np.datetime64(datetime.strptime(record.get_value(), '%d-%m-%Y %H:%M:%S')))
    retval = []
    for record in results:
        retval.append((np.datetime64(record)-np.datetime64(t0))/np.timedelta64(1, 's'))
    return retval


def db_additiontime(heat_number, ladle_number, use_number):
    query = f' from(bucket: "{bucket}") \
    |> range(start: 0) \
    |> filter(fn: (r) => r["_measurement"] == "additions")\
    |> filter(fn: (r) => r.heatID == "{ladle_number:02d}_{use_number:04d}_{heat_number}")\
    |> filter(fn: (r) => r["_field"] == "Addition_DateTime") '

    result = query_api.query(org=org, query=query)
    results = []
    for table in result:
        for record in table.records:
            results.append(np.datetime64(datetime.strptime(record.get_value(), '%d-%m-%Y %H:%M:%S')))
    return results


def db_additionweight(heat_number, ladle_number, use_number):
    query = f' from(bucket: "{bucket}") \
    |> range(start: 0) \
    |> filter(fn: (r) => r["_measurement"] == "additions")\
    |> filter(fn: (r) => r.heatID == "{ladle_number:02d}_{use_number:04d}_{heat_number}")\
    |> filter(fn: (r) => r["_field"] == "Product_Weight") '

    result = query_api.query(org=org, query=query)
    results = []
    for table in result:
        for record in table.records:
            results.append(float(record.get_value()))
    return results


def db_check_model_data(heat_number, ladle_number, use_number):
    print( heat_number, ladle_number, use_number)
    query = f' from(bucket: "{buckets}") \
    |> range(start: 0) \
    |> filter(fn: (r) => r["_measurement"] == "model1") \
    |> filter(fn: (r) => r.heatID == "{ladle_number:02d}_{use_number:04d}_{heat_number}")\
    |> filter(fn: (r) => r["_field"] == "Time") \
    |> sort() \
    |> yield(name: "sort") '
    result = query_api.query(org=orgs, query=query)
    return len(result) == 0
def db_save(field, heat_number, ladle_number, use_number, value):
    p = influxdb_client.Point("model1").tag("heatID", f'{ladle_number:02d}_{use_number:04d}_{heat_number}').field(field, value)
    write_api.write(bucket=buckets, org=orgs, record=p)

# def delete():
#     start = "1970-01-01T00:00:00Z"
#     stop = "2024-01-01T00:00:00Z"
#     delete_api.delete(start, stop, '_measurement="model"', bucket=bucket, org=org)
