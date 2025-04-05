import time
from datetime import datetime, timedelta


class Timer:
    def __init__(self):
        self.start_time = None

    def start(self):
        self.start_time = time.time()

    def stop(self):
        print('>>> Elapsed time: {0}\n'.format(str(timedelta(seconds=time.time() - self.start_time))[:-3]))


def strify(iterable_obj, delimiter=','):
    return delimiter.join(iterable_obj)


date_format = {'tweet': '%a %b %d %H:%M:%S %z %Y',
               'youtube': '%Y-%m-%d'}


def str2obj(str, fmt='youtube'):
    if fmt == 'tweet' or fmt == 'youtube':
        return datetime.strptime(str, date_format[fmt])
    else:
        return datetime.strptime(str, fmt)


def obj2str(obj, fmt='youtube'):
    if fmt == 'tweet' or fmt == 'youtube':
        return obj.strftime(date_format[fmt])
    else:
        return obj.strftime(fmt)


# twitter's snowflake parameters
twepoch = 1288834974657
datacenter_id_bits = 5
worker_id_bits = 5
sequence_id_bits = 12
max_datacenter_id = 1 << datacenter_id_bits
max_worker_id = 1 << worker_id_bits
max_sequence_id = 1 << sequence_id_bits
max_timestamp = 1 << (64 - datacenter_id_bits - worker_id_bits - sequence_id_bits)


def make_snowflake(timestamp_ms, datacenter_id, worker_id, sequence_id, twepoch=twepoch):
    """generate a twitter-snowflake id, based on
    https://github.com/twitter/snowflake/blob/master/src/main/scala/com/twitter/service/snowflake/IdWorker.scala
    :param: timestamp_ms time since UNIX epoch in milliseconds"""
    timestamp_ms = int(timestamp_ms)
    sid = ((timestamp_ms - twepoch) % max_timestamp) << datacenter_id_bits << worker_id_bits << sequence_id_bits
    sid += (datacenter_id % max_datacenter_id) << worker_id_bits << sequence_id_bits
    sid += (worker_id % max_worker_id) << sequence_id_bits
    sid += sequence_id % max_sequence_id
    return sid


def melt_snowflake(snowflake_id, twepoch=twepoch):
    """inversely transform a snowflake id back to its components."""
    snowflake_id = int(snowflake_id)
    sequence_id = snowflake_id & (max_sequence_id - 1)
    worker_id = (snowflake_id >> sequence_id_bits) & (max_worker_id - 1)
    datacenter_id = (snowflake_id >> sequence_id_bits >> worker_id_bits) & (max_datacenter_id - 1)
    timestamp_ms = snowflake_id >> sequence_id_bits >> worker_id_bits >> datacenter_id_bits
    timestamp_ms += twepoch
    return timestamp_ms, datacenter_id, worker_id, sequence_id


def count_track(track_list, start_with_rate=False, subcrawler=False):
    if subcrawler:
        total_track_cnt = 0
        for i in range(len(track_list)):
            total_track_cnt += count_track(track_list[i], start_with_rate=start_with_rate, subcrawler=False)
        return total_track_cnt
    else:
        if len(track_list) == 0:
            return 0

        if start_with_rate:
            track_cnt = -track_list[0]
        else:
            track_cnt = 0

        if len(track_list) == 1:
            return track_cnt + track_list[0]

        for i in range(len(track_list) - 1):
            if track_list[i + 1] <= track_list[i]:
                track_cnt += track_list[i]
        track_cnt += track_list[-1]
        return track_cnt
