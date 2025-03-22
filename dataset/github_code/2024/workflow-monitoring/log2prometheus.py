import prometheus_client
import sys
import time

# field indices
PID = 2
INODE = 7
OPERATION = 8
RESULT = 9
FILEHANDLE = 10
OFFSET = 11
SIZE = 12
PATH = 14

line_num = 0 # current progress
file_metrics = dict() # maps handle IDs to the prometheus counter instances, counting the reads / writes per file
totals = prometheus_client.Counter('totals', 'All operation sizes.', ['operation'])

prometheus_client.start_http_server(8000)

for line in sys.stdin:
    #time.sleep(0.5)
    line_num += 1
    if line_num % 10000 == 0:
        print("\rLine", line_num, end='')
    line = line.rstrip()
    fields = [f.strip().replace('.', '') if i < 14 else f.strip() for (i, f) in enumerate(line.split(','))]
    fields[14] = '_'.join(fields[14:])
    while len(fields) > 15:
        fields.pop()

    if fields[OPERATION] == 'O':
        file_metrics[fields[FILEHANDLE]] = {
            'opsizes': prometheus_client.Counter('opsizes_file_' + fields[FILEHANDLE], 'inode: ' + fields[INODE] + ', path: ' + fields[PATH], ['operation']),
            'offsets': prometheus_client.Gauge('offsets_file_' + fields[FILEHANDLE], 'inode: ' + fields[INODE] + ', path: ' + fields[PATH], ['operation']),
        }
    elif fields[OPERATION] == 'C':
        del file_metrics[fields[FILEHANDLE]]
    elif fields[OPERATION] == 'R' or fields[OPERATION] == 'W':
        if (int(fields[RESULT]) > 0):
            file_metrics[fields[FILEHANDLE]]['offsets'].labels(operation=fields[OPERATION]).set(int(fields[OFFSET]) + int(fields[RESULT]))
            file_metrics[fields[FILEHANDLE]]['opsizes'].labels(operation=fields[OPERATION]).inc(int(fields[RESULT]))
            totals.labels(operation=fields[OPERATION]).inc(int(fields[RESULT]))

if line_num >= 10000:
    print('')

print('Processed lines / events:', line_num)

while True:
    time.sleep(10)
