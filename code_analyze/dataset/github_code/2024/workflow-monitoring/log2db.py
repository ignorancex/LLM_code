import sqlite3
import sys

# takes csv log on stdin and puts it into the sqlite db specified in argument 1
# example usage: cat log.csv | python3 log2db.py log.db

conn = sqlite3.connect(sys.argv[1])
c = conn.cursor()

c.execute('PRAGMA synchronous = 0;')
c.execute('PRAGMA journal_mode = OFF;')

c.execute('CREATE TABLE IF NOT EXISTS event (\
    time_start UNSIGNED INT64,\
    time_end UNSIGNED INT64,\
    pid UNSIGNED INT32,\
    utime_start UNSIGNED INT64,\
    utime_end UNSIGNED INT64,\
    stime_start UNSIGNED INT64,\
    stime_end UNSIGNED INT64,\
    inode_uid UNSIGNED INT64,\
    type CHAR,\
    result INT64,\
    handle_uid UNSIGNED INT64,\
    offset UNSIGNED INT64,\
    size UNSIGNED INT64,\
    flags UNSIGNED INT64,\
    path TEXT\
)')

line_num = 0

for line in sys.stdin:
    line_num += 1
    if line_num % 10000 == 0:
        print("\rLine", line_num, end='')
    line = line.rstrip()
    fields = [f.strip().replace('.', '') if i < 14 else f.strip() for (i, f) in enumerate(line.split(','))]
    fields[14] = '_'.join(fields[14:])
    while len(fields) > 15:
        fields.pop()

    try:
        c.execute('INSERT INTO event VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)', fields)
    except Exception as e:
        print("\nException in line ", line_num, ': ', e, "\nLine: ", line, sep='')

c.close()
conn.commit()
conn.close()

if line_num >= 10000:
    print('')

print('Processed lines / events:', line_num)
