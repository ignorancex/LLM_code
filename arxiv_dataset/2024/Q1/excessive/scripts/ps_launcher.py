#!/usr/local/bin/python

import argparse, math, pathlib
import subprocess, sys, time

from argparse import RawTextHelpFormatter

walltime = time.time()

msg  = 'Plane Curve Search Launcher:  This is a simple python launcher for running\n'
msg += 'the plane_search.c and genus7_plane_search.c apps. The search space is broken\n'
msg += 'up into tiles and can naturally be run independently on multiple nodes of\n'
msg += 'a machine.\n\n'
msg += 'We break the search space into "jobs" and distribute one job to each of a\n'
msg += 'certain number of instances. Each instance will process the elements\n'
msg += 'of its job and write its output to disk. If any job returns a code other\n'
msg += 'than 0, all jobs are immediately stopped.'

# Parse some arguments
parser = argparse.ArgumentParser(description=msg,formatter_class=RawTextHelpFormatter)
parser.add_argument('app_name', type=str, help='relative path to plane_search or genus7_plane_search executable')
parser.add_argument('log_directory', type=str, help='relative path to log/data directory')
parser.add_argument('num_tiles', type=int, help='total number of tiles to split search into')
parser.add_argument('genus_or_case', type=int, help='target genus for plane_search or target case for genus7_plane_search')
parser.add_argument('-s','--start', type=int, help='index of start tile')
parser.add_argument('-t','--stop', type=int, help='index of stop tile')
parser.add_argument('-c','--cat', action='store_true', help='cat outfiles together')
parser.add_argument('-l','--logfile', type=str, help='logfile prefix (default: log)')
parser.add_argument('-o','--outfile', type=str, help='outfile prefix (default: out)')
args = parser.parse_args()

# Rename arguments
app_name = args.app_name
data_dir = args.log_directory
genus_or_case = args.genus_or_case
num_tiles = args.num_tiles
start = args.start if args.start else 0
stop = args.stop if args.stop else num_tiles
out_prefix = args.outfile if args.outfile else 'out'
log_prefix = args.logfile if args.logfile else 'log'

num_jobs = stop-start
if num_jobs <= 0:
  raise ValueError('Invalid number of jobs: {}'.format(num_jobs))

# Look for executable
app_path = pathlib.Path(app_name)
if not app_path.is_absolute():
  app_path = pathlib.Path.cwd() / app_path
if not app_path.exists():
  raise OSError('Executable does not exist: {}'.format(app_name))
if not app_path.is_file():
  raise OSError('app_name is not a file: {}'.format(app_name))  

# Look for data directory; make it if it doesn't exist.
data_path = pathlib.Path(data_dir)
if not data_path.is_absolute():
  data_path = pathlib.Path.cwd() / data_path
while not data_path.exists():
  data_path.mkdir()
  time.sleep(.1) # avoid starting jobs running before this is created

# Start the jobs!
print('Starting {} plane curve search jobs ... '.format(num_jobs), end='')
running_jobs = []
for job in range(start,stop):
  log_file = data_path / (log_prefix + '.{}'.format(job))
  out_file = data_path / (out_prefix + '.{}'.format(job))
  cmd = [str(app_path), str(genus_or_case), str(out_file), str(num_tiles), str(job)]
  fp = open(log_file,'w')
  proc = subprocess.Popen(cmd,stdout=fp,stderr=subprocess.STDOUT)
  running_jobs.append((proc,fp,job))
print('Go!')
print('\nWaiting for jobs to finish ...')

# Prepare finishing message: e.g., 'Finished: 15, job id: 12'
job_field = str(len(str(num_jobs)))
num_field = str(len(str(stop)))
finish_msg = 'Finished: {:' + job_field + 'd},'
finish_msg += ' job id: {:' + num_field + 'd},'
finish_msg += ' {}'

# Wait for jobs to finish!
finished = False
while running_jobs and not finished:
  for proc,log_file,job_id in running_jobs:
    if finished:
      running_jobs.remove((proc,log_file,job_id))
      log_file.close()
      proc.terminate()
      continue
    retcode = proc.poll()
    if retcode is not None:
      running_jobs.remove((proc,log_file,job_id))
      log_file.close()
      print(finish_msg.format(num_jobs-len(running_jobs),job_id,time.strftime('%c')))
      if retcode != 0:
        print('  Return code: {}'.format(retcode))
        finished = True
      break
    time.sleep(.1)
    log_file.flush()

print('Finished compute!')
time.sleep(.5)

if args.cat:
  print('Catting results ... ', end='')
  outfiles = data_path / (out_prefix + '.*')
  catfile = data_path / out_prefix
  cmd = 'cat ' + str(outfiles) + ' > ' + str(catfile)
  subprocess.Popen(cmd,shell=True)
  print('Done!')

################################################################################

def format_time(t):
  t = t*1.0
  hours = int(math.floor(t / 3600))
  t -= 3600*hours
  mins  = int(math.floor(t / 60))
  t -= 60*mins
  print_time = ''
  if hours > 0:    
    print_time += '{}h '.format(hours)
    print_time += '{}m '.format(mins)
    print_time += '{:.2f}s'.format(t)
  elif mins > 0:
    print_time += '{}m '.format(mins)
    print_time += '{:.2f}s'.format(t)
  else:
    print_time += '{:.2f}s'.format(t)    
  return print_time

walltime = time.time() - walltime
print('Wall time: {}'.format(format_time(walltime)))
