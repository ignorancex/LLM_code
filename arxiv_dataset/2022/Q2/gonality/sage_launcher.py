#!/usr/local/bin/python

import argparse, math, pathlib
import subprocess, sys, time

walltime = time.time()

msg = 'Sage Script Launcher:  '
msg += 'This is a simple launcher for running Sage scripts on multiple '
msg += 'nodes of a machine. The idea is that one wants to use Sage to '
msg += 'do some kind of mathematical operation on elements of a search space, and '
msg += 'that each element of this space can be processed '
msg += 'independently of the others. We break the search space ' 
msg += 'into "jobs" and distribute one job to each of a certain number '
msg += 'of Sage instances. Each Sage instance will process the elements '
msg += 'of its job and write its output to disk. Thus, an appropriate '
msg += 'Sage script must accept three arguments:'
msg += '  "num_jobs", "job", and "filename". '
msg += 'Additional required arguments can come afterward in the Sage script, '
msg += 'and these are specified with the -e flag in sage_launcher.py. '
msg += 'If any job returns a code other than 0, all jobs are immediately stopped.'

# Parse some arguments
parser = argparse.ArgumentParser(description=msg)
parser.add_argument('sage_script', type=str, help='relative path to Sage script')
parser.add_argument('log_directory', type=str, help='relative path to log/data directory')
parser.add_argument('num_jobs', type=int, help='number of Sage instances to run')
parser.add_argument('-c','--cat', action='store_true', help='cat outfiles together')
parser.add_argument('-l','--logfile', type=str, help='logfile prefix (default: log)')
parser.add_argument('-o','--outfile', type=str, help='outfile prefix (default: out)')
parser.add_argument('-s','--sage', type=str, help='command to run sage (default: sage)')
parser.add_argument('-e','--extra_args', type=str, help='Additional arguments expected by the Sage script')
args = parser.parse_args()

# Rename arguments
script_name = args.sage_script
data_dir = args.log_directory
num_jobs = args.num_jobs
sage_cmd = args.sage if args.sage else 'sage'
out_prefix = args.outfile if args.outfile else 'out'
log_prefix = args.logfile if args.logfile else 'log'

if num_jobs <= 0:
  raise ValueError('Invalid number of jobs: {}'.format(num_jobs))

# Grab all extra arguments for sage script
if args.extra_args is None:
  extra_args = []
else:
  extra_args = args.extra_args.split(' ')

# Prepare for time printing
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

# Look for sage script
script_path = pathlib.Path(script_name)
if not script_path.is_absolute():
  script_path = pathlib.Path.cwd() / script_path
if not script_path.exists():
  raise OSError('Script does not exist: {}'.format(script_name))
if not script_path.is_file():
  raise OSError('script_name is not a file: {}'.format(script_name))  

# Look for data directory; make it if it doesn't exist.
data_path = pathlib.Path(data_dir)
if not data_path.is_absolute():
  data_path = pathlib.Path.cwd() / data_path
while not data_path.exists():
  data_path.mkdir()
  time.sleep(.1) # avoid starting jobs running before this is created

# Start the jobs!
print('Starting {} Sage jobs ... '.format(num_jobs), end='')
running_jobs = []
for job in range(num_jobs):
  log_file = data_path / (log_prefix + '.{}'.format(job))
  out_file = data_path / (out_prefix + '.{}'.format(job))
  cmd = [sage_cmd, str(script_path), str(num_jobs), str(job), str(out_file)]
  cmd += extra_args
  fp = open(log_file,'w')
  proc = subprocess.Popen(cmd,stdout=fp,stderr=subprocess.STDOUT)
  running_jobs.append((proc,fp,job))
print('Go!')
print('\nWaiting for jobs to finish ...')

# Prepare finishing message: e.g., 'Finished: 15, job id: 12'
num_field = str(len(str(num_jobs)))
finish_msg = 'Finished: {:' + num_field + 'd},'
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

walltime = time.time() - walltime

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
print('Wall time: {}'.format(format_time(walltime)))
