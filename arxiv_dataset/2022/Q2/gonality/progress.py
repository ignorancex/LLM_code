#!/usr/local/bin/python

import math, sys, time

class Progress(object):
  r"""
  Rudimentary progress indicator for loops

  INPUT:

  - ``num_steps`` -- positive integer; total number of steps in the loop

  - ``steps_until_print`` -- positive integer (default: num_steps / 100); number
    of passes through loop before a percentage is printed

  """
  def __init__(self, num_steps, steps_until_print=None):

    if steps_until_print is None:
      steps_until_print = max(num_steps / 100,1)
    self._num_steps = num_steps
    self._steps_until_print = steps_until_print
    self._current_step = 0
    self._printed = 0
    self._steps_since_print = 0
    self._start_time = time.time()
    self._print_time = self._start_time
    self._profile = {}
    self._profile_order = []
    self._profile_time_stamp = self._start_time

  def __call__(self, steps=1):
    r"""
    Alert self that we have taken a step and print updates as needed

    INPUT:

    - ``steps`` -- int (default: 1); specifies the number of steps taken by this
      particular call (for giant-stepping through a search space)

    """
    self._current_step += steps
    self._steps_since_print += steps
    i = self._current_step
    steps = self._num_steps
    if self._steps_since_print >= self._steps_until_print:
      sys.stdout.write('{:5.2f}% '.format( float(i * 100.0 / steps)))
      sys.stdout.flush()
      self._printed += 1
      self._steps_since_print = 0
    if self._printed == 10:
      t = time.time()
      print_time = t - self._print_time
      time_msg = self.format_time(print_time)
      sys.stdout.write(time_msg + '\n')
      sys.stdout.flush()        
      self._printed = 0
      self._print_time = t      

  def profile(self, name):
    r"""
    Store information about a particular process

    INPUT:

    - ``name`` -- string

    """
    if name == 'Total time':
      raise ValueError('String "Total time" is reserved')
    t = time.time()
    D = self._profile
    if name in D:
      D[name] += t - self._profile_time_stamp
    else:
      D[name] = t - self._profile_time_stamp
      self._profile_order.append(name)
    self._profile_time_stamp = t

  def format_time(self,t):  
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
    
    
  def finalize(self):
    r"""
    Finish and write elapsed time
    """
    sys.stdout.write('\n')
    self._profile['Total time'] = time.time() - self._start_time
    self._profile_order.append('Total time')
    width = max(len(c) for c in self._profile_order)
    msg = '{:>' + str(width) + '}: {}\n'
    for name in self._profile_order:
      time_str = self.format_time(self._profile[name])
      sys.stdout.write(msg.format(name,time_str))
    sys.stdout.write('\n')
    sys.stdout.flush()
