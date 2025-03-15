# decorator to make a gif from a function

import itertools
import numpy as np
import matplotlib.pyplot as plt
import imageio

def gif(save='gif.gif',time=1.0,cla=True,verbose=False,**kwargs):
  # { gif
  def decorator(fun):
    # { decorator
    def kwiter(**kwargs):
      # { kwiter
      def flatten(obj):
        # { flatten
        out = []
        if hasattr(obj,'__iter__') and not isinstance(obj,str):
          for el in obj:
            out.extend(flatten(el))
        else:
          out.append(obj)
        return out
        # flatten }
      # { kwiter
      keys, vals = zip(*kwargs.items())
      vals = [flatten(val) for val in vals]
      lens = list(map(len,vals))
      vals = [val*int(max(lens)/vlen) for val,vlen in zip(vals,lens)]
      for ivals in zip(*vals):
        yield {k:v for k,v in zip(keys, ivals)}
      # kwiter }
    # { decorator
    frames = []
    for i,kwset in enumerate(kwiter(**kwargs)):
      if verbose: print(i,flush=True)
      if cla: plt.cla()
      # call function for each iter
      fun(**kwset)
      # save the plot data
      fig = plt.gcf()
      fig.canvas.draw()
      frame = np.frombuffer(fig.canvas.tostring_rgb(),dtype='uint8')
      frames.append(frame.reshape(fig.canvas.get_width_height()[::-1] + (3,)))
    # write the result to file
    imageio.mimsave(save,frames,fps=(i+1)/time)
  return decorator

