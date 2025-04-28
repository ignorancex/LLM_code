import os

#from matplotlib.backends.backend_pdf import PdfPages
#from matplotlib.pyplot import step, legend, xlim, ylim, show
#from numpy import ma
import aplpy
import astropy.io.fits
import matplotlib.pylab as plt
import numpy as np
#import scipy.optimize as optimize

#import matplotlib as mpl
import kep_rotation_maxvel_dom as kmaxvel

#mpl.rcParams['xtick.direction'] = 'in'
#mpl.rcParams['ytick.direction'] = 'in'

def main():
  mol='h2co' # name of your molecule, just for displaying purposes
  sname = 'B5-IRS1' # name of your source, just for displaying purposes
  pa = 67.1+90 # 0 is north?
  pvmap_name='cleaned_images/B5IRS1_C18O_robust05_multi_K_PV_157.1 degdeg_12arcsec_0.44width_cont_centered_arcsec.fits' # if your offset units is not arcseconds this will not work.

  dist=302. #pc

  velocity_units_kms=False #velocity units of your pv map, if False assumes is m/s

  rms= 1.0 # noise of your pv map in the units of your pv map, usually Jy/b, but in this case is K

  chan_width = 41.6667 # depends on the units of your pv map, in my case was m/s, this is for plotting errors only

  mass=0.1 # initial guess mass in msun

  vsys=10200. #initial guess vsys, depends on the units of your pv map, in my case was m/s

  #limits to the values that vsys can take in case you are also fitting for this parameeter
  vsysmin=9000.
  vsysmax=11000.

  # if True, this is considered a free parameter of the fit.
  vary_vsys=False
  vary_mass=True

   # range in offsets (arcseconds usually, that you want to fit)
  offrange_side1=[-3.,-0.4] # blue-shifted side
  offrange_side2=[0.4,3.] # reds-shifted side

  rms_level=10.# choose the s/n level to find the points to fit the kep curve

  # range of positive offsets where resultant kep curve will be plot
  min_offset_kepcurve=0.02 # usually in arsecond, don't write 0 because kep curve is undefined there, just something small
  max_offset_kepcurve=3 # this will also be used to draw the horizontal v=vsys line

  vertical_line_limits=[7000.,13000.] # depends on the units of your pv map, in my case was m/s

  [maxvel_array,maxoff_array]=kmaxvel.maxvel(pvmap_name,rms*rms_level,'me',offrange_side1)
  [maxvel_array2,maxoff_array2]=kmaxvel.maxvel(pvmap_name,rms*rms_level,'me',offrange_side2)

  maxvel_array_both=np.concatenate([maxvel_array,maxvel_array2])
  maxoff_array_both=np.concatenate([maxoff_array,maxoff_array2])

  if velocity_units_kms:
    fit_results=kmaxvel.kepfit_maxvel(maxvel_array_both*1e3,maxoff_array_both,[vsys*1e3,vary_vsys],[mass,vary_mass],dist,vsysmin*1e3,vsysmax*1e3)

  else:
    fit_results=kmaxvel.kepfit_maxvel(maxvel_array_both,maxoff_array_both,[vsys,vary_vsys],[mass,vary_mass],dist,vsysmin,vsysmax)

  print ('mass (msun) +/- err: ',fit_results[2], fit_results[3])
  print ('vsys (km/s)  +/- err: ',fit_results[0], fit_results[1])

  offset_arr=np.arange(min_offset_kepcurve,max_offset_kepcurve,0.01)

  vel_kep=kmaxvel.keprot(offset_arr,fit_results[2],dist)
  sign=maxvel_array[0]-fit_results[0]
  sign=sign/np.abs(sign)

  if sign<0.:
    color1='blue'
    color2='red'
  else:
    color1='red'
    color2='blue'

  fig1=plt.figure()
  ax = plt.axes()
  ax.set_xscale("log")
  ax.set_yscale("log")

  offset_arr=sign*offset_arr
  offset_arr_fit=np.arange(min(np.min(np.abs(maxoff_array)),np.min(np.abs(maxoff_array2)))-0.01,max(np.max(np.abs(maxoff_array)),np.max(np.abs(maxoff_array2)))+0.1,0.01)
  vel_kep_fit=kmaxvel.keprot(offset_arr_fit,fit_results[2],dist)

  if velocity_units_kms:
    ax.errorbar(np.abs(maxoff_array),np.abs(maxvel_array-fit_results[0]*1e-3),yerr=chan_width,linestyle='none',marker='.',color=color1)
    ax.errorbar(np.abs(maxoff_array2),np.abs(maxvel_array2-fit_results[0]*1e-3),yerr=chan_width,linestyle='none',marker='.',color=color2)
    plt.plot(offset_arr_fit,vel_kep_fit*1e-3,color='black')

    list_kep=[np.array([
      offset_arr,
      (fit_results[0]+sign*vel_kep)*1e-3])]

    list_kep2=[np.array([
        -offset_arr,
        (fit_results[0]-sign*vel_kep)*1e-3])]
    vsys=fit_results[0]*1e-3

  else:
    ax.errorbar(np.abs(maxoff_array),np.abs(maxvel_array-fit_results[0]),yerr=chan_width,linestyle='none',marker='.',color=color1)
    ax.errorbar(np.abs(maxoff_array2),np.abs(maxvel_array2-fit_results[0]),yerr=chan_width,linestyle='none',marker='.',color=color2)
    plt.plot(offset_arr_fit,vel_kep_fit,color='black')

    list_kep=[np.array([
      offset_arr,
      fit_results[0]+sign*vel_kep])]

    list_kep2=[np.array([
        -offset_arr,
        fit_results[0]-sign*vel_kep])]
    vsys=fit_results[0]

  fig=plt.figure()
  gc1=aplpy.FITSFigure(pvmap_name,dimensions=[0,1],figure=fig,hdu=0)
  gc1.show_colorscale(cmap='YlGnBu',aspect='auto',interpolation='nearest',vmin=0)#,vmax=vmax)

  gc1.show_lines(list_kep,color='black')
  gc1.show_lines(list_kep2,color='black')

  gc1.show_markers(maxoff_array,maxvel_array,facecolor='black',edgecolor='black',marker='+')
  gc1.show_markers(maxoff_array2,maxvel_array2,facecolor='black',edgecolor='black',marker='+')

  list_hp2=[]

  list_hp2.append(np.array([
      [-max_offset_kepcurve,max_offset_kepcurve],
      [vsys,vsys],]))
  list_hp2.append(np.array([
      [0,0],
      [vertical_line_limits[0],vertical_line_limits[1]],]))

  gc1.show_lines(list_hp2,color='black')

  gc1.ticks.show()
  gc1.ticks.set_color('black')
  gc1.ticks.set_length(10)
  vf=fit_results[0]#/1e3
  vfe=fit_results[1]#/1e3
  # plt.title(sname+' '+mol+'  Mass %.2f'%fit_results[2]+' +/- %.2f'%fit_results[3]+' Msun'+' Vsys %.2f'%vf+' +/- %.2f'%vfe+' m/s')#+'+/-'+str(round(fit_results[3],2))+'Msun')
  plt.show()

if __name__=='__main__':
  main()
