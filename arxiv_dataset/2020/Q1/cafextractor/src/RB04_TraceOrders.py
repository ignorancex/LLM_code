import numpy as np
import copy
from termcolor import colored

import GLOBALutils
import CAFEx_SetupFile as CS
import CAFEutilities


# ==============================
#		Order Tracing
# ==============================
def TraceOrders(Flats):
	order = CS.order_trace_degree
	ampli = CS.order_aperture_ampl
	o_coeff = []
	Nord = []
	for i,Flat in enumerate(Flats):
		o_coeff_tmp,Nord_tmp, rms_all = GLOBALutils.get_them(Flat,ampli,order)
		o_coeff.append(o_coeff_tmp)
		Nord.append(Nord_tmp)
		print "    --> MasterFlat #"+np.str(i+1)+": "+np.str(Nord[i])+" orders found with mean rms = "+np.str(np.round(np.mean(rms_all),decimals=3))+" pix."
	return o_coeff,Nord

# ==============================
#		Order Checking
# ==============================
def SelectOrders(o_coeff,Nord, yshift, cv, y0Nominal_first = None):
	nx, ny = 2048, 2048
	CS.var.set_OrderProp(CAFEutilities.jdnight(cv.night))
	
	# ===== Define nominal number of orders and First order from CAFEx_SetupFile
	if y0Nominal_first == None:
		y0Nominal_first = CS.var.y0Nominal_first #+ yshift
	else:
		y0Nominal_first = y0Nominal_first #+ yshift
	Nominal_Nord = np.int(CS.var.Nominal_Nord)
	
	# ===== Initialize arrays to update the number of orders and trace coefficients
	new_Nord = []
	new_o_coeff = [] 
	for i in range(len(Nord)):
		new_o_coeff.append(np.zeros((Nominal_Nord,o_coeff[0].shape[1])) )
		new_Nord.append(0)
	
	# ===== Select the orders from the first nominal order
	for ii in range(len(o_coeff)):	
		
		# ===== Determine the location of the order at the central column
		y0 = []
		for jj in range(Nominal_Nord):
			p0 = np.poly1d(o_coeff[ii][jj,:])
			y0.append(p0(CS.var.x0_first)) #
	
		y0 = np.array(y0)
	
		# ===== Find the first order (the closest to the Nominal first order)
		elem = CAFEutilities.get_closer_frame(y0Nominal_first,y0)
		first = elem[0] 
		Y_of_first_order = y0[first]
		print "    --> First order is order #"+np.str(first)+" @ y = "+ np.str(np.round(Y_of_first_order,decimals=3))+" pix."
		# ===== Difference against nominal location
		diff_ref_flat = y0[first] - y0Nominal_first
		print "    --> delta_Y with ref. flat location = "+np.str(np.round(diff_ref_flat,decimals=3))+" pix."
		
		# ===== Warning message in case of large difference between current and nominal position
		if np.abs(diff_ref_flat) > 3.:
			print colored("    --> WARNING: Large difference between ref. and current flat in the first ",'red')
			print colored("                 detected order. Please check!","red")
		
		# ===== Select the "Nominal_Nord" orders from the first order.
		new_o_coeff[ii] = o_coeff[ii][first:first+Nominal_Nord,:]
		new_Nord[ii] = Nominal_Nord #Nord-first
		if np.shape(o_coeff[ii])[0]-first < Nominal_Nord:
			Norders_missing = np.int(Nominal_Nord - (np.shape(o_coeff[ii])[0]-first))
			print "    --> WARNING: Less orders than Nominal. Adding "+str(Norders_missing)+" orders artificially"
			new_o_coeff[ii] = np.concatenate((new_o_coeff[ii], np.zeros((Norders_missing,np.shape(o_coeff[ii])[1]))))
		
		
		
		#print first,first+Nominal_Nord
		print np.shape(new_o_coeff[ii])
	
	return new_o_coeff,new_Nord, Y_of_first_order
	
	
	
	
	
	