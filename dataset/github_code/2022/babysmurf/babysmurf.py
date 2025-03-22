import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import scipy.signal

def make_measured_signal(tvec,sig,squid,noise=None,reset_rate=1.e4,nphi0=4,fsamp=2.4e6):
   """
   Take the detector signal and produce a modulated resonance frequency. 
   This is only the bottom of the resonance!

   Args:
   tvec: time vector of signal. units = seconds
   sig: signal vector, of same size as time vector. units = radians
   squid: single period of SQUID curve, units = Hz (of resonance frequency) vs x-axis of 0 to 2pi. 
   noise: noise vector, of same size as MODULATED signal. units = ?; defaults None
   reset_rate: flux ramp sawtooth reset rate. units = Hz, defaults 10kHz
   nphi0: number of phi0 per ramp, need not be integer. units = none, defaults 4
   fsamp = per-channel sample rate of SMuRF. units = Hz, defualts 2.4MHz

   Returns:
   meas (delta(t)/reset_rate * frame_size) x 1: SQUID modulated signal. units = Hz, frame_size = fsamp / reset_rate
   tt (length(tvec) * frame_size) x 1: time vector of meas; this is sampled at fsamp. units = Hz
   ph (delta(t) * reset_rate) x 1: resampled signal at one point per flux ramp frame. mostly useful for plotting. 
   """

   framesize = int(fsamp / reset_rate)
   freqnorm = nphi0 * reset_rate / fsamp # normalized phi0 frequency
   nframes = np.floor((tvec[-1] - tvec[0]) / (1/reset_rate)) # enough frames to fully sample the signal at the reset rate

   obs_n = np.zeros((int(framesize * nframes), 1)) # preallocate the observed signal
   sig_sampled = np.interp(np.arange(nframes)/reset_rate,np.ravel(tvec),np.ravel(sig)) # resample the signal to guarantee the right cadence

   # construct the SQUID signal
   # interpolate SQUID curve to the correct number of points
   # upsample squid,sig to fsamp rate and offset that, which gives finer resolution
   squidint = np.interp(np.linspace(0,1,int(fsamp/nphi0),endpoint=False),np.linspace(0,1,int(len(squid)),endpoint=False),squid)
   sigmod = np.matlib.repmat(np.reshape(np.ravel(squidint),(int(fsamp/nphi0),1)),nphi0,1)


   for ii in np.arange(nframes):
     ii = int(ii)
     poff = sig_sampled[ii] # get the single point of phase that offsets the entire frame
     fracperiod = poff / (2*np.pi) # convert this phase offset to fraction of a period
     #print(fracperiod)
     sigmod2 = np.roll(sigmod, 1*int(np.round(fracperiod * len(squidint)))) # shift the SQUID curve by the desired fraction of a period
     # downsample again
     obs_n[framesize*ii:framesize*(ii+1),:] = np.reshape(sigmod2[::int(1*reset_rate)],(framesize,1))


   # fill everything back in
   meas = obs_n
   tt = np.arange(len(meas)) / fsamp
   ph = sig_sampled

   return meas, tt, ph


def make_squid_curve(lda,amp,npts=600):
   """
   Make a SQUID curve given a lambda and amplitude. Returns a single period. 
   Some work will have to be done to resample this into the correct dimensions.
   This should be thought of as returning a frequency (in Hz) vs an x-axis of 0 to 2pi Phi0.
   
   Args:
   lda: SQUID curve lambda, we usually prefer chips that have lambda in the neighborhood of 0.33. Higher values are less sinusoidal. 
   amp: amplitude of the curve, units = Hz
   npts: number of points, defaults 600

   Returns:
   squid: a single period of the SQUID curve. May need to resample to get it to work with other functions. units = Hz
   """
   framesize = int(npts)
   freqnorm = 1 / framesize
   sigsin = np.sin(2*np.pi*freqnorm*(np.arange(framesize)))
   sigmod = (lda*np.transpose(sigsin)) / (1 + lda*np.transpose(sigsin))
   
   sigrange = np.max(sigmod) - np.min(sigmod)
   squid = (sigmod - np.mean(sigmod)) * amp/sigrange # center and scale

   return squid

def make_s21_measurement(fvec,resfreq,q,qc):
   """
   Make an S21 vs time matrix given the resonator S21 and its resonance frequency movement vs time. 
   No resonator/SQUID physics here! This just perturbs the S21 back and forth to the resonance frequency. 

   Args: 
   fvec: frequency points at which response is sampled, units = Hz
   resfreq: resonance frequency vs time, doesn't actually matter the sample rate
   q: quality factor, unitless
   qc: coupling quality factor, unitless, allowed to be complex to allow for asymmetry

   Returns:
   s21matrixmag, s21matrixphase: complex resonator response vs time, for passing into 
   """

   s21matrixmag = np.zeros((len(fvec),len(resfreq)))
   s21matrixphase = np.zeros((len(fvec),len(resfreq)))

   for ii in np.arange(len(resfreq)): # is there a way to vectorize this?
      s21matrixmag[:,ii],s21matrixphase[:,ii] = make_res_s21(resfreq[ii],fvec,q,qc)

   return s21matrixmag,s21matrixphase
   

def add_noise(sig,noise_f,noise_spec,fsamp=2.4e6):
   """
   Add some noise on top of a signal before passing to demodulation. 
   This should be thought of as a frequency noise, so SQUID + resonator effects but not detector effects (those go straight into signal!)
   
   Args:
   sig: input signal to make noisy
   noise_f: frequencies at which spectrum (below) is sampled
   noise_spec: single-sided noise power spectral density to sample from; (n_points x 1), noise psd in Hz/rtHz
   fsamp: sample rate of sig, units = Hz, defaults 2.4MHz

   Returns: 
   noisy_sig: same size as input signal, but with added noise. Still in units of resonance frequency [Hz]
   """

   noise = gen_noise_tod(noise_spec,noise_f,len(sig),fsamp)

   noisy_sig = np.ravel(noise) + np.ravel(sig)
   return noisy_sig 

def gen_noise_tod(noisespec,noisef,nsamp,fsamp=2.4e6):
   """
   Generate a timestream of noise given the power spectral density

   Args:
   noisespec: frequency noise psd, units Hz/rtHz
   noisef: same size as noisespec; the frequency points at which the noise psd is sampled.  Must include frequencies from 0 Hz all the way
           to the Nyquist frequency or the input to the inverse FFT will be poorly interpolated!
   nsamp: number of samples to generate in the timestream
   fsamp: sample rate, defaults 2.4MHz
   
   Returns:
   tod: nsamp x 1
   """
   ff = np.fft.rfftfreq(n=nsamp,d=1./fsamp) # generate full frequency array
   speci = npinterp(ff,noisef,noisespec) # interpolate psd to correct shape
   speci = speci * np.exp(1j*np.random.rand(len(speci))*2*np.pi)
   speci[0] = np.real(speci[0]) # force DC term to be real
   tod = np.fft.irfft(speci,n=nsamp,norm="forward") * np.sqrt(fsamp/(2*nsamp))
   return tod

def lms_fit_demod_track(respmag,respphase,fvec,eta,reset_rate=10.e3,nphi0=4,gain=1/32,blank=[0,1],fsamp=2.4e6,nharm=3):
   """
   The main SMuRF demodulation loop. Assumes that you started tracking perfectly, ie that the probe tone begins at the bottom of the resonance dip. 

   Args:
   resp:* the resonator (complex) S21 vs time. This is what the resonator is ACTUALLY doing at each timestep sampled at fsamp. (nframes*fsamp/reset_rate x length(fvec), where nframes = Delta(t) in seconds * reset_rate). 
   fvec: frequencies at which the resonator response above is sampled. 
   eta: eta calibration to apply for this resonance. A complex number, from estimate_eta
   reset_rate: flux ramp sawtooth reset rate. Units = Hz, defaults 10kHz
   nphi0: number of phi0 per flux ramp, unitless, defaults 4
   gain: feedback gain parameter, unitless, defaults 1/32. I need to figure out how to map this onto lms_gain in pysmurf.
   blank: start and end fractions of the flux ramp frame to use for phase estimation. unitless, defaults [0,1]
   fsamp: sample rate, units = Hz, defaults 2.4MHz
   nharm: number of harmonics to use in estimation, defaults 3. Currently SMuRF supports nharm <= 3; future iterations could use more.


   Returns:
   demod_sig: (nframes x 1), sampled at the reset rate. The demodulated phase of the first harmonic, units = radians
   phases: (length(measurement) x nharm), sampled at fsamp. The phases of each harmonic estimate in real time, units = radians
   yhat: (same size as measurement), sampled at fsamp rate. The estimated resonance frequency, aka the probe tone frequency applied. 
   errs: (same size as measurement), sampled at fsamp rate. The errors between the tracked frequency and estimate, units = Hz. 
   alphas: (length(measurement) x (2*nharm+1)), sampled at fsamp. The alpha coefficients at each timestep. 
   """
   framesize = int(fsamp / reset_rate) # number of time samples per flux ramp frame
   nframes = int(np.shape(respmag)[0] // framesize) # number of full flux ramp frames in the real timestream

   H = make_obs_mat(reset_rate,nphi0,fsamp,nharm) # construct observation coefficient matrix

   alpha = np.ones((2*nharm+1,)) # initialize coefficient vector alpha
   yhat = np.zeros((framesize*nframes,1)) # initialize frequency estimate vector yhat
   # estimate will start at center frequency of first timestep
   alphas = np.zeros((framesize*nframes,2*nharm+1))
   phases = np.zeros((framesize*nframes,nharm)) # initialize phase estimates of harmonics
   errs = np.zeros((framesize*nframes,1)) # initialize measurement vector, this is the frequency errors that we are trying to minimize

   for ii in range(framesize*nframes):
      blankmin = np.round(framesize*blank[0]) # get where in frame to start
      blankmax = np.round(framesize*blank[1]) # ...and where to stop

      rowno = np.mod(ii+framesize,framesize) # figure out where in the frame we are
      h = H[rowno,:] # get the harmonic components of this frame
      yhat[ii] = np.matmul(h,alpha) # estimate is based on existing alpha
      # this proceedings Eqs. 8 and 9.

      if ii==0:
         yhat[ii] = 0 # initialize at the center of subband; this could be changed later.

      centerfreq = estimate_res_freq(fvec,respmag[0,:],respphase[0,:])[0]
      probetone = (centerfreq - yhat[ii])[0]
      actualfreq = estimate_res_freq(fvec,respmag[ii,:],respphase[ii,:])[0]
      # estimate the frequency error
      errs[ii] = estimate_freq_error(fvec,respmag[ii,:],respphase[ii,:],eta,probetone)
 
      if np.logical_or(rowno < blankmin, rowno > blankmax):
         alpha = alpha # in the blankoff region, do not update the coefficients
      else:
         # update the coefficients for the next point (proceedings Eq. 10)
         alpha = alpha + gain*errs[ii]*np.transpose(h) / (np.sum(np.power(h,2))) # h^2 is a normalization factor

      # save the coefficients for inspection
      alphas[ii,:] = alpha
      # save the phase estimate for each harmonic
      for jj in range(nharm):
         phases[ii,jj] = np.arctan2(alpha[2*jj+1],alpha[jj])

   demod = get_frame_averages(np.unwrap(phases[:,0]),reset_rate,fsamp)

   return demod, phases, yhat, errs, alphas 

def lms_fit_demod_untrack(respmag,respphase,fvec,eta,reset_rate=10.e3,nphi0=4,gain=1/32,fsamp=2.4e6,nharm=3):
   """
   Same as above, lms_fit_demod_untrack, but the probe tone doesn't move....

   Args:
   resp:* the resonator (complex) S21 vs time. This is what the resonator is ACTUALLY doing at each timestep sampled at fsamp. (nframes*fsamp/reset_rate x length(fvec), where nframes = Delta(t) in seconds * reset_rate). 
   fvec: frequencies at which the resonator response above is sampled. 
   eta: eta calibration to apply for this resonance. A complex number, from estimate_eta
   reset_rate: flux ramp sawtooth reset rate. Units = Hz, defaults 10kHz
   nphi0: number of phi0 per flux ramp, unitless, defaults 4
   gain: feedback gain parameter, unitless, defaults 1/32. I need to figure out how to map this onto lms_gain in pysmurf.
   NO BLANK OFF: this function is essentially the same as setting the blank-off to [0,0]  or something I guess

   fsamp: sample rate, units = Hz, defaults 2.4MHz
   nharm: number of harmonics to use in estimation, defaults 3. Currently SMuRF supports nharm <= 3; future iterations could use more.


   Returns:
   demod_sig: (nframes x 1), sampled at the reset rate. The demodulated phase of the first harmonic, units = radians
   phases: (length(measurement) x nharm), sampled at fsamp. The phases of each harmonic estimate in real time, units = radians
   errs: (same size as measurement), sampled at fsamp rate. The errors between the (constant) probe tone and resonance frequency, estimated via eta, units = Hz. 
   alphas: (length(measurement) x (2*nharm+1)), sampled at fsamp. The alpha coefficients at each timestep. 
   """
   framesize = int(fsamp / reset_rate) # number of time samples per flux ramp frame
   nframes = int(np.shape(respmag)[0] // framesize) # number of full flux ramp frames in the real timestream

   H = make_obs_mat(reset_rate,nphi0,fsamp,nharm) # construct observation coefficient matrix

   alpha = np.ones((2*nharm+1,)) # initialize coefficient vector alpha
   yhat = np.zeros((framesize*nframes,1)) # initialize frequency estimate vector yhat
   # estimate will start at center frequency of first timestep
   alphas = np.zeros((framesize*nframes,2*nharm+1))
   phases = np.zeros((framesize*nframes,nharm)) # initialize phase estimates of harmonics
   errs = np.zeros((framesize*nframes,1)) # initialize measurement vector, this is the frequency errors that we are trying to minimize

   for ii in range(framesize*nframes):
      rowno = np.mod(ii+framesize,framesize) # figure out where in the frame we are
      h = H[rowno,:] # get the harmonic components of this frame
      # this proceedings Eqs. 8 and 9.

      if ii==0:
      #if rowno==0:
         yhat[ii] = 0 # initialize at the center of subband; this could be changed later.
      else:
         yhat[ii] = np.matmul(h,alpha) # estimate is based on existing alpha

      centerfreq = estimate_res_freq(fvec,respmag[0,:],respphase[0,:])[0]
      #probetone = (centerfreq - yhat[ii])[0]
      probetone = centerfreq # keep this here at all times
      actualfreq = estimate_res_freq(fvec,respmag[ii,:],respphase[ii,:])[0]
      # estimate the frequency error
      errs[ii] = estimate_freq_error(fvec,respmag[ii,:],respphase[ii,:],eta,probetone)
        
      # maybe do something like comparing errs to expectation?
      expect = yhat[ii] - errs[ii]

      alpha = alpha + gain*expect*np.transpose(h) / (np.sum(np.power(h,2))) # h^2 is a normalization factor
    
      # which we just save for inspection
      alphas[ii,:] = alpha
    
      # save the phase estimate for each harmonic
      for jj in range(nharm):
         phases[ii,jj] = np.arctan2(alpha[2*jj+1],alpha[jj])

      # do not actually update alpha, though--it grows out of control
      alpha = np.ones((2*nharm+1,))
   demod = get_frame_averages(np.unwrap(phases[:,0]),reset_rate,fsamp)

   return demod, phases, errs, alphas 

def lms_fit_demod_meas(measurement,reset_rate=10.e3,nphi0=4,gain=1/32,blank=[0,1],fsamp=2.4e6,nharm=3):
   """
   The main SMuRF demodulation loop. This assumes "measurement" is perfect estimate of resonator frequency. 
   Args:
   measurement: the measured signal, in (nframes * fsamp / reset_rate) x 1, ie sampled at fsamp. Unit = Hz
   reset_rate: flux ramp sawtooth reset rate. Units = Hz, defaults 10kHz
   nphi0: number of phi0 per flux ramp, unitless, defaults 4
   gain: feedback gain parameter, unitless, defaults 1/32. I need to figure out the normalization on this thing. 
   blank: start and end fractions of the flux ramp frame to use for phase estimation. unitless, defaults [0,1]
   fsamp: sample rate, units = Hz, defaults 2.4MHz
   nharm: number of harmonics to use in estimation, defaults 3. Currently SMuRF supports nharm <= 3; future iterations could use more.


   Returns:
   demod_sig: (nframes x 1), sampled at the reset rate. The demodulated phase of the first harmonic, units = radians
   phases: (length(measurement) x nharm), sampled at fsamp. The phases of each harmonic estimate in real time, units = radians
   yhat: (same size as measurement), sampled at fsamp rate, the tracked frequency ie the probe tone applied based on the estimate of the resonance frequency
   errs: (same size as measurement), sampled at fsamp rate. The errors between the tracked frequency and estimate, units = Hz. 
   alphas: (length(measurement) x (2*nharm+1)), sampled at fsamp. The alpha coefficients at each timestep. 
   """
   framesize = int(np.round(fsamp / reset_rate)) # number of time samples per flux ramp frame
   nframes = int(np.round(np.shape(measurement)[0] / framesize)) # number of full flux ramp frames in the measurement

   H = make_obs_mat(reset_rate,nphi0,fsamp,nharm) # construct observation coefficient matrix

   alpha = np.ones((2*nharm+1,)) # initialize coefficient vector alpha
   alphas = np.zeros((framesize*nframes,2*nharm+1))
   yhat = np.zeros((framesize*nframes,1)) # initialize frequency estimate vector yhat

   phases = np.zeros((framesize*nframes,nharm)) # initialize phase estimates of harmonics
   errs = np.zeros((framesize*nframes,1)) # also initialize error vector

   for ii in range(framesize*nframes):
      blankmin = np.round(framesize*blank[0]) # get where in frame to start
      blankmax = np.round(framesize*blank[1]) # ...and where to stop

      rowno = np.mod(ii+framesize,framesize) # figure out where in the frame we are
      h = H[rowno,:] # get the harmonic components of this frame
      yhat[ii] = np.matmul(h,alpha) # estimate is based on existing alpha
      # this is proceedings Eqs. 8, 9

      # get the error
      errs[ii] = measurement[ii] - yhat[ii]
 
      if np.logical_or(rowno < blankmin, rowno > blankmax):
         alpha = alpha # in the blankoff region, do not update the coefficients
      else:
         # update the coefficients for the next point (proceedings Eq. 10)
         alpha = alpha + gain*errs[ii]*np.transpose(h) / (np.sum(np.power(h,2))) # h^2 is a normalization factor

      # save the phase estimate for each harmonic
      for jj in range(nharm):
         phases[ii,jj] = np.arctan2(alpha[2*jj+1],alpha[jj])
      # save the coefficients for inspection
      alphas[ii,:] = alpha

   demod = get_frame_averages(phases[:,0],reset_rate,fsamp)

   return demod, phases, yhat, errs, alphas 

def make_obs_mat(reset_rate=10.e3,nphi0=4,fsamp=2.4e6,nharm=3):
   """
   construct the observation matrix H for the SMuRF coefficient estimation
   this is currently SMuRF paper eq. 10

   Args:
   reset_rate: flux ramp sawtooth reset rate. Units = Hz, defaults 10kHz
   nphi0: number of phi0 per flux ramp, unitless, defaults 4
   fsamp: sample rate, units = Hz, defaults 2.4MHz
   nharm: number of harmonics to use in estimation, defaults 3. Currently SMuRF supports nharm <= 3; future iterations could use more.

   Returns:
   H: (M x 2N+1) matrix, where M is number of samples in a flux ramp frame (fsamp / reset_rate) and N is the number of harmonics used in the estimation. 
   """
   freqnorm = nphi0 * reset_rate / fsamp # normalized frequency, this is phi0 per sample
   framesize = int(np.round(fsamp / reset_rate)) # number of time samples per flux ramp frame

   H = np.zeros((int(framesize),2*nharm+1)) # construct observation matrix, per proceedings Eq. 6
   for ii in range(nharm):
     H[:,2*ii] = np.cos(2*np.pi*(ii+1)*freqnorm*np.arange(framesize))
     H[:,2*ii+1] = np.sin(2*np.pi*(ii+1)*freqnorm*np.arange(framesize))

   H[:,2*nharm] = np.ones((int(framesize),)) # add constant term

   return H

def get_frame_averages(phase,reset_rate=10.e3,fsamp=2.4e6):
   """
   Average over flux ramp frames to return one sample per frame

   Args:
   phase: the per-harmonic phase (probably of the principal harmonic) as returned by lms_demod, units=radians. Implicit assumption that this starts at the beginning of a flux ramp frame. 
   reset_rate: flux ramp reset rate, units = Hz, defaults 10kHz
   fsamp: per channel sample rate, units = Hz, defaults 2.4MHz

   Returns:
   avg: demodulated phase averaged over the flux ramp periods, so at the reset_rate. Still in units of radians. 
   """
   # do this by averaging method to avoid for loops
   framesize = int(np.round(fsamp / reset_rate)) # number of samples per flux ramp frame
   nframes = len(phase)//framesize
   phase_reshaped = np.reshape(phase,(-1,int(nframes),int(framesize)))
   avg = np.transpose(np.sum(phase_reshaped,-1) / framesize)

   return avg

def filter_and_downsample(sig,filt,fsamp_new=200.,fr_rate=10.e3):
   """
   Filter and downsample a signal. 
   The SMuRF data streamer does this before passing off to ocs, gcp, etc. 
   Other data products are available for debugging only, but usually we only 
   write this streamed data to disk. 

   Args:
   sig: signal to filter and downsample. Probably in units of radians. 
   filt: filter coefficients [b,a], to be passed to scipy.filtfilt. 
   fsamp_new: new sample frequency to downsample to, units=Hz, defaults 200Hz
   fr_rate: flux ramp frequency (that data came out of the baseband processor at), units=Hz, defaults 10kHz

   Returns:
   new_sig: filtered and downsampled signal, this is probably what is written to disk
   """
   b = filt[0]
   a = filt[1]
   filtered = scipy.signal.filtfilt(b,a,sig)
   downsample_factor = fr_rate / fsamp_new # factor by which to downsample
   new_sig = filtered[::int(downsample_factor)]
   return new_sig

def make_res_s21(f0,fvec,q,qc):
   """
   Make a resonator S21 (amplitude, phase) to interact with. 
   Note that this resonator is not guaranteed to be physical. You want Qc > Q, though. 

   Args:
   f0: center frequency, units = Hz
   fvec: frequency points at which to sample the response, units = Hz
   # width: +/- range of frequencies to produce response for, units = Hz. 
   q: quality factor, unitless
   qc: coupling quality factor, unitless, allowed to be complex to allow for asymmetry
   # asym: asymmetry, for probing effects of the asymmetry, unitless, defaults 0

   Returns (all len(fvec) x 1):
   amp: S21 amplitude
   ph: S21 phase
   """
   s21 = (1 - (q * qc**-1 / (1 + 2j * q * (fvec - f0) / f0)))
   amp = np.abs(s21)
   ph = np.angle(s21)

   return amp,ph

def estimate_eta(fvec,res_s21mag,res_s21phase,offset=30.e3,method='dphase'):
   """
   Estimate eta calibration angle per Eq. 3 of the SMuRF paper. 
   This estimation is critical to the SMuRF estimate of frequency!!!
   This step is done in the setup_notches step of pysmurf.

   Args:
   fvec: frequency points the S21 is sampled at, units=Hz
   res_s21*: complex transmission magnitude and phase, same size as fvec
   offset: offset from center at which to sample eta, units=Hz, defaults 30kHz
   method: passed to estimate_res_freq to get the resonance frequency

   Returns:
   eta: complex coefficients by which to rotate and scale resonator circle to perform frequency error estimates. eta_phase is in radians!
   """

   # get the bottom of the resonance, which is not guaranteed to be the same as the center frequency used to construct it if asymmetric
   f0,f0idx = estimate_res_freq(fvec,res_s21mag,res_s21phase,method)

   # then find the +/- offset points
   fpidx = np.where(np.abs(fvec-(f0+offset)) == np.min(np.abs(fvec-(f0+offset))))[0][0]
   fnidx = np.where(np.abs(fvec-(f0-offset)) == np.min(np.abs(fvec-(f0-offset))))[0][0]

   fpts = fvec[np.asarray([fnidx,f0idx,fpidx])]
   magpts = res_s21mag[np.asarray([fnidx,f0idx,fpidx])]
   phasepts = res_s21phase[np.asarray([fnidx,f0idx,fpidx])]

   # transform amp,phase into real,imaginary components
   Ivec = res_s21mag * np.cos(res_s21phase)
   Qvec = res_s21mag * np.sin(res_s21phase)

   eta = (2*offset) / np.complex(Ivec[fpidx]-Ivec[fnidx],Qvec[fpidx]-Qvec[fnidx])
   return eta

def estimate_res_freq(fvec,res_s21mag,res_s21phase,method='dphase'):
   """
   Estimate the resonance frequency given a frequency vector and S21 response. 
   More asymmetric resonances may be highly sensitive to the estimation method. 

   Args:
   fvec: frequency vector at which the S21 response is sampled, units = Hz
   res_21*: complex resonator transmission response, split into magnitude and phase, same size as fvec
   method: method of estimation. Currently supported:
     'ddmag': get the point of largest magnitude second derivative
     'dphase (Default)': get the point of maximum phase slip
     'minmag': get the point of minimum magnitude
     < will add more later >

   Returns:
   f0: resonance center frequency, units = Hz
   idx: index in fvec off the offset
   """

   mag = res_s21mag
   phase = res_s21phase

   if method=='ddmag':
      idx = np.where(np.diff(np.diff(mag)) == np.max(np.diff(np.diff(mag))))[0][0] + 1 # many derivatives net offsets by 1 relative to main vector

   elif method=='dphase':
      idx = np.where(np.diff(phase)==np.max(np.diff(phase)))[0][0]

   elif method=='minmag':
      idx = np.where(mag == np.min(mag))[0][0]

   else:
      raise ValueError("Estimation type not supported!")

   return fvec[idx],idx

def estimate_freq_error(fvec,res_s21mag,res_s21phase,eta,probe_f):
   """
   Estimate frequency error per Eq. 4 of the SMuRF paper. 
   Assume that the fvec,res_s21 are the TRUE state of the resonance

   Args:
   fvec: frequency points the S21 is sampled at, units=Hz
   res_s21*: complex transmission magnitude and phase, same size as fvec
   eta: calibration factor, as estimated from estimate_eta
   probe_f: frequency of the resonator probe tone, units=Hz. Currently assuming a perfectly spectrally clean probe tone. 

   Returns:
   fhat: estimate of the frequency shift, units = Hz
   """

   # get the complex transmission at that probe index
   s21 = res_s21mag * np.exp(1j*res_s21phase)
   trans = np.matlib.interp([probe_f],fvec,s21)[0]

   return np.real(trans * eta) # real vs imaginary component depends on digitizer I/Q relative to resonance I/Q; this works here

