from functions import *

print len(sys.argv)
if len(sys.argv) == 2:
    inputname = str(sys.argv[1])
    data = pyfits.open(inputname)
    spec = data[0].data
    dspec = data[1].data #* 0.87
    tab = data[2].data
    hdr = data[0].header
    x0 = hdr['CRVAL2'] * 1
    delta_lam = hdr['CDELT2'] * 1
    [num,fibernum] = spec.shape
    lamb = np.arange(x0,x0+delta_lam*num-0.00001,delta_lam)
elif len(sys.argv) == 3:
    inputname = str(sys.argv[1])
    errorname = str(sys.argv[2])
    data = pyfits.open(inputname)
    spec = data[0].data
    dataerror = pyfits.open(errorname)
    spec = data[0].data
    dspec = dataerror[0].data #* 0.87
    tab = data[1].data
    hdr = data[0].header
    x0 = hdr['CRVAL2'] * 10
    delta_lam = hdr['CDELT2'] * 10
    [num,fibernum] = spec.shape
    lamb = np.arange(x0,x0+delta_lam*num-0.00001,delta_lam)
else:
    print "Error! No Input Files"
    exit()


wlmask = (lamb > 8300) & (lamb < 9000)
lamb = lamb[wlmask]
spec = spec[wlmask]
dspec = dspec[wlmask]

xmin = find_nearest(lamb, 8718)
xmax = find_nearest(lamb, 8723)
#xmin = find_nearest(lamb, 8936)
#xmax = find_nearest(lamb, 8940)
plt.figure()
plt.plot(spec[:,1])
plt.plot(dspec[:,1])
print len(np.arange(xmin,xmax))
print len(spec[xmin:xmax,1])
plt.plot(np.arange(xmin,xmax),spec[xmin:xmax,1],'r',lw=3)
plt.title('Red lines indicates the wavelength range used for SNR calculation')
plt.xlabel('pixel')
plt.ylabel('flux')
plt.show()

idx=[]
for i in np.arange(len(tab['object'])):
    if tab['magnitude'][i] > 0:
        mag = tab['magnitude'][i]
        snr1=np.median(spec[xmin:xmax,i])/np.std(spec[xmin:xmax,i])
        snr2=np.median(spec[xmin:xmax,i]/(dspec[xmin:xmax,i]))
        print snr1, snr2

        plt.plot(mag, snr1, 'ro')
        plt.plot(mag, snr2, 'g^')
        print mag, snr1, snr2

plt.title('SNR with 2 methods, red: median/std, green: median(spec/errorspec)')
plt.axhline(y=5,c='r')
plt.ylim(0,50)
plt.xlim(18.2,23)
plt.xlabel('g mag')
plt.ylabel('SNR')
plt.show()
