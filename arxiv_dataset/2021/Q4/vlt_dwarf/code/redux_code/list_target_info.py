import pyfits
import sys

if len(sys.argv) > 1:
    inputname = str(sys.argv[1])
else:
    inputname = raw_input('Please type in the input file name--> ')
hdulist=pyfits.open(inputname)
idx=hdulist[1].data['index']
ra=hdulist[1].data['ra']
dec=hdulist[1].data['dec']
mag=hdulist[1].data['magnitude']
obj=hdulist[1].data['object']
type=hdulist[1].data['type']

print "#   fiber            ra             dec          object     type    mag"
for i in range(len(hdulist[1].data)):
	print('%8i %15.8f %15.8f %15s   %s %10.3f' %(idx[i], ra[i], dec[i], obj[i], type[i], mag[i]))
