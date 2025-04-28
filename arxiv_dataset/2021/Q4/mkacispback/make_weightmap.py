from astropy.io import fits
import sys, os
import numpy as np

def main():
    hdu = fits.open(sys.argv[1])
    outfilename = sys.argv[2]
    
    data = hdu[0].data
    sumrow = np.sum(data, axis=1)
    total = np.sum(sumrow)
    if total > 0:
        normed = sumrow
    else: 
        normed = np.repeat(0,len(sumrow))

    with open(outfilename, 'w') as f:
        for i in normed:
            f.write(f'{i}\n')

    if total > 0:
        with open('total_'+os.path.splitext(outfilename)[0]+'.total' , 'w') as f:
            f.write(f'{total}\n')


if __name__ == "__main__":
    main()