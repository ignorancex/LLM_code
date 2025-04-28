from astropy.io import fits
from astropy.table import Table
import os, shutil, argparse
import numpy as np

def rigid_transform_3D(A, B):
    # finding rotation/translation between 3 points
    # http://nghiaho.com/?page_id=671
    assert A.shape == B.shape

    num_rows, num_cols = A.shape
    if num_rows != 2:
        raise Exception(f"matrix A is not 2xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 2:
        raise Exception(f"matrix B is not 2xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)
    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    t = -R @ centroid_A + centroid_B

    return R, t

def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', default='../filter/evt2_clean.fits', help='input event file to create fake event files')
    parser.add_argument('--output_dir', default='.', help='output directory which will contain generated fake event files')
    args = parser.parse_args()
    return args

def main():
    args = parseArguments()

    empty_ccd,nonempty_ccd = [],[]
    for ccd_id in [0,1,2,3,5,6,7]:

        f=fits.open(os.path.join(args.input_file))
        t0=Table(f[1].data)
        t0.remove_columns(['status', 'phas'])
        t1=t0[t0['ccd_id']==ccd_id].copy()

        if len(t1) == 0:
            empty_ccd.append(ccd_id)
            continue
        else: 
            nonempty_ccd.append(ccd_id)

        #energy = 10keV (10,0000 eV)
        newrow0 = np.array([400000, 1., ccd_id, 1., 1., 1., 1., 1., 1., 1., 1., 1., 1000, 1000, 10000., 400, 1, 0])
        allnewrow = np.tile(newrow0,1024*1024)
        allnewrow = allnewrow.reshape(1024*1024,len(newrow0))

        #Create a transformation from chipx/y -> x/y
        A=np.array([t1['chipx'].value,t1['chipy'].value])
        B=np.array([t1['x'].value,t1['y'].value])
        ret_R, ret_t = rigid_transform_3D(A,B)

        #Create a new event file with every pixel has one count
#        print(len(newrow0), len(t0.columns), len(t0.dtype))
#        print(t0.columns)
        tnew = Table(rows=allnewrow, names=t0.columns, dtype=t0.dtype)
        tnew['chipx']=np.repeat(np.arange(1024),1024)+1
        tnew['chipy']=np.tile(np.arange(1024)+1,1024)
        boolarr=np.repeat(False,1024*1024*32)
        tnew['status'] = boolarr.reshape(1024*1024,32)

        C=np.array([tnew['chipx'].value,tnew['chipy'].value])
        C2 = ret_R @ C + ret_t
        tnew['x']=C2[0]; tnew['y']=C2[1]

        f[1].data=np.array(tnew)
        f.writeto(os.path.join(args.output_dir,f"temp_fake_events_ccd{ccd_id}.fits"), overwrite=True)
        f.close()

    if len(empty_ccd) > 0: 
        for ccd_id in empty_ccd:
            shutil.copyfile(os.path.join(args.output_dir,f"temp_fake_events_ccd{nonempty_ccd[-1]}.fits"),\
                os.path.join(args.output_dir,f"temp_fake_events_ccd{ccd_id}.fits"))

if __name__ == "__main__":
    main()
