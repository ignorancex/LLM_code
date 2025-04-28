from copy import deepcopy
import numpy as np
import healpy as hp


def GetCount_ibox(ib, nside=4, boxsize=1200, thickness=100, pos_obs=[0,0,0], ppd=10):
    itt = (np.arange(0, ppd, 1) + 0.5)*boxsize/ppd #
    ttpos = np.meshgrid(itt, itt, itt, indexing='xy')
    ttpos[0] = ttpos[0].reshape(-1)
    ttpos[1] = ttpos[1].reshape(-1)
    ttpos[2] = ttpos[2].reshape(-1)
    ttpos = np.array(ttpos)
    # print(ttpos.shape)
    ttids = np.arange(ttpos.shape[-1])
    count_allbox = np.zeros((12*nside*nside, len(ttids)), dtype=np.int32) #nbox,
#     print("Box: ", ib)
    dismin = (ib+0.0)*thickness
    discen = (ib+0.5)*thickness
    dismax = (ib+1.0)*thickness
    xcopym = int(np.floor((pos_obs[0] - dismax) / boxsize))
    xcopyp = int(np.floor((pos_obs[0] + dismax) / boxsize))
    ycopym = int(np.floor((pos_obs[1] - dismax) / boxsize))
    ycopyp = int(np.floor((pos_obs[1] + dismax) / boxsize))
    zcopym = int(np.floor((pos_obs[2] - dismax) / boxsize))
    zcopyp = int(np.floor((pos_obs[2] + dismax) / boxsize))
    for ix in range(xcopym, xcopyp + 1):
        for iy in range(ycopym, ycopyp + 1):
            for iz in range(zcopym, zcopyp + 1):
#                 print(ix,iy,iz)
                ttpos2 = deepcopy(ttpos)
                ttpos2[0,:] = ttpos2[0,:] + ix*boxsize - pos_obs[0]
                ttpos2[1,:] = ttpos2[1,:] + iy*boxsize - pos_obs[1]
                ttpos2[2,:] = ttpos2[2,:] + iz*boxsize - pos_obs[2]
                squarerpos = (ttpos2[0,:])*(ttpos2[0,:]) + \
                             (ttpos2[1,:])*(ttpos2[1,:]) + \
                             (ttpos2[2,:])*(ttpos2[2,:])
                indr = (squarerpos >= dismin*dismin) * (squarerpos < dismax*dismax)
                indpix = hp.vec2pix(nside, ttpos2[0,indr], ttpos2[1,indr], ttpos2[2,indr], nest=False)
#                 uqpi÷x, uqcount = np.unique(indpix, return_counts=True
                for ipix in np.unique(indpix):
                    count_allbox[ipix, ttids[indr][indpix==ipix]] += 1 #ib,
    return count_allbox



