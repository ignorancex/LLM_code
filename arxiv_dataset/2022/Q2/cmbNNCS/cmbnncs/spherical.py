import numpy as np
import healpy as hp


class Cut(object):
    '''
    cut a Healpix map to 12 parts, or to 12*subblocks_nums parts
    '''
    def __init__(self, maps_in, subblocks_nums=1, nest=False):
        '''
        :param maps_in: the input healpix maps (one map with the shape of (12*nside**2,) or multiple maps with the shape of (N, 12*nside**2))
        :param nest: bool, if False map_in is assumed in RING scheme, otherwise map_in is NESTED
        :param subblocks_nums: int, the number after dividing the square nside*nside into small squares,
                               subblocks_nums=1^2, 2^2, 4^2, 8^2, 16^2, ..., default 1
        '''
        self.maps_in = maps_in
        self.nest = nest
        self.subblocks_nums = subblocks_nums
    
    @property
    def _multi_map(self):
        if len(self.maps_in.shape)==1:
            return False
        elif len(self.maps_in.shape)==2:
            self.multi_map_n = self.maps_in.shape[0]
            return True
    
    @property
    def nside(self):
        if self._multi_map:
            nsd = int(np.sqrt(self.maps_in[0].shape[0]/self.subblocks_nums/12))
        else:
            nsd = int(np.sqrt(self.maps_in.shape[0]/self.subblocks_nums/12))
        return nsd
    
    def _expand_array(self, original_array):
        '''
        to be used in nestedArray2nestedMap, expand the given small array into a large array
        
        :param original_array: with the shape of (2**n, 2**n), where n=1, 2, 4, 6, 8, 10, ...
        '''
        add_value = original_array.shape[0]**2
        array_0 = original_array
        array_1 = array_0 + add_value
        array_2 = array_0 + add_value*2
        array_3 = array_0 + add_value*3
        array_3_1 = np.c_[array_3, array_1]
        array_2_0 = np.c_[array_2, array_0]
        array = np.r_[array_3_1, array_2_0]
        return array

    def _ordinal_array(self):
        '''
        obtain an array containing the ordinal number, the shape is (nside, nside)
        '''
        circle_num = (int(np.log2(self.nside**2)) - 2)//2 #use //
        ordinal_array = np.array([[3.,1.],[2.,0.]])
        for i in range(circle_num):
            ordinal_array = self._expand_array(ordinal_array)
        return ordinal_array, circle_num

    def nestedArray2nestedMap(self, map_cut):
        '''
        reorder the cut map into NESTED ordering to show the same style using 
        plt.imshow() as that using Healpix
        
        :param map_cut: the cut map, the shape of map_cut is (nside**2,)
        
        return the reorded data, the shape is (nside, nside)
        '''
        array_fill, circle_num = self._ordinal_array()
        for i in range(2**(circle_num+1)):
            for j in range(2**(circle_num+1)):
                array_fill[i][j] = map_cut[int(array_fill[i][j])]
        #array_fill should be transposed to keep the figure looking like that in HEALPix
        array_fill = array_fill.T
        return array_fill

    def nestedMap2nestedArray(self, map_block):
        '''
        Restore the cut map(1/12 of full sky map) into an array which is in NESTED ordering
        
        need transpose if the map is transposed in nestedArray2nestedMap function
        '''
        map_block = map_block.T #!!!
        map_cut = np.ones(self.nside**2)
        array_fill, circle_num = self._ordinal_array()
        for i in range(2**(circle_num+1)):
            for j in range(2**(circle_num+1)):
                map_cut[int(array_fill[i][j])] = map_block[i][j]
        return map_cut
    
    def _block(self, Map, block_n):
        '''
        return one block of one of the original maps
        '''
        if self.nest:
            map_NEST = Map
        else:
            #reorder the map from RING ordering to NESTED
            map_NEST = hp.reorder(Map, r2n=True)
        
        map_part = map_NEST[block_n*self.nside**2 : (block_n+1)*self.nside**2]
        map_part = self.nestedArray2nestedMap(map_part)
        return map_part
    
    def block(self, block_n):
        if self._multi_map:
            map_part = []
            for i in range(self.multi_map_n):
                map_part.append(self._block(self.maps_in[i], block_n))
        else:
            map_part = self._block(self.maps_in, block_n)
        return map_part
    
    def _block_all(self, Map):
        '''
        return all blocks (12 blocks) of one of the original maps
        '''
        map_parts = []
        for blk in range(12*self.subblocks_nums):
            map_parts.append(self._block(Map, blk))
        return map_parts
    
    def block_all(self):
        if self._multi_map:
            map_parts = []
            for i in range(self.multi_map_n):
                map_parts.append(self._block_all(self.maps_in[i]))
        else:
            map_parts = self._block_all(self.maps_in)
        return map_parts


class Block2Full(Cut):
    '''
    stitch a cut map (1/12 of full sky map) to a full sky map with other parts is zeros
    '''
    def __init__(self, maps_block, block_n, subblocks_nums=1, base_map=None, nest=False):
        '''
        :param maps_block: the cut map in NESTED ording (one map in 2D array with the shape of (nside, nside) 
        or multiple maps in 3D array with the shape of (N, nside, nside) or multiple maps in a list with each element has the shape of (nside,nside))
        :param block_n: int, the number of cut map, 0, 1, 2, ..., 11
        :param nest: bool, if False base_map is assumed in RING scheme, otherwise base_map is NESTED
        :param subblocks_nums: int, the number after dividing the square nside*nside into small squares,
                               subblocks_nums=1^2, 2^2, 4^2, 8^2, 16^2, ..., default 1
        '''
        self.maps_block = maps_block
        self.block_n = block_n
        self.base_map = base_map
        self.nest = nest
        self.subblocks_nums = subblocks_nums
    
    @property
    def _multi_map(self):
        # list -> array
        if isinstance(self.maps_block, list):
            self.maps_block = np.array(self.maps_block)
        
        if len(self.maps_block.shape)==2:
            return False
        elif len(self.maps_block.shape)==3:
            self.multi_map_n = self.maps_block.shape[0]
            return True
    
    @property
    def nside(self):
        if self._multi_map:
            nsd = self.maps_block.shape[1]
        else:
            nsd = self.maps_block.shape[0]
        return nsd
    
    def _full(self, map_block):
        '''
        return a full sphere map
        '''
        map_block_array = self.nestedMap2nestedArray(map_block)
        if self.base_map is None:
            map_NEST = np.zeros(12*self.subblocks_nums*self.nside**2)
        else:
            if self.nest:
                map_NEST = self.base_map
            else:
                map_NEST = hp.reorder(self.base_map, r2n=True)
        map_NEST[self.block_n*self.nside**2 : (self.block_n+1)*self.nside**2] = map_block_array
        map_RING = hp.reorder(map_NEST, n2r=True)
        return map_RING
    
    def full(self):
        if self._multi_map:
            map_full = []
            for i in range(self.multi_map_n):
                map_full.append(self._full(self.maps_block[i]))
            map_full = np.array(map_full)
        else:
            map_full = self._full(self.maps_block)
        return map_full


#%% piece plane map
def sphere2piecePlane(sphere_map, nside=None):
    '''
    cut full map to 12 blocks, then piecing them together into a plane
    this is only for the case of subblocks_nums=1
    '''    
    ct = Cut(sphere_map, subblocks_nums=1)
    blocks = ct.block_all()
    if nside is None:
        nside = ct.nside
        
    piece_map = np.zeros((nside*4, nside*3))
    #part 1
    piece_map[nside*3:, :nside] = blocks[1] #block 1
    piece_map[nside*3:, nside:nside*2] = blocks[5]
    piece_map[nside*3:, nside*2:] = blocks[8]
    #part 2
    piece_map[nside*2:nside*3, :nside] = blocks[0]
    piece_map[nside*2:nside*3, nside:nside*2] = blocks[4]
    piece_map[nside*2:nside*3, nside*2:] = blocks[11]
    #part 3
    piece_map[nside:nside*2, :nside] = blocks[3]
    piece_map[nside:nside*2, nside:nside*2] = blocks[7]
    piece_map[nside:nside*2, nside*2:] = blocks[10]
    #part 4
    piece_map[:nside, :nside] = blocks[2]
    piece_map[:nside, nside:nside*2] = blocks[6]
    piece_map[:nside, nside*2:] = blocks[9]
    return piece_map

def _piecePlane2blocks(piece_map, nside=None):
    '''
    :param Map: plane map whose shape is (nside*4, nside*3)
    this is only for the case of subblocks_nums=1
    '''
    if nside is None:
        nside = piece_map.shape[-1]//3
    
    blocks = {}
    #part 1
    blocks['block_1'] = piece_map[nside*3:, :nside] #block 1
    blocks['block_5'] = piece_map[nside*3:, nside:nside*2]
    blocks['block_8'] = piece_map[nside*3:, nside*2:]
    #part 2
    blocks['block_0'] = piece_map[nside*2:nside*3, :nside]
    blocks['block_4'] = piece_map[nside*2:nside*3, nside:nside*2]
    blocks['block_11'] = piece_map[nside*2:nside*3, nside*2:]
    #part 3
    blocks['block_3'] = piece_map[nside:nside*2, :nside]
    blocks['block_7'] = piece_map[nside:nside*2, nside:nside*2]
    blocks['block_10'] = piece_map[nside:nside*2, nside*2:]
    #part 4
    blocks['block_2'] = piece_map[:nside, :nside]
    blocks['block_6'] = piece_map[:nside, nside:nside*2]
    blocks['block_9'] = piece_map[:nside, nside*2:]
    return blocks

def piecePlanes2blocks(piece_maps, nside=None):
    '''this is only for the case of subblocks_nums=1'''
    if nside is None:
        nside = piece_maps.shape[-1]//3
        
    if len(piece_maps.shape)==2:
        return _piecePlane2blocks(piece_maps, nside=nside)
    elif len(piece_maps.shape)==3:
        blocks = {}
        #part 1
        blocks['block_1'] = piece_maps[:, nside*3:, :nside] #block 1
        blocks['block_5'] = piece_maps[:, nside*3:, nside:nside*2]
        blocks['block_8'] = piece_maps[:, nside*3:, nside*2:]
        #part 2
        blocks['block_0'] = piece_maps[:, nside*2:nside*3, :nside]
        blocks['block_4'] = piece_maps[:, nside*2:nside*3, nside:nside*2]
        blocks['block_11'] = piece_maps[:, nside*2:nside*3, nside*2:]
        #part 3
        blocks['block_3'] = piece_maps[:, nside:nside*2, :nside]
        blocks['block_7'] = piece_maps[:, nside:nside*2, nside:nside*2]
        blocks['block_10'] = piece_maps[:, nside:nside*2, nside*2:]
        #part 4
        blocks['block_2'] = piece_maps[:, :nside, :nside]
        blocks['block_6'] = piece_maps[:, :nside, nside:nside*2]
        blocks['block_9'] = piece_maps[:, :nside, nside*2:]
        return blocks
    
def _piecePlane2sphere(piece_map, nside=None):
    '''
    :param Map: plane map whose shape is (nside*4, nside*3)
    this is only for the case of subblocks_nums=1
    '''
    if nside is None:
        nside = piece_map.shape[-1]//3
        
    blocks = _piecePlane2blocks(piece_map, nside=nside)
    base_map = np.zeros(12*nside**2)
    for i in range(12):
        full_map = Block2Full(blocks['block_%s'%i], i, base_map=base_map).full()
        base_map = full_map
    return full_map

def piecePlanes2spheres(piece_maps, nside=None):
    '''
    piece_maps: plane maps with shape of (N, nside*4, nside*3)
    '''
    if len(piece_maps.shape)==2:
        return _piecePlane2sphere(piece_maps, nside=nside)
    elif len(piece_maps.shape)==3:
        sph_maps = []
        for i in range(len(piece_maps)):
            sph_maps.append(_piecePlane2sphere(piece_maps[i], nside=nside))
        sph_maps = np.array(sph_maps)
        return sph_maps

def blocks2piecePlane(blocks, nside=None):
    if nside is None:
        nside = blocks['block_0'].shape[-1]
    
    piece_map = np.ones((4*nside, 3*nside))
    #part 1
    piece_map[nside*3:, :nside] = blocks['block_1'] #block 1
    piece_map[nside*3:, nside:nside*2] = blocks['block_5']
    piece_map[nside*3:, nside*2:] = blocks['block_8']
    #part 2
    piece_map[nside*2:nside*3, :nside] = blocks['block_0']
    piece_map[nside*2:nside*3, nside:nside*2] = blocks['block_4']
    piece_map[nside*2:nside*3, nside*2:] = blocks['block_11']
    #part 3
    piece_map[nside:nside*2, :nside] = blocks['block_3']
    piece_map[nside:nside*2, nside:nside*2] = blocks['block_7']
    piece_map[nside:nside*2, nside*2:] = blocks['block_10']
    #part 4
    piece_map[:nside, :nside] = blocks['block_2']
    piece_map[:nside, nside:nside*2] = blocks['block_6']
    piece_map[:nside, nside*2:] = blocks['block_9']
    return piece_map


#%% plane map (obtained by reshape the RING map)
def sphere2plane(map_RING):
    nside = int(np.sqrt(map_RING.shape[0]/12))
    map_plane = np.reshape(map_RING, (nside*3, nside*4))
    #reverse along the last dimension eg: a[...,::-1] or a[:,::-1]
    map_plane = map_plane[:, ::-1]
    return map_plane

def plane2sphere(map_plane):
    #reverse along the last dimension eg: a[...,::-1] or a[:,::-1]
    map_plane = map_plane[:, ::-1]
    map_RING = np.reshape(map_plane, (-1))
    return map_RING


#%%calculate correlation (similarity or distance) between pixels around one pixel
class SimilarityIdx(object):
    def __init__(self, nside=512):
        self.nside = nside
    
    @property
    def map_shape(self):
        return (self.nside*4, self.nside*3)
    
    def check_edge(self, idx_0, idx_1, idx_centerPixel=()):
        if idx_0<0 or idx_0>=self.map_shape[0] or idx_1<0 or idx_1>=self.map_shape[1]:
            return idx_centerPixel[0], idx_centerPixel[1]
        else:
            return idx_0, idx_1
    
    def get_pixIdx(self):
        """ calculate the pixel index surrounding a specific pixel """
        self.indexes = [[[], []] for i in range(8)]
        for i in range(self.map_shape[0]):
            for j in range(self.map_shape[1]):
                idx_01 = self.check_edge(i, j-1, idx_centerPixel=(i,j))
                idx_02 = self.check_edge(i, j+1, idx_centerPixel=(i,j))
                idx_03 = self.check_edge(i-1, j, idx_centerPixel=(i,j))
                idx_04 = self.check_edge(i+1, j, idx_centerPixel=(i,j))
                idx_05 = self.check_edge(i-1, j-1, idx_centerPixel=(i,j))
                idx_06 = self.check_edge(i-1, j+1, idx_centerPixel=(i,j))
                idx_07 = self.check_edge(i+1, j-1, idx_centerPixel=(i,j))
                idx_08 = self.check_edge(i+1, j+1, idx_centerPixel=(i,j))
                self.indexes[0][0].append(idx_01[0]); self.indexes[0][1].append(idx_01[1])
                self.indexes[1][0].append(idx_02[0]); self.indexes[1][1].append(idx_02[1])
                self.indexes[2][0].append(idx_03[0]); self.indexes[2][1].append(idx_03[1])
                self.indexes[3][0].append(idx_04[0]); self.indexes[3][1].append(idx_04[1])
                self.indexes[4][0].append(idx_05[0]); self.indexes[4][1].append(idx_05[1])
                self.indexes[5][0].append(idx_06[0]); self.indexes[5][1].append(idx_06[1])
                self.indexes[6][0].append(idx_07[0]); self.indexes[6][1].append(idx_07[1])
                self.indexes[7][0].append(idx_08[0]); self.indexes[7][1].append(idx_08[1])
        return self.indexes
    
    # def get_similarity(self):
    #     self.get_pixIdx()
    #     map_reshape = self.map.reshape(-1)
    #     similaritiy = []
    #     for i in range(8):
    #         similaritiy.append(np.abs(map_reshape - self.map[self.indexes[i][0], self.indexes[i][1]]))
    #     return similaritiy


#%%
def add_frame(map_cut, sides=(512,512), frame_width=10):
    ''' add a frame to one block 
    frame_width: int, the width of the frame. Default: 10 pixels
    '''
    #here use map_cut.copy() is right, if use map_frame=map_cut, 
    #otherwise, the input map_cut will change also!!!
    side_H, side_W = sides
    map_frame = map_cut.copy()
    for i in range(side_H):
        for j in range(side_W):
            for pix in range(frame_width):
                if i==0:
                    map_frame[i+pix,j]=1e6
                elif i==side_H-1:
                    map_frame[i-pix,j]=1e6
                elif j==0:
                    map_frame[i,j+pix]=1e6
                elif j==side_W-1:
                    map_frame[i,j-pix]=1e6
    return map_frame


#%% power spectrum
# def cl2dl(Cl, ell_start=0):
#     if ell_start==0:
#         lmax_cl = len(Cl) - 1
#     elif ell_start==2:
#         lmax_cl = len(Cl) + 1
    
#     ell = np.arange(lmax_cl + 1)
#     factor = ell * (ell + 1.) / 2. / np.pi
#     if ell_start==0:
#         Dl = Cl[2:] * factor[2:]
#     elif ell_start==2:
#         Dl = Cl * factor[2:]
#     return ell[2:], Dl

def cl2dl(Cl, ell_start, ell_in=None, get_ell=True):
    '''
    ell_start: 0 or 2, which should depend on Dl
    ell_in: the ell of Cl (as the input of this function)
    '''
    if ell_start==0:
        lmax_cl = len(Cl) - 1
    elif ell_start==2:
        lmax_cl = len(Cl) + 1
    
    ell = np.arange(lmax_cl + 1)
    if ell_in is not None:
        if ell_start==2:
            ell[2:] = ell_in
    
    factor = ell * (ell + 1.) / 2. / np.pi
    if ell_start==0:
        Dl = np.zeros_like(Cl)
        Dl[2:] = Cl[2:] * factor[2:]
        ell_2 = ell
    elif ell_start==2:
        Dl = Cl * factor[2:]
        ell_2 = ell[2:]
    if get_ell:
        return ell_2, Dl
    else:
        return Dl

def dl2cl(Dl, ell_start, ell_in=None, get_ell=True):
    '''
    ell_start: 0 or 2, which should depend on Dl
    ell_in: the ell of Cl (as the input of this function)
    '''
    if ell_start==0:
        lmax_cl = len(Dl) - 1
    elif ell_start==2:
        lmax_cl = len(Dl) + 1
    
    ell = np.arange(lmax_cl + 1)
    if ell_in is not None:
        if ell_start==2:
            ell[2:] = ell_in
    
    factor = ell * (ell + 1.) / 2. / np.pi
    if ell_start==0:
        Cl = np.zeros_like(Dl)
        Cl[2:] = Dl[2:] / factor[2:]
        ell_2 = ell
    elif ell_start==2:
        Cl = Dl / factor[2:]
        ell_2 = ell[2:]
    if get_ell:
        return ell_2, Cl
    else:
        return Cl

def anafast_spectra(Map, Map2=None, lmax=None, gal_cut=0, is_fullMap=True, block_n=4,
                    ell_start=2):
    """
    ell_start: 0 or 2. If 0, the output \ell start from 0, if 2, the output \ell start from 2. Default: 2
    """
    if not is_fullMap:
        Map = Block2Full(Map, block_n, base_map=None).full()
    Cl = hp.anafast(Map, map2=Map2, lmax=lmax, gal_cut=gal_cut)
    ell, Dl = cl2dl(Cl, ell_start=0)
    if ell_start==0:
        return ell, Dl
    elif ell_start==2:
        return ell[2:], Dl[2:]


#%% pixel size
class PixelSize:
    def __init__(self, nside, unit='deg'):
        """
        Calculate the pixel size in a Healpix map
        
        https://wikimili.com/en/Degree_(angle)
        https://wikimili.com/en/Radian
        
        Parameters
        ----------
        nside : TYPE
            DESCRIPTION.
        unit : str, the unit of angle, 'deg', 'arcmin', or 'arcsec', default: 'deg'

        Returns
        -------
        None.

        """
        self.nside = nside
        self.unit = unit
    
    @property
    def pixel_nums(self):
        return self.nside**2 * 12
    
    @property
    def sphere_area(self):
        """
        The area of a sphere in square degrees (deg^2), square arcminutes (arcmin^2), or square arcseconds (arcsec^2)
        
        1 deg^2 = (pi/180)^2 sr
        N deg^2 = 4pi sr --> N = 4pi/(pi/180)^2 = 360^2/pi
        
        https://wikimili.com/en/Square_degree
        https://wikimili.com/en/Steradian
        """
        if self.unit=='deg':
            return 360**2/np.pi
        elif self.unit=='arcmin':
            return 360**2/np.pi * 60**2
        elif self.unit=='arcsec':
            return 360**2/np.pi * 3600**2
    
    @property
    def pixel_size(self):
        """
        Pixel size in square degrees (deg^2), square arcminutes (arcmin^2), or square arcseconds (arcsec^2)
        
        https://infogalactic.com/info/Minute_and_second_of_arc
        """
        ps = self.sphere_area/self.pixel_nums
        if self.unit=='deg':
            print('The pixel size is %.5f deg^2'%ps)
        elif self.unit=='arcmin':
            print('The pixel size is %.5f arcmin^2'%ps)
        elif self.unit=='arcsec':
            print('The pixel size is %.5f arcsec^2'%ps)
        return ps
    
    @property
    def pixel_length(self):
        """
        Pixel length in degrees (deg), arcminutes (arcmin), or arcseconds (arcsec)
        
        https://baike.baidu.com/item/%E5%B9%B3%E6%96%B9%E5%BA%A6/9396914
        """        
        pl = np.sqrt(self.pixel_size)
        if self.unit=='deg':
            print('The pixel length is {:.5f} deg'.format(pl))
        elif self.unit=='arcmin':
            print('The pixel length is {:.5f} arcmin'.format(pl))
        elif self.unit=='arcsec':
            print('The pixel length is {:.5f} arcsec'.format(pl))
        return pl
    
