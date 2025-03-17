
import os 
import numpy as np
import cv2
import numpy as np
from  utils import data_utils
from dataloaders import transforms_VINE as T
import torchvision.transforms as Tr
from PIL import Image
from torchvision.transforms import InterpolationMode
import torch
from torchvision.utils import make_grid
from tqdm import tqdm

SENSORS = ['images','masks']


LABEL_COLORS = [
        [0,0,0],
        [255, 255, 255]
]

ID_2_LABELS  = {0:0,
                1:1}

LABELS_2_ID  =  {0:0,
                1:1}

ID_2_COLORS =  {0:[0,   0,  0],
                1:[255, 255, 255]}

# ====== Original ==============
# CLASSES = [0,1]
# ==============================

MAX_CLASSES  = max(ID_2_LABELS.values())+1
LABELS       = ['Field','Corn']

MAX_ANGLE = 180

VIZ_TRANSFORM = Tr.ToTensor()
RESTORE       = Tr.ToPILImage()

AUGMENT = T.Compose([  T.ToTensor(),
            #T.RandomRotate(0,MAX_ANGLE),
            T.RandomHorizontalFlip(0.8),
            T.AdjustSaturation(),
            T.AdjustBrightness(),
            ])

PREPROCESSING =  T.Compose([T.ToTensor()])

def build_file_path(files,i):
    file_name = np.array(files['files'])[i]
    file_type = np.array(files['type'])[i]
    root_file = np.array(files['root'])[i]
    file = os.path.join(root_file,file_name + '.' + file_type)
    return(file,file_name)

def find_vector(tensor,vector):
    if isinstance(tensor,np.ndarray):
        # Tensor must be a numpy
        tensor = torch.tensor(tensor)
    
    if tensor.shape[-1]<tensor.shape[0]:
        tensor = torch.permute(tensor,(-1,0,1))
    tensor = tensor.clone()
    #print(f"vector shape: {tensor.shape}, vector shape: {vector.shape}")
    assert tensor.shape[0] == len(vector) # Tensor's channels must match the vector's dim
    # create a mask witht the same channel dim 
    mask = torch.ones(tensor.shape[1:3])
    for i,channel in enumerate(tensor):
        mask = mask*(channel == vector[i])
    
    counter = torch.sum(mask)
    return(mask,counter)

def save_mask_to_png(mask, name):
    mask_pil = Image.fromarray(np.uint8(mask))
    mask_pil.convert('RGB')
    mask_pil.save(name)

def img_to_tboard_batch(rgb,mask,pred):
    batch = []
    for rgb,mask,pred in zip(rgb,mask,pred):
        img = img_to_tensorboard(rgb,mask,pred)
        img = torch.unsqueeze(img,dim=0)
        batch.append(img)
    #tb_format = torch.stack(batch,dim=0)
    return(batch)

def img_to_tensorboard(rgb,mask,pred):
    
    assert len(rgb.shape) == 3  # (C,H,W)
    assert rgb.shape[0] == 3    # C = [R G B] 

    mask = conv_mask_to_img_torch(mask)
    pred = conv_mask_to_img_torch(pred)

    img = (rgb,mask,pred)

    img = [RESTORE(x)       for x in img] 
    img = [x.convert('RGB') for x in img]
    img = [VIZ_TRANSFORM(x) for x in img]

    img= torch.cat(img,dim=2)
    return(img)

#def conv_img_to_mask_np(mask):

    if not isinstance(mask,(np.ndarray, np.generic)):
        mask = np.array(mask)
    if mask.shape[-1]<mask.shape[0]:
        mask = np.transpose(mask,(-1,0,1))

    #shape = mask.shape[1:]
    #shape = mask.shape[:2]
    
    new_mask = np.zeros((mask.shape[0],mask.shape[1],MAX_CLASSES),dtype=np.uint8)
    for c,color_vector in ID_2_COLORS.items():
        if color_vector == None:
            continue
        label = ID_2_LABELS[c]
        mask_bin,n_pixels = find_vector(mask,color_vector)
        unq_labels = np.unique(mask_bin)
        new_mask[:,:,label] = mask_bin
    return(new_mask)

def conv_img_to_mask_np(img):
    """
    Convert the input image to a binary mask.

    Args:
        img (np.array): Input image as numpy array.

    Returns:
        np.array: Binary mask as numpy array.
    """
    if len(img.shape) == 3:  # if input image is RGB, convert to grayscale
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:  # if input image is already grayscale, no need to convert
        img_gray = img

    # threshold to create binary mask
    img_gray[img_gray < 128] = 0
    img_gray[img_gray >= 128] = 1

    return img_gray.astype(np.float32)




def conv_mask_to_img_torch(mask,color = LABEL_COLORS):
    mask  =conv_mask_to_img_np(mask,color = LABEL_COLORS)
    mask = torch.tensor(mask,dtype = torch.uint8)
    mask = torch.permute(mask,(-1,0,1))
    return(mask)

def conv_mask_to_img_np(mask,color = LABEL_COLORS):
    if not isinstance(mask,(np.ndarray, np.generic)):
        mask = np.array(mask)
    if mask.shape[0]<mask.shape[-1]:
        mask = np.transpose(mask,(1,2,0))
        
    #idx_mask   = np.argmax(mask,axis=-1)
    mask_png   = np.zeros((mask.shape[0],mask.shape[1],3),dtype=np.float32)
    #unq_labels = np.unique(idx_mask)
    unq_labels = np.unique(mask)

    for i,class_value in enumerate(unq_labels):
        id = LABELS_2_ID[class_value]
        color_vector = ID_2_COLORS[id]
        if color_vector == None:
            continue
        mask_png[mask == class_value,:]=color_vector
    
    return(mask_png)




class VARGEMDataset():
    def __init__(self,root,dataset,sensor,**kwargs):
        #self.data_dir = 'freibrg_forest'
        #self.data_dir = ''
           
        assert os.path.isdir(root)
        assert dataset in ['altum','x7']
        assert sensor in SENSORS

        self.root       = root
        self.set_dir    = dataset
        self.sensor_dir = sensor
        self.dir = os.path.join(self.root, self.set_dir,self.sensor_dir)
        
        if not os.path.isdir(self.dir):
            raise NameError("Path does not exist: " + self.dir)
 

    def _get_files(self):

        files = data_utils.get_files(self.dir)
        if len(files)==0:
            raise ValueError("No Files found in: " + self.dir)

        file_name_clean = []
        file_type_clean = []
        for file in files:
            f,ftype = file.split('.')
            file_name_clean.append(f)
            file_type_clean.append(ftype)

        indices   = np.argsort(file_name_clean)
        files     = np.array(file_name_clean)[indices]
        file_type = np.array(file_type_clean)[indices]
        root_list = [self.dir]*len(file_type)
        
        return({'root':root_list,
                'files': files,
                'type': file_type})



class MVARGEMDataset():
    def __init__(self,**kwargs):
        self.root = kwargs['root']
        self.rgb_dir  = kwargs['rgb_dir']
        self.mask_dir = kwargs['mask_dir']
        self.set      = kwargs['set']
        self.num_classes = kwargs['num_classes']
        self.labels = LABELS
        
        self.dataset_mode = 'DISK'
        
        if 'dataset_mode' in kwargs:
            self.dataset_mode = kwargs['dataset_mode']

        assert self.dataset_mode in ['RAM','DISK']
        
        self.target_size = [240,240]
        self.resize = Tr.Resize(self.target_size,interpolation=InterpolationMode.NEAREST)
        #self.resize = Tr.Resize(self.target_size)
        self.aug_bool = False
        if 'aug_bool' in kwargs:
            self.aug_bool = kwargs['aug_bool']
        
        self.aug = AUGMENT              
        self.preprocessing = PREPROCESSING

        self.rgb  = VARGEMDataset(self.root,self.set,self.rgb_dir)
        self.mask = VARGEMDataset(self.root,self.set,self.mask_dir)

        self.rgb_files  = self.rgb._get_files()
        self.mask_files = self.mask._get_files()
        
        self.dataset_len = len(self.rgb_files['files'])
        #print(f'\n[FOREST|{self.set}] Loaded {self.dataset_len} Files\n')

        if self.dataset_mode == 'RAM':
            self. load_dataset_to_RAM()

    def __str__(self):
        return ' '.join([type(self).__name__ + '\n',
                        f'Loaded {self.dataset_len} Files\n',
                        f'Target size: {self.target_size}\n',
                        f'Loading mode: {self.dataset_mode}'
                        ])

    def load_dataset_to_RAM(self):
        self.rgb_bag = []
        self.mask_bag =[]
        self.names_bag=[]

        for index in tqdm(range(self.dataset_len),"Loading to RAM: "):
            rgb, mask, id =  self.load_data(index)
            
            mask = conv_img_to_mask_np(mask)
            rgb,mask = self.preprocessing(rgb,mask)
            mask[mask>0]=1 # Needed because resize causes label distorchen

            self.rgb_bag.append(rgb)
            self.mask_bag.append(mask)
            self.names_bag.append(id)
        
        
    def get_data(self,indx):
        
        if self.dataset_mode == 'RAM':
            rgb = self.rgb_bag[indx]
            mask = self.mask_bag[indx]
            file_name= self.names_bag[indx]

        else:
            rgb, ndvi, mask, file_name = self.load_data(indx)
            mask = conv_img_to_mask_np(mask)
            rgb,mask = self.preprocessing(rgb,mask)
            mask[mask>0]=1 # Needed because resize causes label distorchen
            mask = mask.squeeze(1) #(!!!)

        return rgb, ndvi, mask, file_name
    
    def load_data(self, index):
        mask_file, file_name = build_file_path(self.mask_files, index)
        rgb_file, _ = build_file_path(self.rgb_files, index)
              
        #file_extension = os.path.splitext(mask_file)[1] 
        #if file_extension == '.tif':
            # Read image using OpenCV instead of skimage
            #img = cv2.imread(file_name, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #elif file_extension == '.npy':
            #img = np.load(file_name)
        #else:
            #raise ValueError(f"Unsupported image file format: {file_name}")
        
        mask = data_utils.load_file(mask_file)
        image = data_utils.load_file(rgb_file)

        nir = image[:, :, 4]
        r = image[:, :, 2]
        b = image[:, :, 0]
        g = image[:, :, 1]
        rgb = cv2.merge([r, g, b])
        
        ndvi = (nir.astype(float) - r.astype(float)) / (nir.astype(float) + r.astype(float) + 1e-7)

        return rgb, ndvi, mask, file_name

    


    def set_aug_flag(self,flag):
        self.aug_bool = flag


    def __getitem__(self,index):
        rgb, ndvi, mask, id =  self.get_data(index)
        
        
        if self.aug_bool:
            rgb,mask = self.aug(rgb,mask)

        return(rgb,ndvi,mask,id)
    
    def __len__(self):
        return(self.dataset_len)



