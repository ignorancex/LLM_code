import numpy as np
from copy import deepcopy

def slopes_to_height(grad_x, grad_y, pad=0, nan_value=-32767):
    # # Integrate the x-gradient along the x direction
    # height_map_x = np.cumsum(grad_x, axis=1)

    # # Integrate the y-gradient along the y direction
    # height_map_y = np.cumsum(grad_y, axis=0)

    # fix, axes = plt.subplots(1, 2, figsize=(8,4))
    # # im1 = axes[0].imshow(grad_y_np)#, cmap='terrain')
    # # im2 = axes[1].imshow(grad_y)#, cmap='terrain')

    # im1 = sns.heatmap(height_map_x, ax=axes[0], annot=True)
    # im2 = sns.heatmap(height_map_y, ax=axes[1], annot=True)

    # height_map_x.shape, height_map_y.shape

    hh = np.full((
        grad_x.shape[0] + pad,
        grad_y.shape[1] + pad),
        fill_value=nan_value
    )
        
        
    failed = False
    for y in range(hh.shape[0]-pad):
        for x in range(hh.shape[1]-pad):
        
            if x == 0 and y == 0:
                hh[y, x] = 0
                continue
            
            if y == 0:
                hh[y, x] = hh[y, x-1] + grad_x[y, x-1]
                continue
                
            if x == 0:
                hh[y,x] = hh[y-1, x] + grad_y[y-1, x]
                continue
                
            if y > 0 and x > 0:
                    
                hy = hh[y-1, x] + grad_y[y-1, x]
                hx = hh[y, x-1] + grad_x[y, x-1]
                
                if hy == hx:
                    hh[y, x] = hx
                else:
                    failed = True
                    hh[y, x] = (hx + hy) / 2

            
            # if x == 0 and y < hh.shape[0]-1:
            #     hh[y+1, x] = hh[y,x] + grad_y[y,x]
            #     print(y+1,x)
            # if x < hh.shape[1] - 1:
            #     hh[y,x+1] = hh[y,x] + grad_x[y,x]
            #     print(y,x+1)
            # if y < hh.shape[1] - 1:
            #     hh[x,y+1] = hh[x,y] + grad_x[x,y]
            #     print(x,y+1)

    if pad:
        for y in range(hh.shape[0]-pad):
                x = hh.shape[0] - pad
                hh[y, x] = hh[y, x-1] + grad_x[y, x-1]
        
        for x in range(hh.shape[1]-pad):
                y = hh.shape[0] - pad
                hh[y, x] = hh[y-1, x] + grad_y[y-1, x]

        #hh[-1, :] = hh[-2,:] + grad_y[-1,:]
    if failed:
        print("Conflict in Reconstruction. Averaged the grad values")
    return hh

def height_to_slopes(heightmap):
    # # Calculate the gradient in both the x and y directions
    # grad_y_np, grad_x_np = np.gradient(heightmap)

    grad_x = heightmap[:, 1:] - heightmap[:, :-1]
    grad_y = heightmap[1:, :] - heightmap[:-1, :]
    
    return grad_x, grad_y

def augment_images(images_list: list, rot90=True, rot180=True, rot270=True, flip_v=True, flip_h=True, transpose=False):
    new_list = deepcopy(images_list)
    for image in images_list:
        if rot90:
            new_list.append(np.rot90(image))
        if rot180:
            new_list.append(np.rot90(np.rot90(image)))
        if rot270:
            new_list.append(np.rot90(np.rot90(np.rot90(image))))
        if flip_h:
            new_list.append(np.fliplr(image))
        if flip_v:
            new_list.append(np.flipud(image))
        if transpose:
            try:
                new_list.append(np.transpose(image, [1,0,2]))
            except:
                print("Can't add transpose!")

    return new_list
