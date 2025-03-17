import os

def save_img(trainset_copy, save_to, index_positions):
    try:
        os.makedirs(save_to)
    except FileExistsError:
        pass
    for i, index in enumerate(index_positions):
        image, label = trainset_copy[index]  
        image_path = os.path.join(save_to, f'image_{index}.png')
        image.save(image_path)      
    return 0