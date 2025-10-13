import torch
import numpy as np

def xy_location_to_num(input_size, crop_size, in_x, in_y,out_n_regions):
    out_n_regions = np.sqrt(out_n_regions)
    out_stride = int((input_size - crop_size) / (out_n_regions - 1))


    out_x, out_y = round(in_x / out_stride), round(in_y / out_stride)
    return int(out_x * out_n_regions + out_y)

def get_patch_predictions(image, patch_size, model, num_classes, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    
    # Make an attribution map, by sliding the predictor over the target image
    image = image.clone()
    image = image.to(device)
    patch_size=32
    # with stride one, no padding, go over every possible patch
    
    patches = []
    locations = []
    for i in range(0,image.shape[1]-patch_size + 2,2):
        for j in range(0,image.shape[2]-patch_size + 2,2):
            # extract patch and interpolate location
            patch = image[:,j:j+patch_size,i:i+patch_size].clone()
            location = xy_location_to_num(image.shape[1], patch_size, i,j, 2401)
    
            patches.append(patch)
            locations.append(location)

    # divide the image into patches
    patches = torch.stack(patches).reshape((-1,49,3,32,32))
    locations = torch.tensor(locations).reshape((-1,49))
    locations= locations.to(device)

    # loop over all mini batches and do predictions
    all_predictions = []
    for i in range(patches.shape[0]):
        with torch.no_grad():
            predictions = model(patches[i], locations[i])
            all_predictions.append(predictions)
    preds = torch.stack(all_predictions).reshape(-1,num_classes)

    return preds

def get_confidence_measure(preds, metric="top_softmax"):
    preds_softmax = torch.nn.functional.softmax(preds,dim=2)

    if metric == "top_softmax":
        max_prediction,_ = torch.max(preds_softmax, axis=2)
        return max_prediction
    elif metric == "negative_entropy":
        logp = torch.log(preds_softmax)
        entropy = - torch.sum(-preds_softmax*logp, axis=2)
        return entropy
    elif metric =="margin":
        values, idx = torch.topk(preds_softmax, axis=2, k=2)
        margins = values[:,:,0] - values[:,:,1]
        return margins
    
def map_attributions_to_pixels(confidences, locations, device):
    map = torch.zeros((128,128)).to(device)
    counts = torch.zeros((128,128)).to(device)
    for i, (x, y) in enumerate(locations):
        map[y:y+32,x:x+32] += confidences[i]
        counts[y:y+32,x:x+32] += 1

    return map / counts  

def convert_region_logits_to_pixel_map(atts, patch_size=32, img_shape=128, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), stride=2):
     
    confidences = get_confidence_measure(atts, "negative_entropy")  
    # convert from region to pixel map
    locations = []
    for i in range(0,img_shape-patch_size + stride,stride):
            for j in range(0,img_shape-patch_size + stride,stride):
                # extract patch and interpolate location
                locations.append([i,j])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    conf_pixels = map_attributions_to_pixels(torch.mean(confidences, axis=0).flatten(), locations, device)
    pixel_map = (conf_pixels - torch.min(conf_pixels)) / (torch.max(conf_pixels) - torch.min(conf_pixels))
    return pixel_map