import argparse
import json
import numpy as np
import torch
import torchvision.models as models
from PIL import Image
from math import ceil

def arg_parser():
    parser = argparse.ArgumentParser(description="predict.py")
    parser.add_argument('--image', type=str, help='Path to image file for prediction.', required=True)
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint file as str.', required=True)
    parser.add_argument('--top_k', type=int, help='Choose top K matches as int.')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='Path to JSON file containing category names.')
    parser.add_argument('--gpu', default=False, action="store_true", help='Use GPU for inference.')

    args = parser.parse_args()
    
    return args


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location='cpu')
    
    if checkpoint['arch'] == 'vgg16':
        model = models.vgg16(pretrained=True)
    else:
        model = models.densenet121(pretrained=True)
    
    for param in model.parameters(): 
        param.requires_grad = False
    
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model


def process_image(image):
    img = Image.open(image)
    
    size = 256, 256
    img.thumbnail(size)
    
    center = img.width/2, img.height/2
    left = center[0] - 224/2
    top = center[1] - 224/2
    right = center[0] + 224/2
    bottom = center[1] + 224/2
    
    img = img.crop((left, top, right, bottom))
    
    np_image = np.array(img)/255
    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    np_image = (np_image - mean)/std
    
    np_image = np.transpose(np_image, (2, 0, 1))
    
    return np_image


def predict(image_path, model, top_k=5, device='cpu'):
    model.to(device)
    
    img = process_image(image_path)
    img_tensor = torch.from_numpy(img).type(torch.FloatTensor).to(device)
    img_tensor.unsqueeze_(0)
    
    with torch.no_grad():
        output = model.forward(img_tensor)
        ps = torch.exp(output)
        top_probs, top_indices = ps.topk(top_k)
        top_probs = top_probs.cpu().numpy().squeeze()
        top_indices = top_indices.cpu().numpy().squeeze()
        idx_to_class = {val: key for key, val in model.class_to_idx.items()}
        top_classes = [idx_to_class[idx] for idx in top_indices]
        top_flowers = [cat_to_name[str(cls)] for cls in top_classes]
    
    return top_probs, top_classes, top_flowers


def print_results(probs, classes, flowers):
    for i in range(len(probs)):
        print("{}. {} - {:.2f}%".format(i+1, flowers[i], probs[i]*100))


def main():
    args = arg_parser()
    
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
    
    device = torch.device("cuda" if args.gpu else "cpu")
    
    model = load_checkpoint(args.checkpoint)
    
    top_probs, top_classes,
if __name__ == '__main__': main()