import os
import torch
import torch.neuron
import logging
import argparse
from cv_helper_class import ImgClassificationNet, Yolov5Net, VisionTransformerNet
from transformers import ViTFeatureExtractor, ViTForImageClassification
from torchvision import models
from common import preprocess_img

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, default="neuron_model/yolov5")
    parser.add_argument("--task", type=str, default="objdetect")
    parser_args, _ = parser.parse_known_args()
    return parser_args

def get_imgcls_model():
    IMGCLS_NETWORK = "RESNET"
    assert(IMGCLS_NETWORK in ["VGG", "RESNET", "RESNEXT", "EFFICIENTNET"])
    
    if IMGCLS_NETWORK == "VGG":
        model_type = "11" # can be 11,11_bn,13,13_bn,16,16_bn,19,19_bn
        assert(model_type in ['11','11_bn','13','13_bn','16','16_bn','19','19_bn'])
        model_name = f"models.vgg{model_type}"
    elif IMGCLS_NETWORK == "RESNET":
        model_type = 50 # can be 18,34,50,101,152   
        assert(model_type in [18,34,50,101,152])
        model_name = f"models.resnet{model_type}"
    elif IMGCLS_NETWORK == "RESNEXT":
        model_type = "50_32x4d" # can be 50_32x4d,101_32x8d,101_64x4d
        assert(model_type in ['50_32x4d','101_32x8d','101_64x4d'])
        model_name=f"models.resnext{model_type}"
    elif IMGCLS_NETWORK == "EFFICIENTNET":
        model_type = 0 # can be 0,1,2,3,4,5,6,7
        assert(model_type in range(8))
        model_name = f"models.efficientnet_b{model_type}"
        
    model_name_eval = eval(model_name)
    model = model_name_eval(pretrained=True) 
    net = ImgClassificationNet(model=model, model_name=model_name)
    return net, model, model_name

def get_objdetect_model():
    model_type = 'l'
    assert(model_type in ['n', 's', 'm', 'l', 'x'])
    model_name = f'yolov5{model_type}'  
    model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)
    net = Yolov5Net(model=model, model_name=model_name, img_size=640)
    return net, model, model_name

def get_vit_model():
    model_name = "vit-base-patch16-224"
    model_type = "vit"
    model = ViTForImageClassification.from_pretrained(f"google/{model_name}")
    net = VisionTransformerNet(model=model, model_name=model_name, model_type=model_type)
    return net, model, model_name

def main(args):
    print(args)
    os.makedirs(args.save_path, exist_ok=True)
    
    if (args.task == "cnn_imgcls") or (args.task == "vit_imgcls"):
        if args.task == "cnn_imgcls":
            net, model, model_name = get_imgcls_model()
        else:
            net, model, model_name = get_vit_model()    
    elif args.task == "objdetect":
        net, model, model_name = get_objdetect_model()

    neuron_model_path = os.path.join(args.save_path, "model_neuron.pt")
    if os.path.exists(neuron_model_path):
        print("Load pre-compiled model")        
        net.load(neuron_model_path)
    else:
        print("Compile model")
        net.analyze()
        net.compile(save_path=args.save_path)
            
if __name__ == "__main__":
    main(parse_args()) 

