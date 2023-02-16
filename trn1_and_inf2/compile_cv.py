import os
import sys
import torch
import torch_neuronx
import argparse
from cv_helper_class import ImgClassificationNet, VisionTransformerNet
from transformers import ViTFeatureExtractor, ViTForImageClassification
from torchvision import models
from common import preprocess_img

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="cnn_imgcls", 
                        choices=['cnn_imgcls', 'vit_imgcls'])
    parser.add_argument("--cnn_network", type=str, default="RESNET", 
                        choices=['VGG', 'RESNET', 'RESNEXT', 'EFFICIENTNET'])
    parser.add_argument("--save_path", type=str, default="neuron_model/resnet")

    parser_args, _ = parser.parse_known_args()
    return parser_args

def get_imgcls_model(args):
    assert(args.cnn_network in ["VGG", "RESNET", "RESNEXT", "EFFICIENTNET"])
    
    if args.cnn_network == "VGG":
        model_type = "11" # can be 11,11_bn,13,13_bn,16,16_bn,19,19_bn
        assert(model_type in ['11','11_bn','13','13_bn','16','16_bn','19','19_bn'])
        model_name = f"models.vgg{model_type}"
    elif args.cnn_network == "RESNET":
        model_type = 50 # can be 18,34,50,101,152   
        assert(model_type in [18,34,50,101,152])
        model_name = f"models.resnet{model_type}"
    elif args.cnn_network == "RESNEXT":
        model_type = "50_32x4d" # can be 50_32x4d,101_32x8d,101_64x4d
        assert(model_type in ['50_32x4d','101_32x8d','101_64x4d'])
        model_name=f"models.resnext{model_type}"
    elif args.cnn_network == "EFFICIENTNET":
        model_type = 0 # can be 0,1,2,3,4,5,6,7
        assert(model_type in range(8))
        model_name = f"models.efficientnet_b{model_type}"
        
    model_name_eval = eval(model_name)
    model = model_name_eval(pretrained=True) 
    net = ImgClassificationNet(model=model, model_name=model_name)
    return net, model, model_name

def get_vit_model():
    model_name = "vit-base-patch16-224"
    model_type = "vit"
    model = ViTForImageClassification.from_pretrained(f"google/{model_name}", torchscript=True)
    net = VisionTransformerNet(model=model, model_name=model_name, model_type=model_type)
    return net, model, model_name

def main(args):

    os.makedirs(args.save_path, exist_ok=True)
    
    if (args.task == "cnn_imgcls") or (args.task == "vit_imgcls"):
        if args.task == "cnn_imgcls":
            net, model, model_name = get_imgcls_model(args)
        else:
            net, model, model_name = get_vit_model()    

    neuron_model_path = os.path.join(args.save_path, f"neuron_{net.model_name}.pt")
    if os.path.exists(neuron_model_path):
        print("=== Load pre-compiled model")        
        net.load(neuron_model_path)
    else:
        print("=== Compile model")
        #net.analyze()
        net.compile(save_path=args.save_path)
            
if __name__ == "__main__":
    main(parse_args()) 
  