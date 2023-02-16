import os
import cv2
import torch
import torch_neuronx
from torchvision.io import read_image
from torchvision import models, transforms, datasets
from transformers import ViTFeatureExtractor, ViTForImageClassification
from transformers import BeitFeatureExtractor, BeitForImageClassification
import types
from common import preprocess_img

class ImgClassificationNet:
    def __init__(self, model, model_name, num_pipeline_cores=1, img_size=224):
        self.model = model
        self.model_name = model_name
        self.model_neuron  = None
        self.model.eval()
        self.num_pipeline_cores = num_pipeline_cores
        self.sample_data = torch.rand([1, 3, img_size, img_size], dtype=torch.float32)
        self.warmup() # There are some cases where analyze_model() and compile_model() do not work without warmup.     

    def _resize_img(self, img):
        """Resize the image"""
        scale_factor = 1
        if img.shape[0] in range(500, 1000):
            scale_factor = 0.8
        elif img.shape[0] in range(1000, 2000):
            scale_factor = 0.4
        elif img.shape[0] in range(2000, 4000):
            scale_factor = 0.2
        elif img.shape[0] in range(4000, 8000):
            scale_factor = 0.1 
            
        resize_img = cv2.resize(img, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
        return resize_img        

    def warmup(self):
        """Warmup the model"""
        try:
            y = self.model(self.sample_data)
        except Exception as e:
            print("[ERROR] Failed warmup!")            

    def analyze(self):
        """Analyze the model - this will show operator support and operator count"""
        #print(self.sample_data.shape)
        #torch.neuron.analyze_model(self.model, self.sample_data)         
        pass    

    def compile(self, save_path="."):
        """Compile the model using torch.neuron.trace to create a Neuron model"""
        try:
            # if self.num_pipeline_cores > 1:
            #     compiler_args = ['--neuroncore-pipeline-cores', str(self.num_pipeline_cores)]
            #     self.model_neuron = torch_neuronx.trace(self.model, self.sample_data, compiler_args=compiler_args)                
            # else:
            self.model_neuron = torch_neuronx.trace(self.model, self.sample_data)
            print("[SUCCEED] Model is jit traceable")
            torch.jit.save(self.model_neuron, os.path.join(save_path, f"neuron_{self.model_name}.pt"))
            #self.model_neuron.save(os.path.join(save_path, f"neuron_{self.model_name}_c{self.num_pipeline_cores}.pt"))
        except Exception as e:
            print("[ERROR] Model is not traceable!")
                
        #self.model_neuron_dp = torch.neuron.DataParallel(self.model_neuron)                
        return self.model_neuron
    
    def load(self, filepath):
        """Load neuron compiled model"""        
        if filepath is not None:
            print('load from filepath')
            self.model_neuron = torch.jit.load(filepath)
        else:
            self.model_neuron = torch.jit.load(self.model_name)
        #self.model_neuron_dp = torch.neuron.DataParallel(self.model_neuron)      
        return self.model_neuron

    def predict(self, x, data_parallel=False):
        """Run the model on inferentia and get the predictions."""
        if not torch.is_tensor(x):
            x = torch.from_numpy(x)        
        if data_parallel:
            outputs = self.model_neuron(x)
            #outputs = self.model_neuron_dp(x)
        else:
            outputs = self.model_neuron(x)
        return outputs
    
    def get_single_predict_result(self, img):
        """Get prediction results"""
        x = preprocess_img.preprocess_imagenet(img)
        outputs = self.predict(x,False).softmax(dim=1)
        top_5 = (-outputs[0]).argsort()[:5]    
        labels = preprocess_img.load_imagenet1k_labels()
        y_pred = top_5[0]; y_str = labels[y_pred]; y_prob = outputs[0][y_pred]
    
        resize_img = self._resize_img(img)
        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        cv2.putText(resize_img, f'{y_str} {y_prob:.4f}', (10,40), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
        return resize_img, y_pred, y_str, y_prob


class Yolov5Net(ImgClassificationNet):
    def __init__(self, model, model_name, num_pipeline_cores=1, img_size=640):
        super().__init__(model, model_name, num_pipeline_cores, img_size)

    def warmup(self):
        super().warmup() 
        
    def analyze(self):
        super().analyze()

    def compile(self, save_path="."):
        return super().compile(save_path)
        
    def load(self, filepath):
        return super().load(filepath)        
               
    def predict(self, x, data_parallel=True):
        return super().predict(x, data_parallel)

    def get_single_predict_result(self, img):
        x = preprocess_img.preprocess_coco(img)
        outputs = self.predict(x)
        out_img = preprocess_img.postprocess_yolov5(outputs, img)
        return out_img


class VisionTransformerNet(ImgClassificationNet):
    def __init__(self, model, model_name, model_type="vit", num_pipeline_cores=1, img_size=224):   
        
        self.model_type = model_type
        assert(self.model_type in ["vit", "beit"])
        print(model_type)
                             
        # change the forward to make it traceable (inf1 only)
        # if not hasattr(model, 'forward_'): model.forward_ = model.forward
        # model.forward = types.MethodType(lambda self,x: self.forward_(x).logits, model)
        super().__init__(model, model_name, num_pipeline_cores, img_size)

        if self.model_type == "vit": 
            self.feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
        else:
            self.feature_extractor = BeitFeatureExtractor.from_pretrained("microsoft/beit-base-patch16-224")
              
        inputs = self.feature_extractor(torch.randn(3, img_size, img_size), return_tensors="pt")
        self.sample_data = inputs['pixel_values']
                      
    def warmup(self):
        super().warmup() 
        
    def analyze(self):
        super().analyze()
        
    def compile(self, save_path="."):
        return super().compile(save_path)
        
    def load(self, filepath):
        return super().load(filepath)        

    def predict(self, x, data_parallel=True):
        return super().predict(x, data_parallel)

    def get_single_predict_result(self, img):
        x = self.feature_extractor(img, return_tensors="pt")['pixel_values']
        logits = self.predict(x, False)['logits'].softmax(dim=1)
        labels = preprocess_img.load_imagenet1k_labels()
        y_pred = logits.argmax(-1).item(); y_str = labels[y_pred]; y_prob = logits[0][y_pred]
        
        resize_img = self._resize_img(img)
        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        cv2.putText(resize_img, f'{y_str} {y_prob:.4f}', (10,40), font, 1, (255, 0, 0), 2, cv2.LINE_AA)
        return resize_img, y_pred, y_str, y_prob
