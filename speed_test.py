import time
import numpy as np
import cv2
from torchvision import transforms
from PIL import Image
from ultralytics import YOLO

def custom_preprocess(image_path, width, height):
    # Open the image with PIL
    img = Image.open(image_path)
    target_size = (256, 256)

    # Convert PIL Image to NumPy array for OpenCV processing
    img_np = np.array(img)

    # Start the timer for resizing
    start_resize = time.time()
    resized_image = cv2.resize(img_np, target_size)
    end_resize = time.time()
    print(f"Resize time: {end_resize - start_resize} seconds")

    # Convert back to PIL Image for further processing
    img_pil = Image.fromarray(resized_image)

    # Start the timer for ToTensor conversion
    start_to_tensor = time.time()
    to_tensor = transforms.ToTensor()
    img_tensor = to_tensor(img_pil)
    end_to_tensor = time.time()
    print(f"ToTensor conversion time: {end_to_tensor - start_to_tensor} seconds")

    # Start the timer for normalization
    start_normalize = time.time()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    img_normalized = normalize(img_tensor)
    end_normalize = time.time()
    print(f"Normalization time: {end_normalize - start_normalize} seconds")

    # Start the timer for adding batch dimension
    start_batch = time.time()
    img_batched = img_normalized.unsqueeze(0)
    end_batch = time.time()
    print(f"Adding batch dimension time: {end_batch - start_batch} seconds")

    # Total time
    total_time = end_batch - start_resize
    print(f"Total preprocessing time: {total_time} seconds")

    return img_batched


"""mean = 0.4272211200448766 #0.2891893974394957 
std = 1.9514025672861495 #1.9843475932292247 """



"""mean =0.4384397828040158 #0.2891893974394957 
std = 1.9455521154556288 #1.9843475932292247 """


mean =0.4353692998265719
std = 1.9398368097425458



def reverse_standard_scaling(mean, std, scaled_data):
        original_data = [(val * std) + mean for val in scaled_data]
        return original_data

import numpy as np

def parse_ground_truth(file_path,object):
    ground_truth_data = {}
    with open(file_path, 'r') as file:
        for line in file:
            # Assuming the format is 'position X: data'
            parts = line.split(': ')
            position = parts[0].split(' ')[1]
            data = parts[1].strip()
            # Convert string data to a NumPy array
            ground_truth_array = np.array(data.split(','), dtype=float)
            ground_truth_data[f'/Users/matthiaspetry/Desktop/Masterarbeit/new_data_w_multiple_objects/testdata/{object}/image{position}.png'] = ground_truth_array
    return ground_truth_data

def create_image_data_list(ground_truth_data):
    image_data_list = []
    for image_name, ground_truth in ground_truth_data.items():
        image_data_list.append({'image': image_name, 'ground_truth': ground_truth})
    return image_data_list

import torch
import torch.nn as nn
import timm

import torch
import torch.nn as nn
import timm

#second 8
import torch
import torch.nn as nn
import timm

import torch
import torch.nn as nn
import timm
import torch
import torch.nn as nn
import timm

import torch
import torch.nn as nn
import timm

class LayerNormFastViT3DPosition(nn.Module):
    def __init__(self, dropout_rate=0.1, vector_input_size=4, intermediate_size=128, hidden_layer_size=64):
        super(LayerNormFastViT3DPosition, self).__init__()

        # Load FastViT model pre-trained on ImageNet
        self.fastvit = timm.create_model('fastvit_t8.apple_dist_in1k', pretrained=False)
        #self.fastvit = timm.create_model('resnet18.a1_in1k', pretrained=True)
        #self.fastvit = timm.create_model('efficientvit_m0.r224_in1k', pretrained=False)

        in_features = self.fastvit.get_classifier().in_features
        self.fastvit.reset_classifier(num_classes=0)  # Remove the classifier

        # Model for processing 4D vector input with LayerNorm
        self.vector_model = nn.Sequential(
            nn.Linear(vector_input_size, intermediate_size),
            nn.ReLU(),
            nn.LayerNorm(intermediate_size),
            nn.Linear(intermediate_size, in_features),
            nn.ReLU(),
            nn.LayerNorm(in_features),
            nn.Dropout(dropout_rate)
        )

        # Enhanced combined output layer with LayerNorm
        self.combined_output_layer = nn.Sequential(
            nn.Linear(in_features * 2, hidden_layer_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_layer_size),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_layer_size),
            nn.Linear(hidden_layer_size, 6)
        )

    def forward(self, x, vector):
        # Extract features using FastViT
        fastvit_features = self.fastvit(x)

        # Process the 4D vector through the vector model
        vector_features = self.vector_model(vector)

        # Concatenate FastViT and vector features
        concatenated_features = torch.cat((fastvit_features, vector_features), dim=1)

        # Final output layer for regression
        final_output = self.combined_output_layer(concatenated_features)

        return final_output




class FastViT3DPosition(nn.Module):
    def __init__(self, dropout_rate=0.2):  # Reduced dropout rate
        super(FastViT3DPosition, self).__init__()
        # Load FastViT model pre-trained on ImageNet
        self.fastvit = timm.create_model('fastvit_ma36.apple_dist_in1k', pretrained=False)

        # Model for 4D vector input
        self.vector_model = nn.Sequential(
            nn.Linear(4, 32),
            nn.LeakyReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 64),
            nn.LeakyReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout_rate),  # Applying dropout less frequently
            nn.Linear(64, 128),
            nn.LeakyReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 6)
        )

        # Replace the original classifier in FastViT for regression
        in_features = self.fastvit.get_classifier().in_features
        self.fastvit.reset_classifier(num_classes=0)  # Remove the classifier
        self.fastvit_output_layer = nn.Linear(in_features, 6)  # New output layer for regression
        
        # Additional layer to combine outputs and produce a 6D vector
        self.combined_fc = nn.Linear(12, 6)

    def forward(self, x, vector):
        # Pass input through FastViT model
        features = self.fastvit(x)
        fastvit_output = self.fastvit_output_layer(features)

        # Normalize the 4D vector input
        vector_norm = torch.norm(vector, p=2, dim=1, keepdim=True)
        vector_normalized = vector / vector_norm

        # Pass 4D vector through second model
        vector_output = self.vector_model(vector_normalized)

        # Concatenate FastViT and vector outputs
        concatenated_output = torch.cat((fastvit_output, vector_output), dim=1)

        # Final layer to get a 6D vector for regression
        final_output = self.combined_fc(concatenated_output)

        return final_output

import torch
import torch.nn as nn
import timm

class RefinedFastViT3DPosition(nn.Module):
    def __init__(self, dropout_rate=0.1, vector_input_size=4, intermediate_size=128):
        super(RefinedFastViT3DPosition, self).__init__()

        # Load FastViT model pre-trained on ImageNet
        self.fastvit = timm.create_model('resnet18.a1_in1k', pretrained=True)
        in_features = self.fastvit.get_classifier().in_features
        self.fastvit.reset_classifier(num_classes=0)  # Remove the classifier

        # Refined model for processing 4D vector input
        self.vector_model = nn.Sequential(
            nn.Linear(vector_input_size, intermediate_size),
            nn.ReLU(),
            nn.BatchNorm1d(intermediate_size),
            nn.Linear(intermediate_size, in_features),  # Matching the FastViT feature dimension
            nn.ReLU(),
            nn.BatchNorm1d(in_features),
            nn.Dropout(dropout_rate)
        )

        # Combined output layer for regression
        self.combined_output_layer = nn.Linear(in_features * 2, 6)

    def forward(self, x, vector):
        # Extract features using FastViT
        fastvit_features = self.fastvit(x)

        # Process the 4D vector through the vector model
        vector_features = self.vector_model(vector)

        # Concatenate FastViT and vector features
        concatenated_features = torch.cat((fastvit_features, vector_features), dim=1)

        # Final output layer for regression
        final_output = self.combined_output_layer(concatenated_features)

        return final_output

import torch
import torch.nn as nn
import timm

import torch
import torch.nn as nn
import timm

class ModifiedFastViT3DPosition(nn.Module):
    def __init__(self, dropout_rate=0.1, vector_input_size=4, intermediate_size=128, hidden_layer_size=64):
        super(ModifiedFastViT3DPosition, self).__init__()

        # Load FastViT model pre-trained on ImageNet
        self.fastvit = timm.create_model('fastvit_t8.apple_dist_in1k', pretrained=True)
        in_features = self.fastvit.get_classifier().in_features
        self.fastvit.reset_classifier(num_classes=0)  # Remove the classifier

        # Refined model for processing 4D vector input
        self.vector_model = nn.Sequential(
            nn.Linear(vector_input_size, intermediate_size),
            nn.ReLU(),
            nn.BatchNorm1d(intermediate_size),
            nn.Linear(intermediate_size, hidden_layer_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_layer_size),
            nn.Dropout(dropout_rate)
        )

        # Attention Mechanism for image features
        self.image_attention = nn.Sequential(
            nn.Linear(in_features, hidden_layer_size),
            nn.Tanh(),
            nn.Linear(hidden_layer_size, in_features)  # Adjust to output the same size as in_features
        )

        # Attention Mechanism for vector features
        self.vector_attention = nn.Sequential(
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.Tanh(),
            nn.Linear(hidden_layer_size, hidden_layer_size)  # Adjust to output the same size as hidden_layer_size
        )

        # Final combined output layer
        self.combined_output_layer = nn.Sequential(
            nn.Linear(in_features + hidden_layer_size, hidden_layer_size),  # Adjust input size
            nn.ReLU(),
            nn.BatchNorm1d(hidden_layer_size),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_layer_size, 6)
        )

    def forward(self, x, vector):
        # Extract features using FastViT
        fastvit_features = self.fastvit(x)

        # Process the 4D vector through the vector model
        vector_features = self.vector_model(vector)

        # Apply attention to image and vector features
        image_attention_weights = self.image_attention(fastvit_features)
        vector_attention_weights = self.vector_attention(vector_features)

        # Weighted sum of features based on attention
        weighted_image_features = image_attention_weights * fastvit_features
        weighted_vector_features = vector_attention_weights * vector_features

        # Concatenate weighted features
        concatenated_features = torch.cat((weighted_image_features, weighted_vector_features), dim=1)

        # Final output layer for regression
        final_output = self.combined_output_layer(concatenated_features)

        return final_output



class AdvancedFastViT3DPosition(nn.Module):
    def __init__(self, dropout_rate=0.1, vector_input_size=4, intermediate_size=128, hidden_layer_size=64):
        super(AdvancedFastViT3DPosition, self).__init__()

        # Load FastViT model pre-trained on ImageNet
        #self.fastvit = timm.create_model('fastvit_t8.apple_dist_in1k', pretrained=False)resnet18.a1_in1k
        #self.fastvit = timm.create_model('resnet18.a1_in1k', pretrained=False)
        self.fastvit = timm.create_model('efficientvit_m0.r224_in1k', pretrained=True)
        in_features = self.fastvit.get_classifier().in_features
        self.fastvit.reset_classifier(num_classes=0)  # Remove the classifier

        # Refined model for processing 4D vector input
        self.vector_model = nn.Sequential(
            nn.Linear(vector_input_size, intermediate_size),
            nn.ReLU(),
            nn.BatchNorm1d(intermediate_size),
            nn.Linear(intermediate_size, in_features),  # Matching the FastViT feature dimension
            nn.ReLU(),
            nn.BatchNorm1d(in_features),
            nn.Dropout(dropout_rate)
        )

        # Enhanced combined output layer with additional layers
        self.combined_output_layer = nn.Sequential(
            nn.Linear(in_features * 2, hidden_layer_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_layer_size),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_layer_size),
            nn.Linear(hidden_layer_size, 6)
        )

    def forward(self, x, vector):
        # Extract features using FastViT
        fastvit_features = self.fastvit(x)

        # Process the 4D vector through the vector model
        vector_features = self.vector_model(vector)

        # Concatenate FastViT and vector features
        concatenated_features = torch.cat((fastvit_features, vector_features), dim=1)

        # Final output layer for regression
        final_output = self.combined_output_layer(concatenated_features)

        return final_output





import os
import cv2
import torch
from pathlib import Path
from sklearn.metrics import *
from tqdm import tqdm
import torchvision.transforms as transforms
import onnxruntime as onnxrt
import time  # Import time module

y_pred = []
y_true = []
import time
import numpy as np
import cv2
from torchvision import transforms
from PIL import Image



def annotate_and_save_images(class_to_annotate, positions_file, object, model, model_Efficient_Net,model2):
    # Load YOLOv8 model

    # Define target size
    target_size = (512, 288)

    data_start = time.time()  # Start timing for YOLO
    ground_truth_data = parse_ground_truth(positions_file, object)
    image_data_list = create_image_data_list(ground_truth_data)
    data_time = time.time() - data_start  # Time taken by YOLO model

    total_time_yolo = 0
    total_time_efficient_net = 0
    total_time_resizing = 0
    total_combined_time = 0 
    loop_time = 0
    net = 0

    for image_data in tqdm(image_data_list, desc="Processing Images"):
        image_name = image_data['image']
        ground_truth = image_data['ground_truth']
        if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Read image
            image = cv2.imread(image_name)

            # Start timing for resizing
            resized_image = cv2.resize(image, target_size)

            start_time = time.time()  # Start timing for YOLO
            results = model.predict(resized_image, verbose=False, imgsz=512)
            yolo_time = time.time() - start_time  # Time taken by YOLO model
            total_time_yolo += yolo_time  # Accumulate time for YOLO

            object_detected = False
            start_time_loop = time.time()
            for result in results:
                boxes = result.boxes.xywhn.tolist()
                classes = result.boxes.cls.tolist()

                for i, cls in enumerate(classes):
                    if cls == class_to_annotate:
                        object_detected = True

                        xn, yn, wn, hn = boxes[i]
                        bbox_tensor = torch.tensor([xn, yn, wn, hn]).unsqueeze(0)   

                        # Open the image with PIL
                        start_process = time.time()

                        target_size2 = (256, 256)

                        # Convert PIL Image to NumPy array for OpenCV processing
                        
                        img_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                        # Start the timer for resizing
     
                        resized_image = cv2.resize(img_np, target_size2)
                       

                        # Convert back to PIL Image for further processing
                        img_pil = Image.fromarray(resized_image)

                        # Start the timer for ToTensor conversion
                        to_tensor = transforms.ToTensor()
                        img_tensor = to_tensor(img_pil)

                    

                        # Start the timer for normalization
                        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        img_normalized = normalize(img_tensor)
                        

                        img_batched = img_normalized.unsqueeze(0)

                        resized_image2 = cv2.resize(img_np, (224,224))
                       

                        # Convert back to PIL Image for further processing
                        img_pil2 = Image.fromarray(resized_image2)

                        # Start the timer for ToTensor conversion
                        to_tensor2 = transforms.ToTensor()
                        img_tensor2 = to_tensor(img_pil2)

                    

                        # Start the timer for normalization
                       
                        img_normalized2 = normalize(img_tensor2)
                        

                        img_batched2 = img_normalized2.unsqueeze(0)
                        

                        
                        with torch.no_grad():
                            start_time = time.time()  # Start timing for EfficientNet
                            #img_batched = torch.randn(1,3,256,256)
                            #bbox_tensor = torch.randn(1,4)
                            outputs = model_Efficient_Net(img_batched, bbox_tensor)
                            efficient_net_time = time.time() - start_time  # Time taken by EfficientNet
                            total_time_efficient_net += efficient_net_time  # Accumulate time for EfficientNet
                            combined = yolo_time + efficient_net_time
                            total_combined_time += combined
                            jp = reverse_standard_scaling(mean,std,outputs.numpy())[0]
                            start_time = time.time()
                            
                            pred = model2(img_batched2)
                            net_time = time.time() - start_time
                            net += net_time

                            #print(torch.argmax(pred, dim=1).numpy())
                            y_pred.append(jp)
                            y_true.append(ground_truth)
            end_time_loop = time.time() - start_time_loop
            loop_time += end_time_loop

            if not object_detected:
                print(f"No specified objects detected in {image_name}")

    # Calculate and print total and mean time for each operation
    num_images = len(image_data_list)
    mean_time_yolo = total_time_yolo / num_images
    mean_time_efficient_net = total_time_efficient_net / num_images
    mean_time_resizing = total_time_resizing / num_images
    mean_net = net_time/num_images
    print(f"Total time taken by YOLO model: {total_time_yolo} seconds")
    print(f"Mean time per image for YOLO model: {mean_time_yolo} seconds")
    print(f"Total time taken by FasterVit model: {total_time_efficient_net} seconds")
    print(f"Mean time per image for FasterVit model: {mean_time_efficient_net} seconds")
    print(f"Mean time taken by Net model: {mean_net} seconds")

    


# Rest of your code...



if __name__ == "__main__":
    model = YOLO("yolov8s.yaml")  # build a new model from scratch
    model = YOLO("/Users/matthiaspetry/Desktop/Masterarbeit/models/Yolov8_best.pt") 


    #state_dict = torch.load("/Users/matthiaspetry/Desktop/efficientvit_m0_r224_in1k26_1.pth",map_location=torch.device('cpu'))
    state_dict = torch.load("/Users/matthiaspetry/Desktop/fastvit_t8_129_1.pth",map_location=torch.device('cpu'))
    #state_dict = torch.load("/Users/matthiaspetry/Desktop/model_131.pth",map_location=torch.device('cpu'))
    #onnx_session = onnxrt.InferenceSession("/Users/matthiaspetry/Downloads/fastvit12_t8-sim.onnx")
    #model_name = "efficientnet_b0.ra_in1k"  # Example model name, change as per need
    #efficientvit_m0.r224_in1k tinynet_e.in1k
    model_name = "efficientvit_m0.r224_in1k"  # Example model name, change as per need
    model2 = timm.create_model(model_name, pretrained=False, num_classes=2)
    state_dict2 = torch.load("models/binary_classification_eff_vit.pth",map_location=torch.device('cpu')) #efficientvit_m0.r224_in1k
    #state_dict2 = torch.load("/Users/matthiaspetry/Desktop/binary_classification_eff_vit.pth",map_location=torch.device('cpu')) #efficientvit_m0.r224_in1k
    # Assuming EfficientNet3DPosition is the class of your model
    model2.load_state_dict(state_dict2)
    model_Efficient_Net = LayerNormFastViT3DPosition()
    #model_Efficient_Net = ModifiedFastViT3DPosition()
    #ModifiedFastViT3DPosition
    # Load the state_dict into the model
    model_Efficient_Net.load_state_dict(state_dict)
    model2.to("cpu")
    model2.eval()
    model_Efficient_Net.to("cpu")
    model_Efficient_Net.eval()
    def mean_time_execution():
        total_time = 0
        number_of_runs = 1

        for _ in range(number_of_runs):
            start_time = time.time()
            
            # Your function calls
            annotate_and_save_images(4, "/Users/matthiaspetry/Desktop/Masterarbeit/more_objects6/Cali_Pyramid_640x360/Pyramid_positions.txt", "Pyramid", model, model_Efficient_Net, model2)
            annotate_and_save_images(2, "/Users/matthiaspetry/Desktop/Masterarbeit/more_objects6/Cali_Cylinder_640x360/Cylinder_positions.txt", "Cylinder", model, model_Efficient_Net, model2)
            annotate_and_save_images(0, "/Users/matthiaspetry/Desktop/Masterarbeit/more_objects6/Cali_Cross_640x360/Cross_positions.txt", "Cross", model, model_Efficient_Net, model2)
            annotate_and_save_images(1, "/Users/matthiaspetry/Desktop/Masterarbeit/more_objects6/Cali_Cube_640x360/Cube_positions.txt", "Cube", model, model_Efficient_Net,model2)
            annotate_and_save_images(3, "/Users/matthiaspetry/Desktop/Masterarbeit/more_objects6/Cali_Hexagon_640x360/Hexagon_positions.txt", "Hexagon", model, model_Efficient_Net,model2)
            annotate_and_save_images(7, "/Users/matthiaspetry/Desktop/Masterarbeit/more_objects6/Cali_Y_Cube_640x360/Y_Cube_positions.txt", "Y_Cube", model, model_Efficient_Net,model2)

            total_time += (time.time() - start_time)

        mean_time = total_time / number_of_runs
        return mean_time

    # Call the function and print the mean time
    mean_time = mean_time_execution()
    print("Mean time for 10 runs: %s seconds" % mean_time)
    print(mean_absolute_percentage_error(y_true, y_pred))
    print(mean_absolute_error(y_true, y_pred))
    print(r2_score(y_true, y_pred))
