import time
import numpy as np
import cv2
from torchvision import transforms
from PIL import Image
import torch
import torch.nn as nn
import timm
import csv

import os
import torch
from pathlib import Path
from sklearn.metrics import *
from tqdm import tqdm
import torchvision.transforms as transforms

mean = 0.43540330533546434
std = 1.9398379424259808



class LayerNormFastViT6DPosition(nn.Module):
    def __init__(self, dropout_rate=0.1, vector_input_size=8, intermediate_size=128, hidden_layer_size=64):
        super(LayerNormFastViT6DPosition, self).__init__()

        # Load FastViT model pre-trained on ImageNet
        self.fastvit = timm.create_model('fastvit_t8.apple_dist_in1k', pretrained=False) 
        #self.fastvit = timm.create_model('efficientvit_m0.r224_in1k', pretrained=True)
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
            ground_truth_data[f'/Users/matthiaspetry/Desktop/afd.v16i.yolov8/train/images/image{position}.jpg'] = ground_truth_array
    return ground_truth_data

def create_image_data_list(ground_truth_data):
    image_data_list = []
    for image_name, ground_truth in ground_truth_data.items():
        image_data_list.append({'image': image_name, 'ground_truth': ground_truth})
    return image_data_list




y_pred = []
y_true = []

def annotate_and_save_images( positions_file, model):
    # Load YOLOv8 model

    # Define target size
    target_size = (512, 288)

    data_start = time.time()  # Start timing for YOLO
    ground_truth_data = parse_ground_truth(positions_file, object)
    image_data_list = create_image_data_list(ground_truth_data)

    with open('image_resultstest.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(['Image Name','CLASS', 'MAPE', 'MAE'])

        for image_data in tqdm(image_data_list, desc="Processing Images"):
            image_name = image_data['image']
            ground_truth = image_data['ground_truth']
            if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Read image
            
                image = cv2.imread(image_name)



                jp_ground_truth = ground_truth[:6]
                class_ = ground_truth[6]
                

                boxes = ground_truth[7:]

                xn, yn, wn, hn = boxes

                # Convert xywhn to xyxyn format
                x1 = xn - (wn / 2)
                y1 = yn - (hn / 2)
                x2 = xn + (wn / 2)
                y2 = yn + (hn / 2)

                # Create the bounding box in xyxyn format
                bbox_xyxyn = torch.tensor([x1, y1, x2, y2], dtype=torch.float32)



                bbox_tensor = torch.tensor([xn, yn, wn, hn], dtype=torch.float32)

                bboxs = torch.cat((bbox_tensor, bbox_xyxyn), 0).unsqueeze(0)



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

                

                
                with torch.no_grad():
                    img_batched = torch.randn(1,3,256,256)
                    bboxs = torch.randn(1,8)
                    outputs = model_Efficient_Net(img_batched, bboxs)
                    jp = reverse_standard_scaling(mean,std,outputs.numpy())[0]
                    y_pred.append(jp)
                    y_true.append(jp_ground_truth)
                    
                    mape = mean_absolute_percentage_error(jp_ground_truth, jp)
                    mae = mean_absolute_error(jp_ground_truth, jp)
                    writer.writerow([image_name, class_, mape, mae])




if __name__ == "__main__":



    state_dict = torch.load("/Users/matthiaspetry/Desktop/fastvit_t8_114_2.pth",map_location=torch.device('cpu'))

    model_Efficient_Net = LayerNormFastViT6DPosition()

    model_Efficient_Net.load_state_dict(state_dict)
    model_Efficient_Net.to("cpu")
    model_Efficient_Net.eval()
    
    annotate_and_save_images("/Users/matthiaspetry/Desktop/afd.v16i.yolov8/train/images/positions.txt", model_Efficient_Net)
    

    print(mean_absolute_percentage_error(y_true, y_pred))
    print(mean_absolute_error(y_true, y_pred))
    print(r2_score(y_true, y_pred))
    