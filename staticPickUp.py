import numpy as np
import cv2
import time
from ultralytics import YOLO
import torch
import torch.nn as nn
import timm
from torchvision import transforms
from PIL import Image
from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive
from dashboard_client import DashboardClient as DashboardClient
import pickle
import keyboard
import math
import roboticstoolbox as rtb
import sys
np.set_printoptions(formatter={'all': lambda x: format(round(x, 4), ',')})


# Add the paths of multiple folders
folders_to_add = [
    '/Users/matthiaspetry/Desktop/Masterarbeit/robotiq_gripper/',
]

for folder in folders_to_add:
    sys.path.append(folder)

import robotiq_gripper
import time

dh_params = np.array([
    [0, 0.15185,       0,   np.pi/2],
    [0, 0,      -0.24355,         0],
    [0, 0,       -0.2132,         0],
    [0, 0.13105,       0,   np.pi/2],
    [0, 0.08535,       0,  -np.pi/2],
    [0, 0.0921,        0,         0]
])

robot = rtb.DHRobot([
    rtb.RevoluteDH(d=dh_params[i, 1], a=dh_params[i, 2], alpha=dh_params[i, 3]) for i in range(6)
], name='UR3e')

def init_undistortion_maps(cameraMatrix, dist, width, height):
    newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, dist, (width, height), 1, (width, height))
    mapx, mapy = cv2.initUndistortRectifyMap(cameraMatrix, dist, None, newCameraMatrix, (width, height), 5)
    return mapx, mapy, roi

def fast_undistort_image(img, mapx, mapy, roi):
    dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    return dst
class EnhancedSimpleCNN(nn.Module):
    def __init__(self, in_channels=3, num_features=1024):  # Increased num_features
        super(EnhancedSimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),  # Added layer
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),  # Added layer
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, num_features, kernel_size=3, stride=2, padding=1, bias=False),  # Increased channels in the final conv layer
            nn.BatchNorm2d(num_features),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)  # Flatten the features to [batch_size, num_features]
        return x

class LayerNormFastViT6DPosition2(nn.Module):
    def __init__(self, dropout_rate=0.1, vector_input_size=8, intermediate_size=128, hidden_layer_size=64, in_channels=3, cnn_features=512):
        super(LayerNormFastViT6DPosition2, self).__init__()

        # The CNN part for feature extraction from images
        self.cnn = EnhancedSimpleCNN(in_channels=in_channels, num_features=cnn_features)

        # Model for processing 4D vector input with LayerNorm
        self.vector_model = nn.Sequential(
            nn.Linear(vector_input_size, intermediate_size),
            nn.ReLU(),
            nn.LayerNorm(intermediate_size),
            nn.Linear(intermediate_size, cnn_features),  # Ensure this matches the CNN output features
            nn.ReLU(),
            nn.LayerNorm(cnn_features),
            nn.Dropout(dropout_rate)
        )

        # Enhanced combined output layer with LayerNorm
        self.combined_output_layer = nn.Sequential(
            nn.Linear(cnn_features * 2, hidden_layer_size),  # Note the multiplication by 2 for concatenated features
            nn.ReLU(),
            nn.LayerNorm(hidden_layer_size),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_layer_size),
            nn.Linear(hidden_layer_size, 6)  # Output layer for a 6D vector
        )

    def forward(self, x, vector):
        # Extract features using the CNN
        cnn_features = self.cnn(x)

        # Process the 4D vector through the vector model
        vector_features = self.vector_model(vector)

        # Concatenate CNN and vector features
        concatenated_features = torch.cat((cnn_features, vector_features), dim=1)

        # Final output layer for regression
        final_output = self.combined_output_layer(concatenated_features)

        return final_output

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


model = YOLO("yolov8s.yaml")  # build a new model from scratch
model = YOLO("/Users/matthiaspetry/Desktop/working/runs/detect/train/weights/best.pt") 

#model = YOLO("/Users/matthiaspetry/Desktop/Masterarbeit/models/Yolov8_best.pt") 

#state_dict = torch.load("/Users/matthiaspetry/Desktop/T8_118_128.pth",map_location=torch.device('cpu'))
state_dict = torch.load("/Users/matthiaspetry/Desktop/T8_201_141.pth",map_location=torch.device('cpu'))
# Assuming EfficientNet3DPosition is the class of your mode
joint_model = LayerNormFastViT6DPosition()
# Load the state_dict into the model
joint_model.load_state_dict(state_dict)

joint_model.to("cpu")
joint_model.eval()


cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)


"""mean = 0.36281999857178565
std = 1.812439526258289""" # 4000 Datapoints


mean = -0.015857931462808714
std = 1.479495674591031


def reverse_standard_scaling(mean, std, scaled_data):
        original_data = [(val * std) + mean for val in scaled_data]
        return original_data

ROBOT_IP ="192.168.188.32"

rtde_c = RTDEControl(ROBOT_IP)
rtde_r = RTDEReceive(ROBOT_IP)
rtde_d = DashboardClient(ROBOT_IP)

def log_info(gripper):
    print(f"Pos: {str(gripper.get_current_position()): >3}  "
          f"Open: {gripper.is_open(): <2}  "
          f"Closed: {gripper.is_closed(): <2}  ")

print("Creating gripper...")
gripper = robotiq_gripper.RobotiqGripper()
print("Connecting to gripper...")
gripper.connect(ROBOT_IP, 63352)
print("Activating gripper...")
gripper.activate()

print("Testing gripper...")
gripper.move_and_wait_for_pos(255, 255, 255)
log_info(gripper)
gripper.move_and_wait_for_pos(0, 255, 255)
log_info(gripper)

with open("/Users/matthiaspetry/Desktop/Masterarbeit/calibration.pkl", "rb") as file:
    cameraMatrix, dist = pickle.load(file)

mapx, mapy, roi = init_undistortion_maps(cameraMatrix, dist, 1920, 1080)

# Parameters
velocity = 1
acceleration = 2

velocity2 = 3
acceleration2 = 5

dt = 0.1
lookahead_time = 0.05
gain = 2000
joint_q = [0.0000,-1.5708,-0.0000,-1.5708,-0.0000,0.0000]

# Move to initial joint position with a regular moveJ
rtde_c.moveJ(joint_q)
rtde_d.connect()
counter = 75

if not cap.isOpened():
    print("Cannot open camera")
    exit() 

key_press_time = time.time()  # Initialize time for key press
is_a_pressed = False  # Flag to tracxddstpfbk 'a' key press

while True:
        ret, frame = cap.read()

        

        dst = fast_undistort_image(frame, mapx, mapy, roi)

        resized_frame = cv2.resize(dst, (512,288))

        if rtde_r.isProtectiveStopped():
            print("Protective Stop")
            time.sleep(6)
            rtde_d.unlockProtectiveStop()
            print(rtde_c.isConnected())
            time.sleep(6)
            rtde_c.moveJ(joint_q)



        
        if keyboard.is_pressed('p') and not is_a_pressed:
            if time.time() - key_press_time >= 0.5:  # Adjust the delay as needed
                print("hier")
                # Save the current frame as an image
                results = model.predict(resized_frame, verbose=False, imgsz=512)

                for result in results:
                    boxes = result.boxes.xywhn.tolist()
                    classes = result.boxes.cls.tolist()

                    for i, cls in enumerate(classes):
                        if cls == 2:

                            object_detected = True

                            xn, yn, wn, hn = boxes[i]

                            # Convert normalized coordinates to pixel coordinates
                            #x, y, w, h = int(xn * frame.shape[1]), int(yn * frame.shape[0]), int(wn * frame.shape[1]), int(hn * frame.shape[0])

                            # Draw rectangle
                            #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green box

                            bbox_tensor = torch.tensor([xn, yn, wn, hn])

                            x1 = xn - (wn / 2)
                            y1 = yn - (hn / 2)
                            x2 = xn + (wn / 2)
                            y2 = yn + (hn / 2)

                            # Create the bounding box in xyxyn format
                            bbox_xyxyn = torch.tensor([x1, y1, x2, y2], dtype=torch.float32)

                            bboxs = torch.cat((bbox_tensor, bbox_xyxyn), 0).unsqueeze(0)
                            target_size2 = (256, 256)

                            # Convert PIL Image to NumPy array for OpenCV processing
                            frame_np = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
                            # Start the timer for resizing
                            resizedframe = cv2.resize(frame_np, target_size2)
                            # Convert back to PIL Image for further processing
                            img_pil = Image.fromarray(resizedframe)
                            # Start the timer for ToTensor conversion
                            to_tensor = transforms.ToTensor()
                            img_tensor = to_tensor(img_pil)
                            # Start the timer for normalization
                            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                            img_normalized = normalize(img_tensor)
                            img_batched = img_normalized.unsqueeze(0)
                            with torch.no_grad():
                    
                          
                                outputs = joint_model(img_batched, bboxs)
                                
                            
                                jp = reverse_standard_scaling(mean,std,outputs.numpy())[0]
                                
                                print(jp)
                                rounded_position = [round(p, 4) for p in jp]
                                with open("CylinderStaticMultiEvaluationT8.txt", 'a') as text_file:
                                    positions_str1 = f"Position {counter}: "
                                    positions_str2 = ",".join(map(str, rounded_position))
                                    text_file.write(positions_str1 + positions_str2 + "\n")
                                    counter += 1
                                between_jp = reverse_standard_scaling(mean,std,outputs.numpy())[0]
                                between_jp2 = reverse_standard_scaling(mean,std,outputs.numpy())[0]
                                between_jp[1] -= 5 * (math.pi / 180)
                                between_jp[2] -= 5 * (math.pi / 180)# Increase joint 2 by 10 degrees (converted to radians)
                                between_jp[3] += 5 * (math.pi / 180)
                                between_jp2[1] -= 10 * (math.pi / 180)
                                rtde_c.moveJ(between_jp, velocity2, acceleration2)
                                rtde_c.moveJ(jp, velocity, acceleration)

                                gripper.move_and_wait_for_pos(255, 255, 255)
                                rtde_c.moveJ(between_jp2, velocity, acceleration)

            
                                    
                                #rtde_c.moveJ([-1.72233421, -1.57549443,  1.25739509, -1.22777946, -1.61194212, -0.13999015],velocity2,acceleration2)
                                gripper.move_and_wait_for_pos(0, 255, 255)
                                rtde_c.moveJ(joint_q,velocity2, acceleration2)
                                
                         
                        

                is_a_pressed = False
                key_press_time = time.time()
        


        

        