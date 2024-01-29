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


import sys

# Add the paths of multiple folders
folders_to_add = [
    '/Users/matthiaspetry/Desktop/Masterarbeit/robotiq_gripper/',
]



for folder in folders_to_add:
    sys.path.append(folder)

import robotiq_gripper
import time


model_name = "efficientvit_m0.r224_in1k"  # Example model name, change as per need
model2 = timm.create_model(model_name, pretrained=False, num_classes=2)
state_dict2 = torch.load("/Users/matthiaspetry/Desktop/binary_classification_eff_vit.pth",map_location=torch.device('cpu'))
# Assuming EfficientNet3DPosition is the class of your model
model2.load_state_dict(state_dict2)
model2.to("cpu")
model2.eval()


class LayerNormFastViT3DPosition(nn.Module):
    def __init__(self, dropout_rate=0.1, vector_input_size=4, intermediate_size=128, hidden_layer_size=64):
        super(LayerNormFastViT3DPosition, self).__init__()

        # Load FastViT model pre-trained on ImageNet
        self.fastvit = timm.create_model('fastvit_t8.apple_dist_in1k', pretrained=False)
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
        self.fastvit = timm.create_model('fastvit_sa12.apple_dist_in1k', pretrained=False)

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

model = YOLO("yolov8s.yaml")  # build a new model from scratch
model = YOLO("/Users/matthiaspetry/Desktop/Masterarbeit/Yolov8_best.pt") 

state_dict = torch.load("/Users/matthiaspetry/Desktop/fastvit37_t8.pth",map_location=torch.device('cpu'))

#onnx_session = onnxrt.InferenceSession("/Users/matthiaspetry/Downloads/fastvit12_t8-sim.onnx")

# Assuming EfficientNet3DPosition is the class of your model
joint_model = LayerNormFastViT3DPosition()
# Load the state_dict into the model
joint_model.load_state_dict(state_dict)

joint_model.to("cpu")
joint_model.eval()


cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)


mean =0.4112082047842611
std = 1.9551321360136171

def reverse_standard_scaling(mean, std, scaled_data):
        original_data = [(val * std) + mean for val in scaled_data]
        return original_data

fps_counter = 0
fps = 0
prev_time = time.time()


#robot = robot_init(host)

#robot.init_realtime_control()  # starts the realtime control loop on the Universal-Robot Controller
time.sleep(1) # just a short wait to make sure everything is initialised"""

ROBOT_IP ="192.168.188.32"

rtde_c = RTDEControl(ROBOT_IP)
rtde_r = RTDEReceive(ROBOT_IP)

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



# Parameters
velocity = 0.1
acceleration = 0.1
dt = 0.5
lookahead_time = 0.05
gain = 2000
joint_q = [0.0000,-1.5708,-0.0000,-1.5708,-0.0000,0.0000]

# Move to initial joint position with a regular moveJ
rtde_c.moveJ(joint_q)

counter = 679 


if not cap.isOpened():
    print("Cannot open camera")
    exit() 

grabcount = 0
while True:
    t_start = rtde_c.initPeriod()
    
    # Capture frame-by-frame
    ret, frame = cap.read()


    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    resized_frame = cv2.resize(frame, (512,288))
    results = model.predict(resized_frame, verbose=False, imgsz=512)
    object_detected = False
    start_time_loop = time.time()
    
    for result in results:
        boxes = result.boxes.xywhn.tolist()
        classes = result.boxes.cls.tolist()

        for i, cls in enumerate(classes):
            if cls == 7:


    
                object_detected = True

                xn, yn, wn, hn = boxes[i]

                # Convert normalized coordinates to pixel coordinates
                #x, y, w, h = int(xn * frame.shape[1]), int(yn * frame.shape[0]), int(wn * frame.shape[1]), int(hn * frame.shape[0])

                # Draw rectangle
                #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green box

                bbox_tensor = torch.tensor([xn, yn, wn, hn]).unsqueeze(0)   

                # Open the image with PIL
                start_process = time.time()

                target_size2 = (256, 256)

                # Convert PIL Image to NumPy array for OpenCV processing
                frame_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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

                resized_image2 = cv2.resize(frame_np, (224,224))
                       

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
                    outputs = joint_model(img_batched, bbox_tensor)
                    
                   
                    jp = reverse_standard_scaling(mean,std,outputs.numpy())[0]
                    rtde_c.servoJ(jp, velocity, acceleration, dt, lookahead_time, gain)
                    pred = model2(img_batched2)
                    grab = torch.argmax(pred, dim=1).numpy()
                    if grab[0] == 0:
                        
                        grabcount += 1
                    if grab[0] == 0 and grabcount == 3:
                        print("Grab")
                        gripper.move_and_wait_for_pos(255, 255, 255)
                        rtde_c.servoStop()
                            
                        rtde_c.moveJ([-1.72233421, -1.57549443,  1.25739509, -1.22777946, -1.61194212, -0.13999015])
                        gripper.move_and_wait_for_pos(0, 255, 255)
                        rtde_c.moveJ(joint_q)
                        grabcount = 0
                        print(grabcount)
                        
                    else:
                        gripper.move_and_wait_for_pos(0, 255, 255)   

                    

                    #cv2.imwrite(f'Gripper/image{counter}.png', framex)
                    #counter += 1
                    



                   
                    rtde_c.waitPeriod(t_start)


    # FPS Calculation
    current_time = time.time()
    fps_counter += 1
    if current_time - prev_time >= 1.0:  # Every second
        fps = fps_counter
        fps_counter = 0
        prev_time = current_time
    
    # Display FPS on the frame
    #cv2.putText(frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    #cv2.imshow('frame', frame)



    if cv2.waitKey(1) == ord('q'):
        rtde_c.servoStop()
        rtde_c.stopScript()
       
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
