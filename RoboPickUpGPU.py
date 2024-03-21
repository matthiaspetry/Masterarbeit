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
import pickle
import math
import roboticstoolbox as rtb
import numpy as np
import keyboard


import sys

# Add the paths of multiple folders
folders_to_add = [
    '/Users/matthiaspetry/Desktop/Masterarbeit/robotiq_gripper/',
]

import glob

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

model_name = "resnet10t.c3_in1k"  # Example model name, change as per need
model2 = timm.create_model(model_name, pretrained=False, num_classes=2)
state_dict2 = torch.load("/home/localedge2/Masterarbeit/models/DesicionModel.pth",map_location=torch.device('cuda'))
# Assuming EfficientNet3DPosition is the class of your model
model2.load_state_dict(state_dict2)
model2.to("cuda")
model2.eval()

def init_undistortion_maps(cameraMatrix, dist, width, height):
    newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(cameraMatrix, dist, (width, height), 1, (width, height))
    mapx, mapy = cv2.initUndistortRectifyMap(cameraMatrix, dist, None, newCameraMatrix, (width, height), 5)
    return mapx, mapy, roi

def fast_undistort_image(img, mapx, mapy, roi):
    dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    return dst



class LayerNormFastViT6DPosition(nn.Module):
    def __init__(self, dropout_rate=0.1, vector_input_size=8, intermediate_size=128, hidden_layer_size=64):
        super(LayerNormFastViT6DPosition, self).__init__()

        # Load FastViT model pre-trained on ImageNet
        self.fastvit = timm.create_model('fastvit_t8.apple_dist_in1k', pretrained=False) 
        #self.fastvit = timm.create_model('resnet18.a1_in1k', pretrained=False)

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
model = YOLO("/home/localedge2/Masterarbeit/models/best223.pt") 
model.to("cuda")
state_dict = torch.load("/home/localedge2/Masterarbeit/models/T8_86_145.pth",map_location=torch.device('cuda'))

# Assuming EfficientNet3DPosition is the class of your model
joint_model = LayerNormFastViT6DPosition()
# Load the state_dict into the model
joint_model.load_state_dict(state_dict)

joint_model.to("cuda")
joint_model.eval()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

#mean = -0.015857931462808714 # T8_201_141
# std = 1.479495674591031

mean = -0.02049437658091184
std = 1.4781722524455945

def reverse_standard_scaling(mean, std, scaled_data):
        original_data = [(val * std) + mean for val in scaled_data]
        return original_data

fps_counter = 0
fps = 0
prev_time = time.time()


ROBOT_IP ="192.168.188.32"

rtde_c = RTDEControl(ROBOT_IP)
rtde_r = RTDEReceive(ROBOT_IP)



print("Creating gripper...")
gripper = robotiq_gripper.RobotiqGripper()
print("Connecting to gripper...")
gripper.connect(ROBOT_IP, 63352)
print("Activating gripper...")
gripper.activate()

print("Testing gripper...")
gripper.move_and_wait_for_pos(255, 255, 255)
#log_info(gripper)
gripper.move_and_wait_for_pos(0, 255, 255)
#log_info(gripper)

#with open("/Users/matthiaspetry/Desktop/Masterarbeit/calibration.pkl", "rb") as file:
#    cameraMatrix, dist = pickle.load(file)

#mapx, mapy, roi = init_undistortion_maps(cameraMatrix, dist, 1920, 1080)

# Parameters
velocity = 3
acceleration = 3
dt = 0.05
lookahead_time = 0.2
gain = 2000
joint_q = [0.0000,-1.5708,-0.0000,-1.5708,-0.0000,0.0000]

rtde_c.moveJ(joint_q)

counter = 174


if not cap.isOpened():
    print("Cannot open camera")
    exit() 

grabcount = 0

addcount = 1
detected = False
startTime = None
released = False

sequnce = []
while True:
    t_start = rtde_c.initPeriod()
    
    # Capture frame-by-frame
    ret, frame = cap.read()

    #dst = fast_undistort_image(frame, mapx, mapy, roi)

    if rtde_r.isProtectiveStopped():
        print("Protective Stop")


    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    resized_frame = cv2.resize(frame, (512,288))
    results = model.predict(resized_frame, verbose=True, imgsz=512,conf=0.80, device="cuda")
    object_detected = False
    start_time_loop = time.time()
   
    




    for result in results:
        print(result.boxes.conf)
        print(result.boxes.cls)
        boxes = result.boxes.xywhn.tolist()
        classes = result.boxes.cls.tolist()

        
        

        for i, cls in enumerate(classes):
            if detected == False:
                detected = True
                startTime = time.time()
            if cls == 1: #or cls == 1 or cls == 2:


    
                object_detected = True

                xn, yn, wn, hn = boxes[i]


                bbox_tensor = torch.tensor([xn, yn, wn, hn])

                x1 = xn - (wn / 2)
                y1 = yn - (hn / 2)
                x2 = xn + (wn / 2)
                y2 = yn + (hn / 2)

                # Create the bounding box in xyxyn format
                bbox_xyxyn = torch.tensor([x1, y1, x2, y2], dtype=torch.float32)

                bboxs = torch.cat((bbox_tensor, bbox_xyxyn), 0).unsqueeze(0)

                # Open the image with PIL
                start_process = time.time()

                target_size2 = (256, 256)

                # Convert PIL Image to NumPy array for OpenCV processing
                #frame_np = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
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
                    
                    outputs = joint_model(img_batched.to("cuda"), bboxs.to("cuda"))
                    
                    jp = reverse_standard_scaling(mean, std, outputs.cpu().numpy())[0]

                    end_effector_pose = robot.fkine(jp)

                    current_position = rtde_r.getActualQ()


                    if end_effector_pose.t[2] > 0.50:
                    
                        a = rtb.mtraj(rtb.quintic, current_position, jp, 50)
                        #a = rtb.jtraj( current_position, jp, 10)
                        for position in a.q:                       
                            rtde_c.servoJ(position, velocity, acceleration, dt, lookahead_time, gain)
                    else: 
                    
                        end_effector_pose.t[0] = end_effector_pose.t[0] - (8* 0.01)
                        end_effector_pose.t[1] = end_effector_pose.t[1] 
                        
                        sol = robot.ikine_LM(end_effector_pose,jp)
                        jp = sol.q
                        
                        
                        rtde_c.servoJ(jp, velocity, acceleration, dt, lookahead_time, gain)
                            

                        pred = model2(img_batched2.to("cuda"))
                        grab = torch.argmax(pred, dim=1).cpu().numpy()
                        if grab[0] == 0:
                            print(grabcount)
                            grabcount += 1
                        if grab[0] == 0 and grabcount == 2:
                            
                            print("Grab")
                            gripper.move_and_wait_for_pos(255, 255, 255)
                            
                            
                            
                            
                            
                            
                            position = rtde_r.getActualQ()
                            #print(f"Position : {position}")
                            pickuptime = time.time() - startTime
                            #print(f"Dynamic Time: {pickuptime}")
                            rounded_position = [round(p, 4) for p in position]
                            #with open("/Users/matthiaspetry/Desktop/Masterarbeit/Y_CubeDynamicEvaluation4mMultiple.txt", 'a') as text_file:
                                #positions_str1 = f"Position {counter}: "
                                #positions_str2 = ",".join(map(str, rounded_position))
                                #text_file.write(positions_str1 + positions_str2 +","+str(pickuptime) + "\n")
                                #counter += 1
                           
                            position[1] -= 10 * (math.pi / 180)
                            rtde_c.servoStop()
                            rtde_c.moveJ(position,3,3)
                            del position
                            
                            detected = False
                            startTime = None
                            rtde_c.moveJ([-1.72233421, -1.57549443,  1.25739509, -1.22777946, -1.61194212, -0.13999015],3,3)
                            gripper.move_and_wait_for_pos(0, 255, 255)
                            rtde_c.moveJ(joint_q,3,3)
                            grabcount = 0
                            addcount = 1
                           
                            #print(grabcount)
                            
                        else:
                            gripper.move_and_wait_for_pos(0, 255, 255)  

                        

                        #cv2.imwrite(f'Gripper/image{counter}.jpg', frame)
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
    #cv2.putText(resized_frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    print(fps)
    cv2.imshow('frame', resized_frame)



    if cv2.waitKey(1) == ord('q'):
        rtde_c.servoStop()
        rtde_c.stopScript()
       
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
