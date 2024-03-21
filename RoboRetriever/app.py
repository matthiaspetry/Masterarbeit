from flask import Flask, jsonify, request, render_template,Response
from flask_cors import CORS
import threading
import time
from rtde_control import RTDEControlInterface as RTDEControl
from rtde_receive import RTDEReceiveInterface as RTDEReceive
import robotiq_gripper
import numpy as np
import cv2
from ultralytics import YOLO
import torch
import torch.nn as nn
import timm
from torchvision import transforms
from PIL import Image
import roboticstoolbox as rtb
import random 
import pickle
import math

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Shared variables
selected_object = None
selected_model = "None"
selected_speed = "None"
is_processing_active = False
rtde_r = None
rtde_c = None
rounded_position = None
pickuptime = None
roboConnection = False
gripperConnection = False
objectPickedUp = False
objectPlacePosition = [-1.72233421, -1.57549443,  1.25739509, -1.22777946, -1.61194212, -0.13999015]
objectPlacePositionSet = False
count = 0

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

with open("/Users/matthiaspetry/Desktop/Masterarbeit/calibration.pkl", "rb") as file:
    cameraMatrix, dist = pickle.load(file)

mapx, mapy, roi = init_undistortion_maps(cameraMatrix, dist, 1920, 1080)



"""mean =0.4112082047842611
std = 1.9551321360136171"""

"""mean = 0.4272211200448766 
std = 1.9514025672861495"""

mean = 0.3665011391762391
std = 1.8153700051203192


def reverse_standard_scaling(mean, std, scaled_data):
        original_data = [(val * std) + mean for val in scaled_data]
        return original_data

class LayerNormFastViT3DPosition(nn.Module):
    def __init__(self, dropout_rate=0.1, vector_input_size=8, intermediate_size=128, hidden_layer_size=64):
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

def main_processing_loop():
    global is_processing_active, rtde_c, rtde_r
    global cap, rounded_position, pickuptime


    model = YOLO("yolov8s.yaml")  # build a new model from scratch
    model = YOLO("/Users/matthiaspetry/Desktop/working/runs/detect/train/weights/best.pt") 
    state_dict = torch.load("/Users/matthiaspetry/Desktop/T8_120_140_T8_newbbox.pth",map_location=torch.device('cpu'))
    # Assuming EfficientNet3DPosition is the class of your model
    joint_model = LayerNormFastViT3DPosition()
    # Load the state_dict into the model
    joint_model.load_state_dict(state_dict)
    joint_model.to("cpu")
    joint_model.eval()

    model_name = "resnet10t.c3_in1k"  # Example model name, change as per need
    model2 = timm.create_model(model_name, pretrained=False, num_classes=2)
    state_dict2 = torch.load("/Users/matthiaspetry/Desktop/binary_classification_res_new.pth",map_location=torch.device('cpu'))
    # Assuming EfficientNet3DPosition is the class of your model
    model2.load_state_dict(state_dict2)
    model2.to("cpu")
    model2.eval()
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    if not cap.isOpened():
        print("Cannot open camera")
        exit() 
    # Parameters
    velocity = 0.1
    acceleration = 0.1
    dt = 0.1
    lookahead_time = 0.2
    gain = 2000
    joint_q = [0.0000,-1.5708,-0.0000,-1.5708,-0.0000,0.0000]

    # Move to initial joint position with a regular moveJ
    detected = False

    grabcount = 0
    mode = "dynamic"

    while True:


        if mode == "static":
            pass


        elif mode == "dynamic": 
            
           
            if is_processing_active:

                if not gripper:
                    print("Error: gripper is not initialized. Please connect to the gripper before using it.")
                    return

                # Check if rtde_c is initialized
                if not rtde_c:
                    print("Error: rtde_c is not initialized. Please connect to the robot before using it.")
                    return

                if not rtde_r:
                    print("Error: rtde_r is not initialized. Please connect to the robot before using it.")
                    return

                
                if not selected_object:
                    print(selected_object)
                    



                
                t_start = rtde_c.initPeriod()
                ret, frame = cap.read()


                
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
                        if detected == False:
                            detected = True
                            startTime = time.time()
                        if cls == int(selected_object):
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
                
                                outputs = joint_model(img_batched, bboxs)
                                
                                # Reverse scaling to get the actual joint positions
                                jp = reverse_standard_scaling(mean, std, outputs.numpy())[0]

                                end_effector_pose = robot.fkine(jp)

                                current_position = rtde_r.getActualQ()
                                #current_position_pose = robot.fkine(current_position)


                                if end_effector_pose.t[2] > 0.50:
                                    
                                
                                    
                                    #a = rtb.jtraj(current_position,jp,5)
                                    a = rtb.mtraj(rtb.quintic, current_position, jp, 100)
                                    #a = rtb.jtraj( current_position, jp, 10)
                                    for position in a.q:
                                        print(position)
                                
                                    
                                        rtde_c.servoJ(position, velocity, acceleration, dt, lookahead_time, gain)

                                    
                                else: #end_effector_pose.t[2]<0.5:
                                    
                                    
                                    end_effector_pose.t[0] = end_effector_pose.t[0] - (10* 0.01)
                                    end_effector_pose.t[1] = end_effector_pose.t[1]
                                    
                                    sol = robot.ikine_LM(end_effector_pose,jp)
                                    jp = sol.q
                                    
                                    
                                    rtde_c.servoJ(jp, velocity, acceleration, dt, lookahead_time, gain)
                                        

                                    pred = model2(img_batched2)
                                    grab = torch.argmax(pred, dim=1).numpy()
                                    if grab[0] == 0:
                                        print(grabcount)
                                        grabcount += 1
                                    if grab[0] == 0 and grabcount == 2:
                                        
                                        print("Grab")
                                        gripper.move_and_wait_for_pos(255, 255, 255)
                                    
                                        position = rtde_r.getActualQ()
                                        pickuptime = time.time() - startTime

    
                                        rounded_position = [round(p, 4) for p in position]
                                       
                                        position[1] -= 10 * (math.pi / 180)
                                        rtde_c.servoStop()
                                        rtde_c.moveJ(position,3,3)
                                        del position
                                        
                                    
                                        #rtde_c.moveJ(position,3,3)
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
                                
                                rtde_c.waitPeriod(t_start)

                                if cv2.waitKey(1) == ord('q'):
                                    rtde_c.servoStop()
                                    rtde_c.stopScript()
                                    cap.release()
                                    cv2.destroyAllWindows()
                                
    
                                    break

cap2 = cv2.VideoCapture(2)
cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
def gen_frames(cap2):  
    while True:
        success, frame = cap2.read()  # read the camera frame
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(cap2), mimetype='multipart/x-mixed-replace; boundary=frame')
            

@app.route('/start_processing', methods=['POST'])
def start_processing():
    global is_processing_active
    is_processing_active = True
    return jsonify({'message': 'Processing loop started'})

@app.route('/stop_processing', methods=['POST'])
def stop_processing():
    global is_processing_active
    global rtde_c
    rtde_c.servoStop()
    rtde_c.stopScript()
    is_processing_active = False
    return jsonify({'message': 'Processing loop stopped'})

@app.route('/select_object', methods=['POST'])
def select_object():
    global selected_object
    data = request.json
    selected_object = data.get('object_type')
    print(selected_object)
    return jsonify({'message': f'Selected object set to {selected_object}'})

@app.route('/select_speed', methods=['POST'])
def select_speed():
    global selected_speed
    data = request.json
    selected_speed = data.get('object_type')
    print(selected_speed)
    return jsonify({'message': f'Selected speed set to {selected_speed}'})

@app.route('/select_model', methods=['POST'])
def select_model():
    global selected_model
    data = request.json
    selected_model = data.get('object_type')
    print(selected_model)
    return jsonify({'message': f'Selected model set to {selected_model}'})

@app.route('/connect_robot', methods=['POST'])
def connect_robot():
    ROBOT_IP ="192.168.188.32"
    global rtde_c
    global rtde_r
    global gripper
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
    global roboConnection
    roboConnection = True
    global gripperConnection
    gripperConnection = True





    return jsonify({'message': f'Robot connected'})

@app.route('/move_2_base', methods=['POST'])
def move_2_base():
    rtde_c.moveJ([0.0000,-1.5708,-0.0000,-1.5708,-0.0000,0.0000])

    return jsonify({'message': f'Moved to base'})


@app.route('/set_position', methods=['POST'])
def set_position():
    #rtde_c.endFreedriveMode()
    global objectPlacePosition
    global objectPlacePositionSet
    #bjectPlacePosition = rtde_r.getActualQ()
    objectPlacePositionSet = True
    global is_processing_active
    is_processing_active = False

    return jsonify({'message': f'Position Set'})

@app.route('/free_mode', methods=['POST'])
def free_mode():
    global is_processing_active
    is_processing_active = "FreeDrive"
    #rtde_c.freedriveMode()
    return jsonify({'message': f'freedrive'})


@app.route('/dataTime')
def dataTime():
    global count
    global pickuptime

    if rounded_position is not None:  # Check if rounded_position is not None
        try:
            data = {'name': str(count), 'Time': pickuptime}
            pickuptime = None
            count += 1
            
            return jsonify(data)
        except Exception as e:  # Handle potential exceptions during FKine calculation
            print(f"Error calculating end effector pose: {e}")
            return jsonify({'error': 'Error calculating position'}), 500  # Return error message with status code 500
    
    return jsonify({'message': 'No pick up time  available'}), 204  # Return empty response with status code 204 (No Content)

@app.route('/dataPosition')
def dataPosition():
    global rounded_position

    if rounded_position is not None:  # Check if rounded_position is not None
        try:
            end_effector_pose = robot.fkine(rounded_position)
            data = {'x': end_effector_pose.t[0], 'y': end_effector_pose.t[1]}
            return jsonify(data)
        except Exception as e:  # Handle potential exceptions during FKine calculation
            print(f"Error calculating end effector pose: {e}")
            return jsonify({'error': 'Error calculating position'}), 500  # Return error message with status code 500

    return jsonify({'message': 'No position data available'}), 204  # Return empty response with status code 204 (No Content)




@app.route('/status')
def status():
    # Example data, replace with your actual data source
    if selected_object == None:
        selected = None
    else:
        selected = int(selected_object)

    if selected == 0:
        obj = "Cross"
    elif selected == 1:
        obj = "Cube"
    elif selected == 2:
        obj = "Cylinder"
    elif selected == 3:
        obj = "Hexagon"
    elif selected == 4:
        obj = "Pyramid"
    elif selected == 5:
        obj = "R_Cylinder"
    elif selected == 6:
        obj = "Star"
    elif selected == 7:
        obj = "Y_Cube"
    elif selected == None:
        obj = "Not Selected"


    data = {
        'isConnected': roboConnection,
        'gripperConnection': gripperConnection,
        'operationalStatus': is_processing_active,
        'currentTask': obj,
        'errorStatus': 'No Errors',
        "objectPickedUp": objectPickedUp,
        "objectPlacePosition": objectPlacePositionSet,
        "selectedModel": selected_model,
        "selectedSpeed": selected_speed,
        
    }
    return jsonify(data)


@app.route('/')
def index():
    return render_template('index.html')

processing_thread = threading.Thread(target=main_processing_loop)
processing_thread.daemon = True
processing_thread.start()

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
