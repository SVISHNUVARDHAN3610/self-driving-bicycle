import pybullet as p
import pybullet_data as pd
import time
import math
import numpy as np

class Env:
    def __init__(self):
        self.action_size = 7
        self.state_size = None
        self.model,self.target  = self.init(-9.8,[18,12,2])
    def init(self,gravity_val,tar_pos):
        # connect out envirnoment to the graphics
        client = p.connect(p.GUI)
        #adding data path
        add    = p.setAdditionalSearchPath(pd.getDataPath())
        #adding_gravity
        gravity= p.setGravity(0,0,gravity_val)
        #loading plane
        plane  = p.loadURDF("plane.urdf")
        #loading our cycle
        model  = p.loadURDF("cycle2.urdf")
        target = p.loadURDF("r2d2.urdf",tar_pos)
        return model,target
    def camera(self,position,orientation,distance):
        #position of the camera link
        xa,ya,za = position 
        #base orentation
        ang = orientation[2] + 1.5708
        # here we are adding 90 degress to base orentation
        #angel and distanc calculation part
        xb  = xa - distance * np.cos(ang)
        yb  = ya - distance*np.sin(ang)
        zb  = za
        target = [xb,yb,zb]
        up_vector = [0,0,2] # camera point towards up 
        # HD  image resolution 1280 * 720 { normal HD only}
        # for full HD resolution 1980 * 1080 but while captuing with full HD it required high specified computer
        # which we  cant afford 
        width,height = 240,240
        # view matrix which is 4*4 matrix = 16 floats
        view_matrix  = p.computeViewMatrix(cameraEyePosition = position,
                                                    cameraTargetPosition =target,
                                                    cameraUpVector = up_vector)
        # projection matrix which is 4*4 matrix = 16 floats
        proj_matrix  = p.computeProjectionMatrixFOV(fov = 90 ,
                                                    aspect = 1.5,
                                                    nearVal = 0.05,
                                                    farVal =100 )# adjusting camera to capture  upto 100 units[metes]
        image        = p.getCameraImage(width = width,height =height,
                                                    viewMatrix =view_matrix,
                                                    projectionMatrix =proj_matrix,
                                                    shadow = True,
                                                    renderer=p.ER_BULLET_HARDWARE_OPENGL)#here rendering for smooth performance
        #returning only RBG values or pickels
        return image[2]
    def laser(self,position1,position2,orientation,distance):
        #postion1 is postion of laser1
        #laser calculation
        xa,ya,za = position1
        ang = orientation[2]
        xb  = xa - distance * np.cos(ang)
        yb  = ya - distance*np.sin(ang)
        zb  = za
        target = [xb,yb,zb]
        laser1 = p.rayTest(position1,target)
        #drawing a line for laser visulation
        p.addUserDebugLine(lineFromXYZ= position1,
                                lineToXYZ= target,
                                    lineColorRGB = [1,0,0],
                                        lifeTime = 0.1)
        #position2 is the position of laser2
        xa,ya,za = position2
        ang = orientation[2]
        #laser caculation
        xb  = xa + distance * np.cos(ang)
        yb  = ya + distance*np.sin(ang)
        zb  = za
        target = [xb,yb,zb]
        laser2 = p.rayTest(position2,target)
        #drawing a line for laser visulation
        p.addUserDebugLine(lineFromXYZ= position2,
                                lineToXYZ= target,
                                    lineColorRGB = [1,0,0],
                                        lifeTime = 0.1)
        return laser1[0][3],laser2[0][3]
    def angle_calculation(self,orientation):
        angle = []
        for i in range(3):
            ore  = orientation[i]*180/np.pi
            angle.append(ore)
        # returning angle in degree 
        # rather than radians for better understanding
        return angle
    def reward_calculation(self,position1,position2,ore):
        #position of our model{"base"}
        base_position = position1
        #position of target{"base"}
        target_postion= position2
        #distance btween them
        distance = np.sqrt(np.square(target_postion[0]-base_position[0])+
                                        np.square(target_postion[1]-base_position[1])+
                                        np.square(target_postion[2]-base_position[2]))
        def dista(distance):
            if distance<=5.5:
                dist_reward = distance
            else:
                dist_reward = -distance
            return dist_reward 
        def angel_reward(angle):
            if angle>=-5 or angle<=5:
                ang_reward = 5
            elif angle>=-75 or angle<=75:
                ang_reward = 5-angle*12/100
            else:
                ang_reward = -8.5
            return ang_reward
        reward = dista(distance=distance)+ angel_reward(ore)

        return reward
    def reset(self):
        
        state        = []
        sub_state    = []
        cam_pos      = p.getLinkState(self.model,7)[0]
        base,ore     = p.getBasePositionAndOrientation(self.model)
        target_postion,_    = p.getBasePositionAndOrientation(self.target)
        orientation  = p.getEulerFromQuaternion(ore)
        distance     = 50
        image        = self.camera(position= cam_pos,
                                        orientation = orientation,
                                            distance =distance)
        l1_position  = p.getLinkState(self.model,8)[0]
        l2_position  = p.getLinkState(self.model,9)[0]
        laser1,laser2= self.laser(position1= l1_position, position2= l2_position,
                                                 orientation= orientation, distance= distance)
        distance     = np.sqrt(np.square(target_postion[0]-base[0])+
                                        np.square(target_postion[1]-base[1])+
                                        np.square(target_postion[2]-base[2]))
        for i in range(3):
            sub_state.append(base[i])
        for i in range(3):
            sub_state.append(target_postion[i])
        for i in range(3):
            sub_state.append(orientation[i])
        sub_state.append(distance)
        for i in range(3):
            sub_state.append(laser1[i])
        for i in range(3):
            sub_state.append(laser2[i])

        state.append(image)
        state.append(sub_state)
        return state
    def step(self,action):
        #joint 0 is streeing
        #joint 2 is front_wheel
        #joint 4 is pendulum
        #joint 6 is rare_wheel
        #joint 7 is camera
        #joint 8 is sensor1
        #joint 9 is sensor2
        #the action contain set of 6 values 
        #action[0],action[1] is for streeing
        #action[2],action[3] is for front_wheel
        #action[4],action[5] is for pendulum
        #action[6],action[7] is for rare_wheel
        p.setJointMotorControl2(bodyUniqueId = self.model,jointIndex =6,
                                        controlMode=p.VELOCITY_CONTROL,
                                        targetVelocity = action[6],
                                        force = action[7])#rare_wheel
        p.setJointMotorControl2(bodyUniqueId = self.model,jointIndex =2,
                                        controlMode=p.VELOCITY_CONTROL,
                                        targetVelocity = action[2],
                                        force = action[3])#front_wheel
        p.setJointMotorControl2(bodyUniqueId = self.model,jointIndex =4,
                                        controlMode=p.VELOCITY_CONTROL,
                                        targetVelocity = action[4],
                                        force = action[5])#pendulum
        p.setJointMotorControl2(bodyUniqueId = self.model,jointIndex =0,
                                        controlMode=p.VELOCITY_CONTROL,
                                        targetVelocity = action[0],
                                        force = action[1])#streeing
        cam_pos      = p.getLinkState(self.model,7)[0]
        base,ore     = p.getBasePositionAndOrientation(self.model)
        orientation  = p.getEulerFromQuaternion(ore)
        distance     = 50
        image        = self.camera(position= cam_pos,
                                        orientation = orientation,
                                            distance =distance)
        l1_position  = p.getLinkState(self.model,8)[0]
        l2_position  = p.getLinkState(self.model,9)[0]
        laser1,laser2= self.laser(position1= l1_position, position2= l2_position,
                                                 orientation= orientation, distance= distance)
        info         = self.angle_calculation(orientation=orientation)
        speed        = p.getBaseVelocity(self.model)[0]
        angle        = info[1]
        if angle<0:
            angle = -angle
        else:
            angle = angle
        tar_pos,_ = p.getBasePositionAndOrientation(self.target)
        reward    = self.reward_calculation(position1=base,
                                                position2=tar_pos,
                                                    ore = angle)
        state = []
        sub_state = []
        target_postion = tar_pos
        distance     = np.sqrt(np.square(target_postion[0]-base[0])+
                                np.square(target_postion[1]-base[1])+
                                np.square(target_postion[2]-base[2]))
        for i in range(3):
            sub_state.append(base[i])
        for i in range(3):
            sub_state.append(target_postion[i])
        for i in range(3):
            sub_state.append(orientation[i])
        sub_state.append(distance)
        for i in range(3):
            sub_state.append(laser1[i])
        for i in range(3):
            sub_state.append(laser2[i])
        def done_statement(dista):
            if dista<=1.5:
                done = True
            else:
                done = False
            return done
        state.append(image)
        state.append(sub_state)
        return state,reward,done_statement(distance),info,speed
    def close(self):
        p.disconnect()
    def simulate(self):
        p.setRealTimeSimulation(1)
