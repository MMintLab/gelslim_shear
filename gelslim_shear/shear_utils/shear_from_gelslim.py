import numpy as np

from std_msgs.msg import Bool

import rospy

from gelslim_utils.camera_parsers.gelslim_camera_parser import GelslimCameraParser
from gelslim_utils.msg import CompleteShear, EssentialShear, DivergenceShear, DifferentialShear

import numpy as np
import scipy.interpolate as interp
import cv2

import torch

import skimage.measure

class ShearGenerator():
    def __init__(self, num_fingers = 2, method: str = 'weighted', channels = ['u','v','div','curl','sol_u','sol_v','irr_u','irr_v','dudt','dvdt'],
                  displacement_threshold: float = 0.1, displacement_midpoint: float = 0.5,
                    reset_threshold = 0.15, pool_kernel_size:int = 25,
                    Farneback_params = (0.5, 3, 15, 3, 5, 1.2, 0),
                      output_size = (13,18), #output size of shear data, 13x18 is the default, in the form (height, width)
                      auto_reset = True,
                      auto_reset_period = 15):
        self.output_size = output_size

        accepted_channels = ['u','v','div','curl','sol_u','sol_v','irr_u','irr_v','dudt','dvdt']
        for channel in channels:
            if channel not in accepted_channels:
                raise ValueError('channel must be one of u, v, div, curl, sol_u, sol_v, irr_u, irr_v, dudt, dvdt')

        #if any of channels contain sol or irr
        if any(['sol' in channel for channel in channels]) or any(['irr' in channel for channel in channels]):
            try:
                from pynhhd import nHHD
            except:
                print("pynhhd not installed, install from https://github.com/bhatiaharsh/naturalHHD")
            self.nhhd = nHHD(grid=output_size, spacings=(1,1))
        if method not in ['weighted','1','2']:
            raise ValueError('method must be weighted, 1, 2')
        self.method = method
    
        self.displacement_threshold = displacement_threshold
        self.displacement_midpoint = displacement_midpoint
        self.reset_threshold = reset_threshold
        self.pool_kernel_size = pool_kernel_size
        #Farneback params of the form (pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags)
        self.Farneback_params = Farneback_params
        
        self.channels = channels
        self.num_channels = len(channels)
        
        self.shear_field_tensor = torch.zeros((num_fingers*self.num_channels, self.output_size[0], self.output_size[1]))
        
        if auto_reset:
            self.prev_displacement2 = 0
            self.auto_reset = True
            self.auto_reset_period = auto_reset_period

    def curl(self, u, v):
        #estimate the partial derivatives
        dudy = torch.gradient(u, axis=0)
        dvdx = torch.gradient(v, axis=1)
        #estimate the curl
        curl = dvdx - dudy
        return curl

    def divergence(self, u, v):
        #estimate the partial derivatives
        dudx = np.gradient(u, axis=1)
        dvdy = np.gradient(v, axis=0)
        #estimate the divergence
        divergence = dudx + dvdy
        return divergence

    def sigmoid_weighting_function(self, x, midpoint, threshold):
        y = 1/(1+np.exp(-1*(x-midpoint)/threshold))
        return y

    def get_grayscale(self,camera_parser):
        image = camera_parser.get_image_color()
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return gray

    def reset_shear(self):
        self.no_shear_image = self.get_grayscale(self.camera_parser)
        if self.method == 'weighted' or self.method == '1':
            self.ulist1 = np.zeros((self.output_size[0],self.output_size[1],self.SMA_window))
            self.vlist1 = np.zeros((self.output_size[0],self.output_size[1],self.SMA_window))
            self.u1 = np.zeros(self.output_size)
            self.v1 = np.zeros(self.output_size)
        if self.method == 'weighted' or self.method == '2':
            self.ulist2 = np.zeros((self.output_size[0],self.output_size[1],self.SMA_window))
            self.vlist2 = np.zeros((self.output_size[0],self.output_size[1],self.SMA_window))
            self.u2 = np.zeros(self.output_size)
            self.v2 = np.zeros(self.output_size)
        if self.method == 'weighted':
            self.u = np.zeros(self.output_size)
            self.v = np.zeros(self.output_size)
        self.last_reset_time = rospy.get_time()

    def extract_flow_field_filtered(self, ref, image, u_list, v_list):
        flow = cv2.calcOpticalFlowFarneback(ref, image, None, self.Farneback_params[0], self.Farneback_params[1], self.Farneback_params[2], self.Farneback_params[3], self.Farneback_params[4], self.Farneback_params[5], self.Farneback_params[6])
        flow_dwn = skimage.measure.block_reduce(flow, (self.pool_kernel_size,self.pool_kernel_size,1), np.mean)
        #reshape flow_dwn to have self.output_size
        flow_dwn = cv2.resize(flow_dwn, (self.output_size[1],self.output_size[0]))
        u = flow_dwn[:,:,0]
        v = flow_dwn[:,:,1]
        #roll the list
        u_list = np.roll(u_list,1,axis=2)
        v_list = np.roll(v_list,1,axis=2)
        #add new values
        u_list[:,:,0] = u
        v_list[:,:,0] = v
        #calculate mean
        u = np.mean(u_list, axis=2)
        v = np.mean(v_list, axis=2)
        return u, v
    
    def reset_callback(self, msg):
        if msg.data:
            self.reset_shear()
    
    def broadcast_shear(self):
        if self.method == 'weighted' or self.method == '1':
            udiff1, vdiff1 = self.extract_flow_field_filtered(self.prev_shear_image, self.image, self.ulist1, self.vlist1)
            self.u1 = self.u1 + udiff1
            self.v1 = self.v1 + vdiff1
        if self.method == 'weighted' or self.method == '2':
            u2, v2 = self.extract_flow_field_filtered(self.no_shear_image, self.image, self.ulist2, self.vlist2)
            self.u2 = u2
            self.v2 = v2
        if self.method == 'weighted':
            weight = self.sigmoid_weighting_function(self.displacement, self.displacement_midpoint, self.displacement_threshold)
            self.u = weight*self.u1 + (1-weight)*self.u2
            self.v = weight*self.v1 + (1-weight)*self.v2
            self.displacement = np.max(np.array([np.std(self.u), np.std(self.v)]))
            if self.auto_reset:
                displacement2 = np.max(np.array([np.std(self.u2), np.std(self.v2)]))
                if displacement2 < self.reset_threshold:
                    if rospy.get_time() - self.last_reset_time > self.auto_reset_period and self.prev_displacement2 >= self.reset_threshold:
                        self.reset_shear()
                        self.last_reset_time = rospy.get_time()
                self.prev_displacement2 = displacement2
        if self.method == '1':
            self.u = self.u1
            self.v = self.v1
        if self.method == '2':
            self.u = self.u2
            self.v = self.v2
        self.shear_msg.u = self.u.flatten()
        self.shear_msg.v = self.v.flatten()
        self.shear_msg.x = self.X.flatten()
        self.shear_msg.y = self.Y.flatten()
        if self.mode == 'divergence' or self.mode == 'differential':
            divergence = self.divergence(self.u, self.v, self.X, self.Y)
            self.shear_msg.divergence = divergence.flatten()
        if self.mode == 'complete':
            divergence = self.divergence(self.u, self.v, self.X, self.Y)
            curl = self.curl(self.u, self.v, self.X, self.Y)
            vfield = np.stack((self.u,self.v),axis=2)
            self.nhhd.decompose(vfield)
            solenoidal = self.nhhd.r
            irrotational = self.nhhd.d
            u_sol = solenoidal[:,:,0]
            v_sol = solenoidal[:,:,1]
            u_irr = irrotational[:,:,0]
            v_irr = irrotational[:,:,1]
            self.shear_msg.divergence = divergence.flatten()
            self.shear_msg.curl = curl.flatten()
            self.shear_msg.hhd_sol_u = u_sol.flatten()
            self.shear_msg.hhd_sol_v = v_sol.flatten()
            self.shear_msg.hhd_irr_u = u_irr.flatten()
            self.shear_msg.hhd_irr_v = v_irr.flatten()
        if self.mode == 'differential':
            if self.method == '2':
                u1, v1 = self.extract_flow_field_filtered(self.prev_shear_image, self.image, self.ulist1, self.vlist1)
                self.u1 = u1
                self.v1 = v1
                self.shear_msg.dudt = self.u1.flatten()
                self.shear_msg.dvdt = self.v1.flatten()
            else:
                raise ValueError('method must be 2 for differential shear')
        self.shear_data_pub.publish(self.shear_msg)
        self.prev_shear_image = self.image

    def update_gray(self):
        self.image = self.get_grayscale(self.camera_parser)