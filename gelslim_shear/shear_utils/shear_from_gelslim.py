import numpy as np
import cv2

import torch

import skimage.measure

class ShearGenerator():
    def __init__(self, method: str = 'weighted', channels = ['u','v','div','curl','sol_u','sol_v','irr_u','irr_v','dudt','dvdt','du','dv'],
                  displacement_threshold: float = 0.1, displacement_midpoint: float = 0.5,
                    reset_threshold = 0.15, pool_kernel_size:int = 25,
                    Farneback_params = (0.5, 3, 15, 3, 5, 1.2, 0),
                      output_size = (13,18), #output size of shear data, 13x18 is the default, in the form (height, width)
                      auto_reset = True,
                      auto_reset_period = 15):
        self.output_size = output_size

        accepted_channels = ['u','v','div','curl','sol_u','sol_v','irr_u','irr_v','dudt','dvdt','du','dv']
        for channel in channels:
            if channel not in accepted_channels:
                raise ValueError('channel must be one of u, v, div, curl, sol_u, sol_v, irr_u, irr_v, dudt, dvdt, du, dv')

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
        
        self.shear_field_tensor = torch.zeros((self.num_channels, self.output_size[0], self.output_size[1]))

        self.tactile_image = None

        self.time = 0.0

        self.base_tactile_image = None

        self.prev_tactile_image = None

        self.prev_time = 0.0
        
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

    def get_grayscale(self,image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        return gray

    def reset_shear(self, base_tactile_image):
        self.update_base_tactile_image(base_tactile_image)
        if self.method == 'weighted' or self.method == '1':
            self.u1 = np.zeros(self.output_size)
            self.v1 = np.zeros(self.output_size)
        if self.method == 'weighted' or self.method == '2':
            self.u2 = np.zeros(self.output_size)
            self.v2 = np.zeros(self.output_size)
        if self.method == 'weighted':
            self.u = np.zeros(self.output_size)
            self.v = np.zeros(self.output_size)
        self.last_reset_time = self.time

    def extract_flow_field(self, ref, image):
        flow = cv2.calcOpticalFlowFarneback(ref, image, None, self.Farneback_params[0], self.Farneback_params[1], self.Farneback_params[2], self.Farneback_params[3], self.Farneback_params[4], self.Farneback_params[5], self.Farneback_params[6])
        flow_dwn = skimage.measure.block_reduce(flow, (self.pool_kernel_size,self.pool_kernel_size,1), np.mean)
        #reshape flow_dwn to have self.output_size
        flow_dwn = cv2.resize(flow_dwn, (self.output_size[1],self.output_size[0]))
        u = flow_dwn[:,:,0]
        v = flow_dwn[:,:,1]
        return u, v
    
    def update_shear(self):
        image = self.get_grayscale(self.tactile_image.permute(1,2,0).numpy().astype(np.uint8))
        if self.prev_tactile_image is None:
            self.prev_tactile_image = self.tactile_image
        prev_shear_image = self.get_grayscale(self.prev_tactile_image.permute(1,2,0).numpy().astype(np.uint8))
        no_shear_image = self.get_grayscale(self.base_tactile_image.permute(1,2,0).numpy().astype(np.uint8))
        if self.method == 'weighted' or self.method == '1':
            udiff, vdiff = self.extract_flow_field(prev_shear_image, image)
            if 'u' in self.channels or 'v' in self.channels or 'div' in self.channels or 'curl' in self.channels or 'sol_u' in self.channels or 'irr_u' in self.channels or 'sol_v' in self.channels or 'irr_v' in self.channels:
                self.u1 = self.u1 + udiff
                self.v1 = self.v1 + vdiff
                if self.method == '1':
                    self.u = self.u1
                    self.v = self.v1
        if self.method == 'weighted' or self.method == '2':
            if self.method == '2':
                if 'dudt' in self.channels or 'dvdt' in self.channels or 'du' in self.channels or 'dv' in self.channels:
                    udiff, vdiff = self.extract_flow_field(prev_shear_image, image)
            if 'u' in self.channels or 'v' in self.channels or 'div' in self.channels or 'curl' in self.channels or 'sol_u' in self.channels or 'irr_u' in self.channels or 'sol_v' in self.channels or 'irr_v' in self.channels:
                u2, v2 = self.extract_flow_field(no_shear_image, image)
                self.u2 = u2
                self.v2 = v2
                if self.method == '2':
                    self.u = self.u2
                    self.v = self.v2
        if self.method == 'weighted':
            if 'u' in self.channels or 'v' in self.channels or 'div' in self.channels or 'curl' in self.channels or 'sol_u' in self.channels or 'irr_u' in self.channels or 'sol_v' in self.channels or 'irr_v' in self.channels:
                weight = self.sigmoid_weighting_function(self.displacement, self.displacement_midpoint, self.displacement_threshold)
                self.u = weight*self.u1 + (1-weight)*self.u2
                self.v = weight*self.v1 + (1-weight)*self.v2
                self.displacement = np.max(np.array([np.std(self.u), np.std(self.v)]))
                if self.auto_reset:
                    displacement2 = np.max(np.array([np.std(self.u2), np.std(self.v2)]))
                    if displacement2 < self.reset_threshold:
                        if self.time() - self.last_reset_time > self.auto_reset_period and self.prev_displacement2 >= self.reset_threshold:
                            self.reset_shear()
                            self.last_reset_time = self.time
                    self.prev_displacement2 = displacement2
        if 'u' in self.channels:
            u_channel_index = self.channels.index('u')
            self.shear_field_tensor[u_channel_index] = torch.tensor(self.u)
        if 'v' in self.channels:
            v_channel_index = self.channels.index('v')
            self.shear_field_tensor[v_channel_index] = torch.tensor(self.v)
        if 'div' in self.channels:
            divergence = self.divergence(self.u, self.v)
            div_channel_index = self.channels.index('div')
            self.shear_field_tensor[div_channel_index] = torch.tensor(divergence)
        if 'curl' in self.channels:
            curl = self.curl(self.u, self.v)
            curl_channel_index = self.channels.index('curl')
            self.shear_field_tensor[curl_channel_index] = torch.tensor(curl)
        if 'sol_u' in self.channels or 'sol_v' in self.channels or 'irr_u' in self.channels or 'irr_v' in self.channels:
            vfield = np.stack((self.u,self.v),axis=2)
            self.nhhd.decompose(vfield)
            solenoidal = self.nhhd.r
            irrotational = self.nhhd.d
            u_sol = solenoidal[:,:,0]
            v_sol = solenoidal[:,:,1]
            u_irr = irrotational[:,:,0]
            v_irr = irrotational[:,:,1]
            if 'sol_u' in self.channels:
                sol_u_channel_index = self.channels.index('sol_u')
                self.shear_field_tensor[sol_u_channel_index] = torch.tensor(u_sol)
            if 'sol_v' in self.channels:
                sol_v_channel_index = self.channels.index('sol_v')
                self.shear_field_tensor[sol_v_channel_index] = torch.tensor(v_sol)
            if 'irr_u' in self.channels:
                irr_u_channel_index = self.channels.index('irr_u')
                self.shear_field_tensor[irr_u_channel_index] = torch.tensor(u_irr)
            if 'irr_v' in self.channels:
                irr_v_channel_index = self.channels.index('irr_v')
                self.shear_field_tensor[irr_v_channel_index] = torch.tensor(v_irr)
        if 'dudt' in self.channels:
            dudt_channel_index = self.channels.index('dudt')
            self.shear_field_tensor[dudt_channel_index] = torch.tensor(udiff/(self.time-self.prev_time))
        if 'dvdt' in self.channels:
            dvdt_channel_index = self.channels.index('dvdt')
            self.shear_field_tensor[dvdt_channel_index] = torch.tensor(vdiff/(self.time-self.prev_time))
        if 'du' in self.channels:
            du_channel_index = self.channels.index('du')
            self.shear_field_tensor[du_channel_index] = torch.tensor(udiff)
        if 'dv' in self.channels:
            dv_channel_index = self.channels.index('dv')
            self.shear_field_tensor[dv_channel_index] = torch.tensor(vdiff)
        self.prev_tactile_image = self.tactile_image
        self.prev_time = self.time
    
    def get_shear_field(self):
        return self.shear_field_tensor

    def update_tactile_image(self, tactile_image):
        self.tactile_image = tactile_image

    def update_base_tactile_image(self, base_tactile_image):
        self.base_tactile_image = base_tactile_image

    def update_time(self, time):
        self.time = time