import matplotlib.pyplot as plt

from typing import List

from matplotlib.animation import FuncAnimation

def plot_vector_field(ax, vf, ch_dim = 0, color = 'blue', scale = 6, title=''):
    u = get_channel(vf, 0, ch_dim)
    v = get_channel(vf, 1, ch_dim)
    fieldplot = ax.quiver(u, v, color = color, angles='xy', scale_units='xy', scale=scale)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.axis('off')
    ax.set_title(title)
    return fieldplot

def plot_scalar_field(ax, sf, max_magnitude = 1, cmap = 'seismic', title=''):
    if cmap not in ['PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']:
        raise ValueError('cmap must be a diverging colormap')
    fieldplot = ax.imshow(sf, cmap=cmap, vmin=-max_magnitude, vmax=max_magnitude)
    ax.axis('off')
    ax.set_title(title)
    return fieldplot

def get_channel(shear_field_tensor, channels, ch_dim=0):
    if ch_dim == 0:
        return shear_field_tensor[channels,:,:]
    elif ch_dim == 1:
        return shear_field_tensor[:,channels,:]
    elif ch_dim == 2:
        return shear_field_tensor[:,:,channels]

class ShearPlotter():
    def __init__(self, ch_dim = 0, max_scalar_magnitude = 6, scale = 6, channels:List[str]=['u','v'], titles=None, colors:List=['blue'], cmaps:List[str]=['seismic'], num_fingers = 1, base_figsize = (7,5)):
        self.ch_dim = ch_dim
        self.channels = channels
        self.num_fingers = num_fingers
        self.max_scalar_magnitude = max_scalar_magnitude
        self.scale = scale
        self.colors = colors
        self.cmaps = cmaps
        self.scalar_fields = [channel for channel in self.channels if channel in ['div', 'curl']]
        self.vector_fields = [channel for channel in self.channels if channel not in ['div', 'curl']]
        self.non_scalar_fields = []
        [self.non_scalar_fields.append(channel.replace('u','').replace('v','')) for channel in self.vector_fields if channel.replace('u','').replace('v','') not in self.non_scalar_fields]
        if len(self.non_scalar_fields) != len(self.vector_fields)//2:
            raise ValueError('Must include both u and v components for each vector field for plotting')
        if len(self.non_scalar_fields) != len(self.colors):
            if len(self.colors) == 1:
                self.colors = self.colors*len(self.non_scalar_fields)
            else:
                raise ValueError('Number of colors must match number of vector fields')
        self.num_scalar_fields = len(self.scalar_fields)
        if self.num_scalar_fields != len(self.cmaps):
            if len(self.cmaps) == 1:
                self.cmaps = self.cmaps*self.num_scalar_fields
            else:
                raise ValueError('Number of colormaps must match number of scalar fields')
        self.shear_field_tensors = []
        figsize = (base_figsize[0]*num_fingers, base_figsize[1]*len(self.channels))
        self.fig, self.ax = plt.subplots(len(self.scalar_fields)+len(self.non_scalar_fields), num_fingers, figsize = figsize)
        if titles is None:
            self.titles = self.non_scalar_fields + self.scalar_fields
        else:
            if len(titles) != len(self.scalar_fields)+len(self.non_scalar_fields):
                raise ValueError('Number of titles must match number of fields')
            self.titles = titles

    def plot_shear_info(self, shear_field_tensors):
        self.plots = []
        self.plot_types = []
        for finger in range(self.num_fingers):
            shear_field_tensor = shear_field_tensors[finger]
            for i, non_scalar_field in enumerate(self.non_scalar_fields):
                if non_scalar_field == '':
                    u_index = self.channels.index('u')
                    v_index = self.channels.index('v')
                elif non_scalar_field == 'sol_':
                    u_index = self.channels.index('sol_u')
                    v_index = self.channels.index('sol_v')
                elif non_scalar_field == 'irr_':
                    u_index = self.channels.index('irr_u')
                    v_index = self.channels.index('irr_v')
                elif non_scalar_field == 'd':
                    u_index = self.channels.index('du')
                    v_index = self.channels.index('dv')
                elif non_scalar_field == 'ddt':
                    u_index = self.channels.index('dudt')
                    v_index = self.channels.index('dvdt')
                vf = get_channel(shear_field_tensor, [u_index, v_index], self.ch_dim)
                plot = plot_vector_field(self.ax[i,finger], vf, ch_dim = self.ch_dim, color = self.colors[i], scale = self.scale, title = self.titles[i])
                self.plots.append(plot)
                self.plot_types.append('vector')
            for i, scalar_field in enumerate(self.scalar_fields):
                sf = get_channel(shear_field_tensor, self.channels.index(scalar_field), self.ch_dim)
                plot = plot_scalar_field(self.ax[i+len(self.non_scalar_fields),finger], sf, max_magnitude=self.max_scalar_magnitude, cmap = self.cmaps[i], title = self.titles[i+len(self.non_scalar_fields)])
                self.plots.append(plot)
                self.plot_types.append('scalar')
        return self.plots, self.plot_types
    
    def update_shear_info(self, frame, shear_field_tensors):
        self.shear_field_tensors = shear_field_tensors
        for i, finger in enumerate(range(self.num_fingers)):
            shear_field_tensor = self.shear_field_tensors[finger]
            for j, non_scalar_field in enumerate(self.non_scalar_fields):
                if non_scalar_field == '':
                    u_index = self.channels.index('u')
                    v_index = self.channels.index('v')
                elif non_scalar_field == 'sol_':
                    u_index = self.channels.index('sol_u')
                    v_index = self.channels.index('sol_v')
                elif non_scalar_field == 'irr_':
                    u_index = self.channels.index('irr_u')
                    v_index = self.channels.index('irr_v')
                elif non_scalar_field == 'd':
                    u_index = self.channels.index('du')
                    v_index = self.channels.index('dv')
                elif non_scalar_field == 'ddt':
                    u_index = self.channels.index('dudt')
                    v_index = self.channels.index('dvdt')
                vf = get_channel(shear_field_tensor, [u_index, v_index], self.ch_dim)
                plot_index = i*(len(self.non_scalar_fields)+len(self.scalar_fields)) + j
                self.plots[plot_index].set_UVC(vf[0,:,:], vf[1,:,:])
            for j, scalar_field in enumerate(self.scalar_fields):
                sf = get_channel(shear_field_tensor, self.channels.index(scalar_field), self.ch_dim)
                plot_index = i*(len(self.non_scalar_fields)+len(self.scalar_fields)) + j + len(self.non_scalar_fields)
                self.plots[plot_index].set_data(sf)
        return self.plots
    
    def animate_shear_info(self, shear_field_tensors, func):
        self.shear_field_tensors = shear_field_tensors
        self.plot_shear_info(shear_field_tensors)
        self.animation = FuncAnimation(self.fig, func, interval=50)
        plt.show()
        return self.animation
    
    def show(self):
        plt.show()