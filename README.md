# The GelSlim 4.0 Shear Field Package
Optical flow-based approximations of shear fields from RGB vision-based tactile sensor GelSlim 4.0 <br />
<p align="center">
  <img src="https://github.com/MMintLab/gelslim_shear/blob/master/media/animations/decomposition_marker.gif?raw=true" alt="GIF of Helmholtz Decomposition and Divergence and Curl"/>
  <br />
  <img src="https://github.com/MMintLab/gelslim_shear/blob/master/media/animations/time_derivative_hex.gif?raw=true" alt="GIF of Time Derivative"/>
  <br />
  <img src="https://github.com/MMintLab/gelslim_shear/blob/master/media/animations/shear_field_small_screw_head.gif?raw=true" alt="GIF of Shear Field Approximations"/>
</p>

For all functionality associated with the GelSlim 4.0, [visit the project website!](https://www.mmintlab.com/research/gelslim-4-0/)

This repository is also used in [Built Different: Tactile Perception to Overcome Cross-Embodiment Capability Differences in Collaborative Manipulation](https://www.mmintlab.com/research/tactile-collaborative/).

Tested On: <br />
<a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-v2.0+-red.svg?logo=PyTorch&style=for-the-badge" /></a>
<a href="#"><img src="https://img.shields.io/badge/python-v3.8+-blue.svg?logo=python&style=for-the-badge" /></a>

## Installation

1. [Install PyTorch](https://pytorch.org/get-started/locally/)

2. Install Dependencies
```bash
pip install -r requirements.txt
```

3. Clone `gelslim_shear` with git to create the root directory

4. Install `gelslim_shear` (run in `gelslim_shear` root directory)
```bash
pip install -e .
```

## Shear Field Generation

This package allows the designing of a multi-channel tensor with a variety of representations of the shear field. We provide a `ShearGenerator` to generate the shear field from an RGB tactile image in the form of a ```3 x H x W``` tensor, with values between 0 and 255 (float or uint8). To do this:

1. Import the `ShearGenerator`:
```python
from gelslim_shear.shear_utils.shear_from_gelslim import ShearGenerator
```

2. Define the generator with your parameters. We only recommend altering `method`, `Farneback_params` and `channels` though you can alter the size of the shear field `output_size` from `(13,18)` to something else. You can define one shear generator for each finger.

```python
shgen = ShearGenerator(method=<choose one from ['1','2', weighted]>, channels=<any combination of ['u','v','div','curl','sol_u','sol_v','irr_u','irr_v','dudt','dvdt','du','dv']>, Farneback_params = (0.5, 3, 15, 3, 5, 1.2, 0))
```

For example:
```python
shgen = ShearGenerator(method='2', channels=['u','v','div','du','dv'], Farneback_params = (0.5, 3, 45, 3, 5, 1.2, 0))
```

The above example will create a shear generator which outputs a `5 x 13 x 18` tensor with each channel representing those in the specified list.

A description of the possible channels:
- `u`: Horizontal component of the shear field
- `v`: Vertical component of the shear field
- `div`: Divergence of the shear field
- `curl`: Curl of the shear field
- `sol_u`: Horizontal component of the solenoidal shear field from the Helmholtz-Hodge Decomposition
- `sol_v`: Vertical component of the solenoidal shear field from the Helmholtz-Hodge Decomposition
- `irr_u`: Horizontal component of the irrotational shear field from the Helmholtz-Hodge Decomposition
- `irr_v`: Vertical component of the irrotational shear field from the Helmholtz-Hodge Decomposition
- `dudt`: Horizontal component of the time derivative of the shear field
- `dvdt`: Vertical component of the time derivative of the shear field
- `du`: Horizontal component of the change in the shear field
- `dv`: Vertical component of the change the shear field

`(0.5, 3, 45, 3, 5, 1.2, 0)` is a good setting for method `2` without many defects and avoids potential coding complexity from the `weighted` method. However, some resolution is lost with these `Farneback_params` as opposed to `(0.5, 3, 15, 3, 5, 1.2, 0)`.

3. Once `shgen` is defined, you can use it in a continuous loop with system time `t` defined (i.e from `rospy.get_time()`, etc.), and the deformed and undeformed tactile images defined as ```3 x H x W``` tensors (we use `H=320` and `W=427`) named `tactile_image` and `base_tactile_image`:

```python
shgen.update_base_tactile_image(base_tactile_image)
while True:
  t = #function to get time
  tactile_image = #function to get tactile image in 3 x H x W tensor
  shgen.update_time(t)
  shgen.update_tactile_image(tactile_image)
  shgen.update_shear()
  shear_field_tensor = shgen.get_shear_field()
  #do something with shear_field_tensor
```

`shgen.get_shear_field()` returns the `len(shgen.channels) x 13 x 18` tensor that represents the shear field.

4. If at any time you'd like to manually reset the shear field with a new `base_tactile_image` (i.e. a recently collected one), simply run:
```python
shgen.reset_shear(base_tactile_image)
```

## Shear Field Visualization
To visualize the various representations within this library, we have included a simple `ShearPlotter` which wraps a series of `matplotlib` functions for easy plotting. Add the following to your code for visualization:
```python
from gelslim_shear.plot_utils.shear_plotter import ShearPlotter
shplot = ShearPlotter(channels=shgen.channels)
```

To plot a a single `shear_field_tensor` with each included representation in subplots, run:

```python
shplot.plot_shear_info([shear_field_tensor])
shplot.show()
```

We have also enabled animations of shear fields. For example if we want to do a live animation of the shear field:

```python
shgen.update_base_tactile_image(base_tactile_image)

def update(frame):
  t = #function to get time
  tactile_image = #function to get tactile image in 3 x H x W tensor
  shgen.update_time(t)
  shgen.update_tactile_image(tactile_image)
  shgen.update_shear()
  shear_field_tensor = shgen.get_shear_field()
  #do something with shear_field_tensor
  shplot.update_shear_info(frame, [shear_field_tensor])
  return shplot.plots

t = #function to get time
tactile_image = #function to get tactile image in 3 x H x W tensor
shgen.update_time(t)
shgen.update_tactile_image(tactile_image)
shgen.update_shear()
shear_field_tensor = shgen.get_shear_field()
shplot = ShearPlotter(channels=shgen.channels)
shplot.animate_shear_info([shear_field_tensosr], update)
```

The reason for `shear_field_tensor` being placed in a list is we allow for the plotting of multiple fingers simultaneously, by adding `shplot = ShearPlotter(num_fingers=2)` for example to the intialization of the plotter. This coupled with the above code and `channels=['u','v','div','du','dv]` will produce a live animation of both fingers as follows:

<p align="center">
  <img src="https://github.com/MMintLab/gelslim_shear/blob/master/media/animations/animation.gif?raw=true" alt="GIF of Live Shear Animation"/>
</p>

`ShearPlotter` also has more initialization arguments:
- `colors`: List of colors to plot each vector field representation, for example: `colors=['blue','green','magenta']`
- `cmaps`: List of diverging colormaps to plot each scalar field (`div` or `curl`) representation, for example: `cmaps=['seismic','PuOr']`
  - <a href="https://matplotlib.org/stable/users/explain/colors/colormaps.html">List of Colormaps</a>
- `titles`: List of titles of each representation subplot, for example: `titles = ['Shear Field', 'Time Differential', 'Divergence']`
- `base_figsize`: Tuple of horizontal, vertical size of each subplot
- `scale`: Scale passed to `matplotlib.pyplot.quiver` for vector field plots, controls the size of the arrows
- `max_scalar_magnitude`: Maximum magnitude for visualizing the scalar fields, this can also be adaptively adjusted based on the data by passing a changing value to `shplot.max_scalar_magnitude`.
- `ch_dim`: Dimension along which the channels are stacked, it's best to keep this at the default `chdim=0`

For more simple plotting, if you wish to plot a single vector or scalar field, you can import these functions:

```python
from gelslim_shear.plot_utils.shear_plotter import plot_vector_field, plot_scalar_field, get_channel
import matplotlib.pyplot as plt
```

Example usage of these functions:
```python
fig, ax = plt.subplots(1,2)
shear_field_tensor = shgen.get_shear_field()
vf = get_channel(shear_field_tensor, [shgen.channels.index('u'), shgen.channels.index('v')])
sf = get_channel(shear_field_tensor, shgen.channels.index('div'))
plot_vector_field(ax[0], vf, title='Shear Field')
plot_scalar_field(ax[1], sf, title='Divergence', cmap='PuOr')
plt.show()
```

The Result:
<p align="center">
  <img src="https://github.com/MMintLab/gelslim_shear/blob/master/media/images/plot_test.png?raw=true" alt="Plot Test"/>
</p>