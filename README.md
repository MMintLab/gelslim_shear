# The GelSlim 4.0 Shear Field Package
Optical flow-based approximations of shear fields from RGB vision-based tactile sensor GelSlim 4.0 <br />
![GIF of Helmholtz Decomposition and Divergence and Curl](https://github.com/MMintLab/gelslim_shear/blob/master/media/animations/decomposition_marker.gif?raw=true)<br />
![GIF of Time Derivative](https://github.com/MMintLab/gelslim_shear/blob/master/media/animations/time_derivative_hex.gif?raw=true)
![GIF of Shear Field Approximations](https://github.com/MMintLab/gelslim_shear/blob/master/media/animations/shear_field_small_screw_head.gif?raw=true)

For all functionality associated with the GelSlim 4.0, [visit the project website!](https://www.mmintlab.com/research/gelslim-4-0/)

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
```
from gelslim_shear.shear_utils.shear_from_gelslim import ShearGenerator
```

2. Define the generator with your parameters. We only recommend altering `method`, `Farneback_params` and `channels` though you can alter the size of the shear field `output_size` from `(13,18)` to something else. You can define one shear generator for each finger.

```
shgen = ShearGenerator(method=<choose one from ['1','2', weighted]>, channels=<any combination of ['u','v','div','curl','sol_u','sol_v','irr_u','irr_v','dudt','dvdt','du','dv']>, Farneback_params = (0.5, 3, 15, 3, 5, 1.2, 0))
```

For example:
```
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

```
shgen.update_base_tactile_image(base_tactile_image)
[Loop]:
    shgen.update_time(t)
    shgen.update_tactile_image(tactile_image)
    shgen.update_shear()
    shear_field_tensor = shgen.get_shear_field()
```

`shgen.get_shear_field()` returns the `len(shgen.channels) x 13 x 18` tensor that represents the shear field.

4. If at any time you'd like to manually reset the shear field with a new `base_tactile_image` (i.e. a recently collected one), simply run:
```
shgen.reset_shear(base_tactile_image)
```
