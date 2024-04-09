# Fast Gaussian Splatting

- **5-10x faster rendering than the original software CUDA rasterizer ([diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization)).**
- **2-3x faster if using offline rendering. (Bottleneck: copying rendered images around, thinking about improvements.)**

https://github.com/dendenxu/fast-gaussian-splatting/assets/43734697/f50afd6f-bbd5-4e18-aca6-a7356a5d3f75

No backward pass is supported yet. 
Will think of ways to add a backward. 
Depth-peeling ([4K4D](https://zju3dv.github.io/4k4d)) is too slow.
Discussion welcomed.

## Installation

No CUDA compilation is required.

```shell
pip install fast_gauss
```

## Usage

Replace the original import of `diff_gaussian_rasterization` with `fast_gauss`.

For example, replace this:

```python
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
```

with this:

```python
from fast_gauss import GaussianRasterizationSettings, GaussianRasterizer
```

And you're good to go.

## Tips

**Note: for the ultimate 5-10x performance increase, you'll need to let `fast_gauss`'s shader directly write to your desired framebuffer.**

Currently, we are trying to automatically detect whether you're managing your own OpenGL context (i.e. opening up a GUI) by checking for the module `OpenGL` during the import of `fast_gauss`.

If detected, all rendering command will return `None`s and we will directly write to the bound framebuffer at the time of the draw call.

Thus if you're running in a GUI (OpenGL-based) environment, the output of our rasterizer will be `None`s and does not require further processing.

- [ ] TODO: Improve offline rendering performance.
- [ ] TODO: Add a warning to the user if they're performing further processing on the returned values.

Note: it's recommended to pass in a CPU tensor in the GaussianRasterizationSettings to avoid explicit synchronizations for even better performance.

- [ ] TODO: Add a warning to the user if GPU tensors are detected.

Note: the second output of the `GaussianRasterizer` is not radii anymore (since we're not gonna use it for the backward pass), but the alpha values of the rendered image instead.
And the alpha channel content seems to be bugged currently, will debug.

- [ ] TODO: Debug alpha channel

## TODOs

- [ ] TODO: Thinks of ways for a backward pass. Welcome to discuss!
- [ ] TODO: Compute covariance from scaling and rotation in the shader, currently it's on the CUDA side.
- [ ] TODO: Compute SH in the shader, currently it's on the CUDA side.

## Environment

This project requires you to have an NVIDIA GPU with the ability to interop between CUDA and OpenGL.
Thus, WSL is [not supported](https://docs.nvidia.com/cuda/wsl-user-guide/index.html#features-not-yet-supported) and OSX (MacOS) is not supported.

For offline rendering (the drop-in replacement of the original CUDA rasterizer), we also need a valid EGL environment.
It can sometimes be hard to set up for virtualized machines. [Potential fix](https://github.com/zju3dv/4K4D/issues/27#issuecomment-2026747401).

- [ ] TODO: Test on more platforms.

## Credits

Inspired by those insanely fast WebGL-based 3DGS viewers:

- [GaussianSplats3D](https://github.com/mkkellogg/GaussianSplats3D) for inspiring our vertex-geometry-fragment shader pipeline.
- [gsplat.tech](https://gsplat.tech/).
- [splat](https://github.com/antimatter15/splat).

Using the algorithm and improvements from:

- [diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization) for the main Gaussian Splatting algorithm.
- [diff_gauss](https://github.com/dendenxu/diff-gaussian-rasterization) for the fixed culling.

CUDA-GL interop & EGL environment inspired by:

- [4K4D](https://zju3dv.github.io/4k4d) where they(I) used the interop for depth-peeling.
- [EasyVolcap](https://github.com/zju3dv/EasyVolcap) for the collection of utilities, including EGL setup.
- [nvdiffrast](https://nvlabs.github.io/nvdiffrast) for their EGL context setup and CUDA-GL interop setup.

## Citation

```bibtex
@misc{fast_gauss,  
    title = {Fast Gaussian Splatting},
    howpublished = {GitHub},  
    year = {2024},
    url = {https://github.com/dendenxu/fast-gaussian-rasterization}
}
```
