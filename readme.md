# Fast Gaussian Rasterization

- **Can be 5-10x faster than the original software CUDA rasterizer ([diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization)).**
- **Can be 2-3x faster if using offline rendering. (Bottleneck: copying rendered images around, thinking about improvements.)**
- **Speedup most visible with high pixel-to-point ratio (large Gaussians, small point count, high-res rendering).**

https://github.com/dendenxu/fast-gaussian-splatting/assets/43734697/f50afd6f-bbd5-4e18-aca6-a7356a5d3f75

No backward pass is supported yet. 
Will think of ways to add a backward. 
Depth-peeling ([4K4D](https://zju3dv.github.io/4k4d)) is too slow.
Discussion welcomed.

## Installation

Install the latest release from PyPI:

```shell
pip install fast_gauss
```

Or the latest commit from GitHub:

```shell
pip install git+https://github.com/dendenxu/fast-gaussian-rasterization
```

No CUDA compilation is required to build `fast_gauss` since we're only shader-based for now.

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


**Note: the speedup is the most visible when the pixel-to-point ratio is high.**

That is, when there are large Gaussians and very high-resolution rendering, the speedup is more visible.

The CUDA-based software implementation is more resolution sensitive and for some extremely dense point clouds (> 1 million points), the CUDA implementation might be faster.

This is because the typical rasterization-based pipeline on modern graphics hardware is [not well-optimized for small triangles](https://www.youtube.com/watch?v=hf27qsQPRLQ&list=WL).


**Note: for best performance, cache the persistent results (for example, the 6 elements of the covariance matrix).**

This is more of a general tip and not directly related to `fast_gauss`.

However, the impact is more observable here since we haven't implemented a fast 3D covariance computation (from scales and rotations) in the shader yet.
Only PyTorch implementation is available for now.

When the point count increases, even the smallest `precomputation` can help.
An example is the concatenation of the base 0-degree SH parameter and the rest, that small maneuver might cost us 10ms on a 3060 with 5 million points.
Thus, store the concatenated tensors instead and avoid concatenating them in every frame.

- [ ] TODO: Implement SH eval in the vertex shader.
- [ ] TODO: Warn users if they're not properly precomputing the covariance matrix.
- [ ] TODO: Implement a more optimized `OptimizedGaussians` for precomputing things and apply a cache. Similar to that of the vertex shader (see [Invokation frequency](https://www.khronos.org/opengl/wiki/Vertex_Shader)).


**Note: it's recommended to pass in a CPU tensor in the `GaussianRasterizationSettings` to avoid explicit synchronizations for even better performance.**

- [ ] TODO: Add a warning to the user if GPU tensors are detected.


**Note: the second output of the `GaussianRasterizer` is not radii anymore (since we're not gonna use it for the backward pass), but the alpha values of the rendered image instead.**

And the alpha channel content seems to be bugged currently, will debug.

- [ ] TODO: Debug alpha channel values

## TODOs

- [ ] TODO: Apply more of the optimization techniques used by similar shaders, including packing the data into a texture and bit reduction during computation.
- [ ] TODO: Thinks of ways for a backward pass. Welcome to discuss!
- [ ] TODO: Compute covariance from scaling and rotation in the shader, currently it's on the CUDA (PyTorch) side.
- [ ] TODO: Compute SH in the shader, currently it's on the CUDA (PyTorch) side.
- [ ] TODO: Try to align the rendering results at the pixel level, small deviation exists currently.
- [ ] TODO: Use indexed draw calls to minimize data passing and shuffling.
- [ ] TODO: Do incremental sorting based on viewport change, currently it's a full resort on with CUDA (PyTorch).

## Implementation

**Goal:**

- Let the professionals do the work.
  - Let GPU do the large-scale sorting.
  - Let the graphics pipeline do the rasterization for us, not the other way around.
  - Let OpenGL directly write to your framebuffer.
- Minimize repeated work.
  - Compute the 3D to 2D covariance projection only once for each Gaussian, instead of 4 times for the quad, enabled by the geometry shader.
- Minimize stalls (minimize explicit synchronizations between GPU and CPU).
  - Enabled by using `non_blocking=True` data passing and moving sync points to as early as possible.
  - Boosted by the fact that we're sorting on the GPU, thus no need to perform synchronized host-to-device copies.

- [ ] TODO: Expand implementation details.

## Environment

This project requires you to have an NVIDIA GPU with the ability to interop between CUDA and OpenGL.
Thus, WSL is [not supported](https://docs.nvidia.com/cuda/wsl-user-guide/index.html#features-not-yet-supported) and OSX (MacOS) is not supported.
Tested on Linux and Windows.

For offline rendering (the drop-in replacement of the original CUDA rasterizer), we also need a valid EGL environment.
It can sometimes be hard to set up for virtualized machines. [Potential fix](https://github.com/zju3dv/4K4D/issues/27#issuecomment-2026747401).

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
    title = {Fast Gaussian Rasterization},
    howpublished = {GitHub},  
    year = {2024},
    url = {https://github.com/dendenxu/fast-gaussian-rasterization}
}
```
