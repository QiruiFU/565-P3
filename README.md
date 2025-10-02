CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Qirui (Chiray) Fu
  * [personal website](https://qiruifu.github.io/)
* Tested on my own laptop: Windows 11, i5-13500HX @ 2.5GHz 16GB, GTX 4060 8GB

## README

<div align="center">
<img src="/img/final-result.png", style="width: 80%">
</div>

### Introduction

In this project, I implemented a path tracer, which can render scenes described in `json` file. This renderer support 4 different materials : light source, diffuse, perfect specular and refraction materials like glass. There are 3 primitive geometries we support : Sphere, cube and triangle. Based on project [tinyobj](https://github.com/tinyobjloader/tinyobjloader), this renderer can convert `obj` file to triangles automatically, making it possible to render complicated scenes.

In `utilities.h` file, you can turn on or off different features to test their effects.

### Visual Features
#### Materials
We support 4 types materials including light source.

* Diffuse:
<img src="/img/cornell-mat-diffuse.png" style="width: 50%">

* Perfect Specular:
<img src="/img/cornell-mat-specular.png" style="width: 50%">

* Refraction Material:
<img src="/img/cornell-refraction-cube.png" style="width: 50%">
<img src="/img/cornell-refraction-sphere.png" style="width: 50%">

#### Load `obj` file into scene & BVH tree
In the file describing scene, you can indicate the path of you obj file, then renderer will parse it automatically. This feature is based on [tinyobj](https://github.com/tinyobjloader/tinyobjloader). Of course rendering thousands of triangles is really slow, we implemented BVH tree to accelerate this process. You can find the comparison of performance between with and without BVH in the next section.

<img src="/img/choppa.png" style="width: 50%">
count of triangle : 130k , source : https://skfb.ly/oS8Ux

<img src="/img/buddha.png" style="width: 50%">
count of triangle : 37k , source : https://skfb.ly/QGxw

#### Anti-Alias
In each iteration, we can generate different rays for the same pixel. In this way we can make edge of geometries more smooth and have less artifacts.
<div align="center">
<img src="/img/AntiAlias-wout.png", style="width: 40%">
<img src="/img/AntiAlias.png", style="width: 40%">
</div>


#### Depth of Field
By simulating behavior of a real camera, we can generate effect of Depth of Filed(DOF), which makes our pictures look more real. The core of this method is after generating a ray in each iteration, we change the origin and direct of it according to camera's aperture radius and focus distanse.

|           | aperture : 0.15  | aperture : 0.30   |
|------------|------------|------------|
| focus dis : 7.0    | <img src="/img/DOF-015-7.png"> | <img src="/img/DOF-03-7.png"> |
| focus dis : 9.5    | <img src="/img/DOF-015-95.png"> | <img src="/img/DOF-03-95.png"> |

### Performance Analysis
#### Russian Roulette
Each time when a ray bounces, we can use a probability `p` to determine whether to kill this ray. If we kill it, just set `remainBouncing` to `0`. If we don't kill it, to make sure the expectation don't change, we need to update the light with `color = color / (1 - p)`, where `1-p` is the probability it survives. This method may lead to higher variance, but can keep energy conservative and make rendering faster.

|    trace depth  |  5     |  10        |
|-----------------|--------|------------|
| Roulette on     | $35.1$ FPS | $22.3$ FPS |
| Roulette off    | $30.6$ FPS | $16.7$ FPS |

#### BVH Spatial Structure
By building a BVH tree, we can divide geometries into a lot of nodes. Every time when we need to check whether a ray could bound this geometry, we can check it with a big bounding box at first. In this way, we can eliminate lots of bounding candidate in one step of computation. To test its performance, we test it on two scenes with different amount of triangles:

|            |  scene1  |  scene2  |
|------------|------------|------------|
| BVH on     | $4.2$ FPS | $5.2$ FPS |
| BVH off    | $2.4$ FPS | $0.1$ FPS |

Scene1 : 251 triangles 

Scene2 : 6652 triangles

#### Sort Material
In order to handle intersections with the same material in each CUDA block, we can sort all intersections before shading. This may lead to contiguous memory visiting and increase the redering speed. However, I found that in the most scenes we have much less types of material compared with quantity of geometries, this method could make rendering slower.

### Gallery & Bloopers
<div align="center">
<img src="/img/cornell-Sword.png" style="width: 30%">
<img src="/img/cornell-Mario.png" style="width: 30%">
<img src="/img/cornell-refraction-statue.png" style="width: 30%">
</div>

##### bloopers:

<div align="center">
<img src="/img/bug-inter.png" style="width: 30%">
<img src="/img/bug-reflect.png" style="width: 30%">
<img src="/img/bug-refraction1.png" style="width: 30%">
</div>
<div align="center">
<img src="/img/bug-refraction2.png" style="width: 30%">
<img src="/img/bug-refraction3.png" style="width: 30%">
</div>