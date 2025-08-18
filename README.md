Personal repo of [SeaClear](https://costinchitic.co/seaclear) project. Also to keep track of my contribution.
Mainly applying computer vision and computations. 
ROS1 + Python. 
Soon C++ for all of my contributions


My contribution so far in coding:
* src/
* include/
* scripts/aruco.py
* scripts/grid_detection.py
* scripts/camera_sync_recorder.py
* scripts/mp4_to_rosbag.py
* scripts/extract_frames.py
* scripts/dualtrajectoryplotter.py
* scripts/rov_prediction_cnn.py
* scripts/color_gopro
* scripts/color_usbcamera
* *.sh


---
# Logic behind `dualtrajectoryplotter.py` pipeline of transformations

## Frames

  - **Camera** $\\rightarrow$ C
  - **Aruco** (World) $\\rightarrow$ W

-----

## 1\) What `aruco.py` (OpenCV) gives

`cv2.solvePnP` returns $(R, t)$ such that:

$$X_C = R \cdot X_W + t$$

This maps a point from the world (aruco) frame into the camera frame.

-----

## 2\) Inverting to get camera $\\rightarrow$ world

After I fetch $R$ and $t$, I pass them to `publish_camera_to_aruco_transform()` which computes exactly the inverse: *"How do I go from camera coordinates to world coordinates?"*

Starting from:

$X\_C = R \\cdot X\_W + t$

Multiply on the left with $R^\\top$:

$$R^\top \cdot X_C = X_W + R^\top \cdot t$$

Rearrange:

$$X_W = R^\top \cdot (X_C - t)$$

or equivalently:

$$X_W = R^\top \cdot X_C + t_{\text{inv}}, \quad t_{\text{inv}} = -R^\top \cdot t$$

So the camera $\\rightarrow$ world transform is defined by:

  - **Rotation**: $R^\\top$
  - **Translation**: $-R^\\top t$

Once I pass this to the TF library, it will always know how to compute the transformations from each camera to world coordinates (in meters).

-----

## 3\) What we publish to TF

In ROS TF, I publish:

```python
transform.header.frame_id = "aruco_marker"  # parent = world
transform.child_frame_id  = "camera"        # child  = camera
```

and I set the rotation to $(R^\\top)$ and the translation to $(-R^\\top t)$.
This tells TF: “the pose of the camera is defined relative to the aruco\_marker (world)”.

-----

## 4\) Using the transform later

When I back-project a pixel into 3D (`pixel_to_3d_point`), the result is expressed in the **camera frame**:

$$p_C = \begin{bmatrix} x_c \\ y_c \\ z_c \end{bmatrix}$$

To interpret this in the **world (aruco) frame**, I use TF. Since TF already knows the static transform

$$
T_{C \to W} =
\begin{bmatrix}
R^\top & -R^\top t \\
0 & 1
\end{bmatrix},
$$it can convert the point as:

$$p_W = R^\top \cdot p_C + (-R^\top t)$$

Thus, TF takes care of expressing any point measured in the camera frame into the common world (aruco\_marker) frame.
