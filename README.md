# STAC: Leveraging Spatio-Temporal Data Associations

**V. Vakhniuk, A. Sarkar, R. Gupta**

In existing live-camera surveillance or traffic monitoring systems, multiple geographically distributed cameras stream live videos to a remote server that runs a high end DNN inference learning model to respond to user queries in real-time. However, as a greater number of cameras get deployed, they pose a serious challenge in terms of compute-intensive tasks over massive data (video) sources in real-time. But as the number of cameras increase, the resource demand for both deep learning (NN) based computation and network bandwidth for communication increases exponentially. To address this, it is important that the live analytics between multiple cameras include cross-camera redundant data elimination for streaming and leverage spatial-temporal associations among cross-camera streams for accurate inference results.
 
We propose an efficient cross-cameras surveillance system called, STAC, that leverages spatio-temporal associations between multiple cameras to provide real-time analytics and inference under constrained network environments. STAC is built using the proposed omni-scale feature learning people reidentification (reid) algorithm that allows accurate detection, tracking and re-identification of people across cameras using the spatio-temporal characteristics of video frames. We integrate STAC with frame filtering and state-of-the-art compression to remove redundant information from cross-camera frames, optimizing video transmission cost while maintaining high accuracy for real-time query inference. We found that our implementation was much faster inference-wise than an established baseline producing superior results when it came to association (mapping detected people across camera streams).

<div align="center">
    <img src="https://github.com/VolodymyrVakhniuk/STAC/blob/main/STAC.png" width="700" height="700" alt="Description" style="border-radius: 15;">
</div>
