
# Papers

Example Structure:

### Paper Name 
- Relevance Level (1 to 3 - 1 is the highest relevance)
- [Link to Paper](www.example.com)
- Summary 
- [(If exists)Link to Implementation](www.example.com)
- Other Details

### Depth Estimation from Single Image Using CNN-Residual Network
- Relevance Level : 1
- [link](http://cs231n.stanford.edu/reports/2017/pdfs/203.pdf)
- Summary: Please read the second paper :)
- Every member should read this one

### Deeper Depth Prediction with Fully Convolutional Residual Networks
- Relevance Level : 1
- [link](https://arxiv.org/pdf/1606.00373.pdf)
- [implementation](https://github.com/iro-cp/FCRN-DepthPrediction)
- This architecture basically employs pretrained famous residual architecture, RESNET50, for automated feature extraction. With removing Fully Connected Layer at the end of RESNET50, remaining architecture, that consists of convolution, pooling, batch normalization, RELU, and Short-Cuts(key feature of residual networks), is dedicated to feature extraction.(For details of RESNET50, see this |[paper](https://arxiv.org/pdf/1512.03385.pdf)) After RESNET50, rest of the architecture focuses on upscaling. The necessity of this layer arises from the required dimension of output. As explained in the paper, the main purpose is to create a depth map whose dimensions are higher than the outputs of intermediate layers(conv,pooling etc.). For this purpose, four different types of upscaling block are proposed. Namely, these are up-convolution, up-projection, fast up-convolution and fast up projection. For internal and implementation details of these blocks, see the figure 2 and related explanations.
- Reader comment about this paper: This paper actually explains all the details of the "Depth Estimation from Single Image Using CNN-Residual Network", therefore, I think this paper and its implementation can be our basis point for further implementaions. 
- Also the results of this work overperforms nearly all the methods. 
- However, I think there is a lack of evaluation of error functions or I have skipped some parts :)
 

### Depth Map Prediction from a Single Image using a Multi-Scale Deep Network
- Relevance Level : 1
-[link](https://arxiv.org/pdf/1406.2283.pdf)
- This paper presents one of the well known work on CNN for Depth Estimation. There are two architectures that tries to enhance different sides of the depth map. The first architecture includes convolutional and pooling layers, and tries to extract the global depth information in terms of spatial domain. You can think this as depth map of general scene. Whereas, the latter one consists of only convolutional layers and uses the output of the previous architecture. This architecture alone focuses on local information such as small objects respect to overall image dimensions. With the input of previous architecture, it generates a refined depth map. 
- Reader Comment: The results are not as good as the previous approach(Residual Network) however, there are more error functions defined which might be beneficial for our experiments and training loss. Overall, this network has a more simple structure and the paper evaluates different datasets such as NYU, Make3D and KITTI. 

### Single image depth estimation by dilated deep residual convolutional neural network and soft-weight-sum inference 
- Relevance Level : 1 
- [link](https://arxiv.org/abs/1705.00534)

### Deep Convolutional Neural Fields for Depth Estimation from a Single Image
- Relevance Level : 1 
- [link](https://arxiv.org/abs/1411.6387)

### Unsupervised Monocular Depth Estimation with Left-Right Consistency
- Relevance Level : 2
- [link](https://arxiv.org/abs/1609.03677)

### Unsupervised Learning of Depth and Ego-Motion from Video
- Relevance Level : 2
- [link](https://people.eecs.berkeley.edu/~tinghuiz/projects/SfMLearner/cvpr17_sfm_final.pdf)

### 3-D Depth Reconstruction from a Single Still Image
- Relevance Level : 3 
- [link](http://www.cs.cornell.edu/~asaxena/learningdepth/saxena_ijcv07_learningdepth.pdf)
- A famous paper from famous person 

### CNN-SLAM: Real-time dense monocular SLAM with learned depth prediction
- Relevance Level : 2
- [link](https://arxiv.org/abs/1704.03489)
- In perspective of SLAM

### Fully Convolutional Network for Depth Estimation and Semantic Segmentation
- Relevance Level : 1
- [link](http://cs231n.stanford.edu/reports/2017/pdfs/209.pdf)
- [implementation](https://github.com/iapatil/depth-semantic-fully-conv)

### Depth Estimation from Monocular Image with Sparse Known Labels
- Relevance Level : 2
- [link](http://sunw.csail.mit.edu/abstract/Depth_Estimation_from.pdf)

### Unsupervised CNN for Single View Depth Estimation: Geometry to the Rescue
- Relevance Level : 1
- [link](https://arxiv.org/pdf/1603.04992.pdf)
- [implementation](https://github.com/Ravi-Garg/Unsupervised_Depth_Estimation) 

### Depth and surface normal estimation from monocular images using regression on deep features and hierarchical CRFs
- Relevance Level : 2
- [link](http://users.cecs.anu.edu.au/~yuchao/files/Li_Depth_and_Surface_2015_CVPR_paper.pdf)

