# Dataset

Example Structure:

### Dataset Name 
- [Link to Dataset](www.example.com)
- [(If exists)Link to Implementation](www.example.com)
- Other Details

### NYU Depth Map
- [link](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)
- There is a mat file which is called as labeled dataset. I think this dataset is useful and essential for our experiments. As far as I understand, mat file includes a struct with different fields listed in the link. The significant parts obviously are "depths" and "images". As far as I see, no special preprocessing is suggested however please warn me if there exists.

### KITTI Outdoor Dataset
- [link](http://www.cvlibs.net/datasets/kitti/raw_data.php)
- This dataset includes outdoor videos recorded with stereo cameras. Although, rectified version of the images exists, still disparity map extraction needed for the ground-truth depth data. I couldn't find depth images directly, yet. However, If we find that kind of source, I suggest to change this link with it because, although disparity map extraction is not a hard process, it does not make sense to spend time on it. What do you say? 

### MAKE3D Dataset
- [link](http://make3d.cs.cornell.edu/data.html)
- Official Readme:
1) Train400Depth.tgz
	Laser Range data with Ray Position
	Data Format: Position3DGrid (55x305x4)
		Position3DGrid(:,:,1) is Vertical axis in meters (Y)
		Position3DGrid(:,:,2) is Horizontal axis in meters (X)
		Position3DGrid(:,:,3) is Projective Depths in meters (Z)
		Position3DGrid(:,:,4) is Depths in meters (d)

2) Train400Img.tar.gz
	Images all in resolution 2272x1704


