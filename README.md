This repository provides the public code of the following paper:
> Ienaga, N., Cravotta, A., Terayama, K. et al. Semi-automation of gesture annotation by machine learning and human collaboration. Lang Resources & Evaluation 56, 673–700 (2022). https://doi.org/10.1007/s10579-022-09586-4

This code can be run on Colab so that the environment setup is not needed. If the link below doesn't work, just download and upload it to your Google Drive.  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/naotoienaga/annotation-tool/blob/master/semi-automatic_gesture_annotation_tool.ipynb)


### Usage
- Use [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) to detect keypoints of the video and annotate part of the input video. Annotate at least one rest and gesture (Any tire name is ok other than PREDICTED and QUERY).
- Set parameters in ipynb file (path to zip file of OpenPose jsons, path to annotation file etc.).
- Run cells.
- An elan file with predicted annotation will be generated. If you are not satisfied with the prediction results, add more annotations and run again (delete PREDICTED tire and QUERY tire). The annotation in the “QUERY” tire is the most uncertain data, efficient improvement of accuracy at the next cycle can be expected if the query is annotated
