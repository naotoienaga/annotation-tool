This repository provides the public code of a paper titled, "Semi-automation of Gesture Annotation by Machine Learning and Human Collaboration"  
(authors: Naoto Ienaga, Alice Cravotta, Kei Terayama, Bryan W. Scotney, Hideo Saito, M. Grazia Busà)

This code can be run on Colab so that the environment setup is not needed. If the link below doesn't work, just download and upload it to your Google Drive.  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/naotoienaga/annotation-tool/blob/master/semi-automatic_gesture_annotation_tool.ipynb)


### Usage
- Use [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) to detect keypoints of the video and annotate part of the input video. Annotate at least one rest and gesture (Any tire name is ok other than PREDICTED and QUERY).
- Set parameters in ipynb file (path to zip file of OpenPose jsons, path to annotation file etc.).
- Run cells.
- An elan file with predicted annotation will be generated. If you are not satisfied with the prediction results, add more annotations and run again (delete PREDICTED tire and QUERY tire). The annotation in the “QUERY” tire is the most uncertain data, efficient improvement of accuracy at the next cycle can be expected if the query is annotated
