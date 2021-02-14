# annotation-tool
Semi automatic annotation tool

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/naotoienaga/annotation-tool/blob/master/notebooks/semi-automatic_gesture_annotation_tool.ipynb)

## Requirements
The code was tested with python 3.6.8, opencv 4.1.1, lightgbm 2.2.4, and anaconda.

## Usage
- Use [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose) to detect key points of the video and annotate part of the input video. Annotate at least one rest and gesture (Any tire name is ok other than PREDICTED and QUERY).
- Edit "config.ini" (Please specify path to json file, path to annotation file path etc.)
- Run `python anno_tool.py`
- An elan file with predicted annotation will be generated. If you are not satisfied with the prediction results, add more annotations and run again (delete PREDICTED tire and QUERY tire). The annotation in the “QUERY” tire is the most uncertain data, efficient improvement of accuracy at the next cycle can be expected if the query is annotated
