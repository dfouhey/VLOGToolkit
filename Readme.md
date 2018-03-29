# Welcome to the VLOG Dataset!

This repository contains useful tools for working with the VLOG dataset. We will be updating
this with additional tools as time goes on.

Dependencies:

* numpy
* scipy
* sklearn

You have to specify paths to files in the following files (although all the scripts will prompt you for paths):

* `DATA_ROOT` should contain the path of the labels file
* `CLIP_ROOT` should contain the path of the clips, dumped in the /X/Y/Z/v_TUVWXYZ/001/clip.mp4 format described in the data folder
* `FRAME_ROOT` should contain the path of the frames, dumped in the /X/Y/Z/v_TUVWXZ/001/frame000001.jpg format described in the data folder

## Evaluation

`eval.py` contains evaluation scripts that will let you reproduce the numbers reported in the paper given
the precomputed predictions provided with the dataset.

## Demo Reading Code

`demo.py` contains demo code that loads and displays the data. This will open a series of windows using `eog` and `cvlc`, so it may
or may not work depending on your computer. It also displays hand bounding boxes by using opencv.



