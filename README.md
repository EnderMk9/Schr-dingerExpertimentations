# SchrödingerExpertimentations
Experimentations on solving numerically Schrödinger's Equation

Requires
- Numpy
- Scipy
- Matplotlib
- Mayavi

For making videos with mayavi, you need ffmpeg, and then in a console in the folder where the frames are located, run:

    ffmpeg -r 30 -pattern_type glob -i '*.png' -vcodec libx264 -crf 20 -pix_fmt yuv420p test.mp4
