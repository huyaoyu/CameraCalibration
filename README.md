# Calibrating a single camera by OpenCV APIs

## Usage

```
python3 Calibration \
    <base dir> <image dir> <output dir> \
    --row <row number or crossings> \
    --col <column number of crossings> \
    --csize <metric size of the chessboard grid> \
    --image-pattern <image file search pattern> \
    --low-distortion
```

The optional arguments:
- row, col: The row and column numbers of the crossings in the target chessboard. Default 6, 8.
- csize: The actial metric size of the grids in the chessboard. Default 0.115. Unit: meter.
- image-pattern: The filename search pattern. Default '*.png'
- low-distortion: Set this flag to enable the low-distortion mode. Under this mode, the focal length fx and fy are forced to be equal and only the k1, k2, p1, p2 distortion coefficients are computed.

\<image dir\> and \<output dir\> are the relative directories under \<base dir\>. 
