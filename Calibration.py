
# Camera calibration using OpenCV APIs.

# Author: 
# Yaoyu Hu <yaoyuh@andrew.cmu.edu>

# General packages.
import argparse
import cv2
import glob
import numpy as np
import os

# Local packages.
from ChessboardDetector import downsample_and_detect_corners

# Global constants.
G_FLAG_DEBUG  = 0
G_DEBUG_COUNT = 2

FN_EXT                    = ".dat"
FN_CAMERA_MATRIX          = "CameraMatrix"
FN_DISTORTION_COEFFICIENT = "DistortionCoefficient"

def test_dir(d):
    '''
    Create the directory if the target does not exist. 

    Arguments:
    d (str): Directory name.

    Returns:
    No return values.
    '''
    if ( not os.path.isdir(d) ):
        os.makedirs(d)

def find_files(pattern):
    '''
    Find files by filename pattern.

    Arguments:
    pattern (str): The search pattern.

    Returns: 
    A list of filenames.
    '''
    fns = sorted( glob.glob( '%s' % ( pattern ) ) )

    if ( 0 == len(fns) ):
        raise Exception('No files found by pattern %s. ' % (pattern))

    return fns

def read_image(fn, flagGray=False, dtype=np.uint8):
    '''
    Read an image from the file system.

    Arguments:
    fn (str): Filename.
    flagGray (bool): Set True to convert the image to grayscale format.
    dtype (NumPy dtype): The desired image data type.

    Returns: 
    NumPy array.
    '''
    if ( not os.path.isfile(fn) ):
        raise Exception("%s does not exits. " % (fn))
    
    img = cv2.imread(fn, cv2.IMREAD_UNCHANGED)
    if ( flagGray and img.ndim >= 3):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 \
            else cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    
    if ( dtype != np.uint8 ):
        return img.astype(dtype)
    else:
        return img

def target_collection( fns, gridRow, gridCol, gridSize ):
    '''
    Detect the chessboard and collect data from image a list.

    Arguments: 
    fns (list of str): The image files.
    gridRow: Row number of crossings on the target chessboard.
    gridCol: Column number of crossings on the target chessboard.
    gridSize: Actual metric size of the grids in the chessboard.

    Returns:
    objPoints: Chessboard crossing coordinates.
    imgPoints: Detected pixel coordinates of the chessboard crossings. 
    img.shape: Shape of the images. (H, W).
    '''

    # Prepare object points.
    objIdx = np.zeros( (gridCol * gridRow, 3), np.float32 )
    objIdx[:, :2] = np.mgrid[0:gridCol, 0:gridRow].T.reshape(-1, 2)*gridSize

    # Arrays to store object points and image points from all images.
    objPoints = [] # 3D points in the real world.
    imgPoints = [] # 2D points in the image plane.

    nImages = len(fns)

    print("%d files to process..." % (nImages))

    count        = 0
    countFailed  = 0

    for f in fns:
        print("Process %s (%d / %d)..." % (f, count+1, nImages), end = '')

        img = read_image(f, flagGray=True)

        # Find the corners on the checkerboard.
        ret, corners = downsample_and_detect_corners(img, (gridCol, gridRow) )

        # Check the result.
        if ( ret == True ):
            objPoints.append(objIdx)
            imgPoints.append(corners)

            imgCC = np.repeat( img.reshape( img.shape[0], img.shape[1], 1 ), 3, axis=-1 )
            imgCC = cv2.drawChessboardCorners(imgCC, (gridCol, gridRow), corners, ret)
            imgResized = cv2.resize(imgCC, (960, 640))
            cv2.imshow('img', imgResized)
            cv2.waitKey(500)

            print("OK.")
        else:
            print("Failed.")

        count += 1

    cv2.destroyAllWindows()

    if ( countFailed > 0 ):
        print("%d of %d images failed." % (countFailed, nImages))

    return objPoints, imgPoints, img.shape

def calibrate_single_camera( imgFns, gridRow, gridCol, gridSize, flagLowDistortion=False):
    """
    Calibrate a single camera.

    Arguments: 
    imgFns (list of str): The image files.
    gridRow: Row number of crossings on the target chessboard.
    gridCol: Column number of crossings on the target chessboard.
    gridSize: Actual metric size of the grids in the chessboard.

    Returns:
    cameraMatrix (array): 3x3 intrinsic matrix. 
    distortionCoefficients (array): Distortion coefficients.
    reprojectError (float): Reprojection error.  
    """

    # Collect the target data.
    objPoints, imgPoints, imgShape = target_collection( 
        imgFns, gridRow, gridCol, gridSize )

    print("Begin calibrating...")

    calibFlags = cv2.CALIB_FIX_K4 + cv2.CALIB_FIX_K5 + cv2.CALIB_FIX_K6
    if ( flagLowDistortion ):
        calibFlags += cv2.CALIB_FIX_ASPECT_RATIO + cv2.CALIB_FIX_K3
        print('Enable low distortion mode. ')

    # Calibration.
    reprojectError, cameraMatrix, distortionCoefficients, _, _ \
     = cv2.calibrateCamera(objPoints, imgPoints, imgShape[::-1], 
        np.eye(3, dtype=np.float64), None, flags=calibFlags )
    # Need cv2.CALIB_RATIONAL_MODEL if number of k > 3. OpenCV 4.5 or later will not need this one.
    # May use cv2.CALIB_ZERO_TANGENT_DIST to disable the calibration of the p distortion coefficients.

    return cameraMatrix, distortionCoefficients, reprojectError

def handle_args():
    parser = argparse.ArgumentParser(description="Calibrate single camera.")

    parser.add_argument("basedir", type=str, 
        help="The base directory.")
    
    parser.add_argument("imgdir", type=str, 
        help="The sub-directory for the input images.")
    
    parser.add_argument("outdir", type=str, 
        help="The sub-directory for the output results.")

    parser.add_argument("--row", type=int, default=8,
        help="The number of row of corners on the chessboard.")

    parser.add_argument("--col", type=int, default=11, 
        help="The number of column of corners on the chessboard.")

    parser.add_argument("--csize", type=float, default=0.0015, 
        help="The width of the squares on the chessboard. Unit m.")

    parser.add_argument("--image-pattern", type=str, default="*.png", 
        help="The file search pattern for the input images.")

    parser.add_argument("--low-distortion", action='store_true', default=False, 
        help='Set this flag to enable low distortion calibration. ')
    
    args = parser.parse_args()

    return args

def main():
    print('%s' % ( os.path.basename(__file__) ))

    # Handle the arguments.
    args = handle_args()

    print("Begin calibrating %s/%s. " % ( args.basedir, args.imgdir ))

    imgDir = "%s/%s" % ( args.basedir, args.imgdir )
    outDir = "%s/%s" % ( args.basedir, args.outdir )

    # Prepare the output directory.
    test_dir( outDir )

    # Prepare the filenames.
    imgFns = find_files( "%s/%s" % ( imgDir, args.image_pattern ) )

    cameraMatrix, distortionCoefficients, reprojectError = \
        calibrate_single_camera( imgFns, args.row, args.col, args.csize, 
        args.low_distortion )

    # Print and save the data.
    print("reprojectError = \n{}".format(reprojectError))
    reprojectErrorWrapper = np.array([reprojectError])
    np.savetxt(os.path.join(outDir, "ReprojectError.dat"), reprojectErrorWrapper)

    print("cameraMatrix = \n{}".format(cameraMatrix))
    np.savetxt( os.path.join(outDir, '%s%s' % (FN_CAMERA_MATRIX, FN_EXT)), cameraMatrix )

    print("distortionCoefficients = \n{}".format(distortionCoefficients))
    np.savetxt( os.path.join(outDir, '%s%s' % (FN_DISTORTION_COEFFICIENT, FN_EXT)), distortionCoefficients )

    print("Done with %s. " % ( imgDir ))

    return 0

if __name__ == "__main__":
    import sys
    sys.exit( main() )