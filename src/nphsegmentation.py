import logging as log
import os
import argparse
import pathlib
from TestFunc import * 
from CSFseg import *
import CTtools
from subprocess import call
from postSkullStrip import postSkullStrip
# TODO: Do qualified imports so it's easier to figure out where everything is from

# TODO: Clear out unused imports, ensure they aren't really needed

# import subprocess
# import sys
# from os.path import exists


def preprocessing(inName, outPath):
    """
    Performs preprocessing of the given *.nii.gz scan, prior to feeding it to the model. 

    Several stages are taken:

    1. Skull stripping
    2. Affine transformation: This is for the postprocessing stage after the model has finished.
    
    """
    
    outName =  outPath.parent / (outPath.name.split('.')[0] + "_Mask.nii.gz")
    # print(outName)
    # iname = "intermediate_" + outPath.name
    print(f"{outPath=}")
    #ipath = outPath.parent / ("intermediate_" + outPath.name)
    ipath = outPath
    print(f"{ipath=}")
    # breakpoint()
    
    ct_scan_path = outPath
    ct_scan_wodevice_bone = CTtools.bone_extracted(inName, ipath)
    #ct_scan_wodevice = ct_scan_path
    # ct_scan_wodevice = outPath
    
    # MNI_152_bone = os.path.join(os.getcwd(),'MNI152_T1_1mm_bone.nii.gz')
    MNI_152_bone = pathlib.Path('MNI152_T1_1mm_bone.nii.gz')
    MNI_152 = os.path.join(os.getcwd(),'MNI152_T1_1mm.nii.gz')
    # bspline_path = os.path.join(os.getcwd(), 'Par0000bspline.txt')
    bspline_path = pathlib.Path('Par0000bspline.txt')

    # nameOfAffineMatrix = ct_scan_path[:ct_scan_path.find('.nii.gz')] + '_affine.mat'
    nameOfAffineMatrix = outPath.name + '_affine.mat'

    call(['flirt', '-in', ct_scan_wodevice_bone, '-ref', MNI_152_bone, '-omat', 
          nameOfAffineMatrix, '-bins', '256', '-searchrx', '-180', '180', '-searchry', '-180', '180', 
          '-searchrz', '-180', '180', '-dof', '12', '-interp', 'trilinear'])

    stripped = postSkullStrip(inName, ipath)
    nii_image = nib.Nifti1Image(stripped.astype(np.float32), affine=np.eye(4))
    nib.save(nii_image, outName) # the corrected raw scans, should have a good number of slices bounded to just the brain + maybe some thin shape of the skull

    #breakpoint()
    
    # Affine transformation
    # call(['flirt', '-in', ct_scan_wodevice, '-ref', MNI_152, '-applyxfm', '-init', nameOfAffineMatrix, '-out', str(ct_scan_wodevice).split('.')[0] + '_MNI152.nii.gz'])
    # call(['flirt', '-in', inName, '-ref', MNI_152, '-applyxfm', '-init', nameOfAffineMatrix, '-out', str(inName.name).split('.')[0] + '_MNI152.nii.gz'])

    # # the code below implement the deformable transformation

    # ct_scan_wodevice_contraststretching = CTtools.contrastStretch(str(inName), output_name=str(inName.name.split('.')[0]))

    # call(['flirt', '-in', ct_scan_wodevice_contraststretching, '-ref', MNI_152, '-applyxfm', '-init', nameOfAffineMatrix, '-out', 
    #     ct_scan_wodevice_contraststretching[:ct_scan_wodevice_contraststretching.find('.nii.gz')]+'_MNI152.nii.gz'])

    # call(['elastix', '-m', ct_scan_wodevice_contraststretching[:ct_scan_wodevice_contraststretching.find('.nii.gz')]+'_MNI152.nii.gz', '-f', MNI_152, '-out', os.path.dirname(ct_scan_path), '-p', bspline_path])
    # # breakpoint()
    

def main(input_path, output_path, rdir, betPath=pathlib.Path('/module/src/skull-strip/'), gtPath='gt', device='cuda', BS=200, modelPath=None):
    log.info("f{input_path.stem=}")
    print(f"{input_path=}")
        
    device = checkDevice(device)
    model  = loadModel(modelPath, device)
    
    # output_path_dict = dict()
    
    # input_name = input_path.name.split('.')[0]
    input_file = input_path.name
    input_name = input_path.name.split('.')[0]
    #breakpoint()
    
    # skull_strip(input_path, betPath / input_file, running_dir="/home/cirrus/projects/vision/Bisque_Module_NPH/Modules/NPHSegmentation/src/")
    # skull_strip(input_path, betPath / input_file, running_dir=rdir)
    preprocessing(input_path, betPath / input_file)
    
    resultName = runTest(input_name, output_path, input_path, betPath, device, BS, model) # Filename
    # maxArea, maxPos, finalimg, outputName = segVent(input_name, output_path, resultName) # outputName is filename

    # result = os.path.join(output_path, outputName) 
    result = os.path.join(output_path, resultName) 
    
    return result

if __name__== "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--modelPath', default='model_backup/epoch50_2Dresnet_skullstrip5Class.pt')
    parser.add_argument('--outputPath', default='reconstructed')
    parser.add_argument('--dataPath', default='data-split/Scans')
    parser.add_argument('--betPath', default='data-split/skull-strip')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch_size', default=200)
    parser.add_argument('--gtPath', default = 'data-split/gt')
    parser.add_argument('--strip_script_path', default = '/module/src')

    args = parser.parse_args()
    
    
    main(input_path  = pathlib.Path(args.dataPath),
         modelPath   = pathlib.Path(args.modelPath),
         output_path = pathlib.Path(args.outputPath),
         betPath     = pathlib.Path(args.betPath),
         gtPath      = args.gtPath,
         device      = args.device,
         BS          = args.batch_size,
         rdir        = args.strip_script_path)
