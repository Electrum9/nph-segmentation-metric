#
import logging as log
import os
import argparse
import pathlib
import subprocess
from TestFunc import * 
from CSFseg import *
from os.path import exists
import CTtools
from subprocess import call
import sys
from postSkullStrip import postSkullStrip



def imageList(dataPath):
    fileName=[]
    fileList=[]
#     if os.path.isfile(dataPath) and '.nii' in dataPath:
#         fileList+=[dataPath]
#         temp=dataPath
#         if '/' in temp: temp=temp.split('/')[-1]
#         fileName+=[temp.split('.nii')[0]]
#         print(fileName)

    if os.path.isdir(dataPath):
        
        fileList += [d for d in os.listdir(dataPath) if '.nii' in d]
        
        for temp in fileList:
            fileName += [temp.split('.nii')[0]]
         
    else:
        raise ValueError('Invalid data path input')
    
    fileList, fileName = zip(*sorted(zip(fileList, fileName)))
    return fileList, fileName

#skull strip
def skull_strip(inName, outName):
# def skull_strip(inName, outName, running_dir):
    # subprocess.call(["fslmaths", "--help"])
    # subprocess.call(['bash', running_dir + 'skull_strip.sh', inName, str(outName), "&> errors.txt"])
    # print("finished shell script")
    # fillHoles(outName)
    # print(outName, 'skull strip done.')

    # ct_scan_path = inName
    # MNI_152_bone = os.path.join(pathlib.Path().cwd() / "MNI/", 'MNI152_T1_1mm_bone.nii.gz')
    # MNI_152 = os.path.join(pathlib.Path().cwd() / "MNI/", 'MNI152_T1_1mm.nii.gz')

    # bspline_path = os.path.join(pathlib.Path().cwd() / "MNI/", 'Par0000bspline.txt')

    # nameOfAffineMatrix = ct_scan_path[:ct_scan_path.find('.nii.gz')] + '_affine.mat'

    # ct_scan_wodevice = ct_scan_path

    outName =  outName.parent / (outName.name + "_Mask.nii.gz")
    print(outName)
    iname = "intermediate_" + outName.name
    ipath = outName.parent / iname
    print(f"{ipath=}")
    
    CTtools.bone_extracted(inName, ipath)


    stripped = postSkullStrip(inName, ipath)
    nii_image = nib.Nifti1Image(stripped.astype(np.float32), affine=np.eye(4))
    nib.save(nii_image, outName) # the corrected raw scans, should have a good number of slices bounded to just the brain + maybe some thin shape of the skull
    # call(['flirt', '-in', ct_scan_wodevice_bone, '-ref', MNI_152_bone, '-omat', nameOfAffineMatrix, '-bins', '256',
    #       '-searchrx', '-180', '180', '-searchry', '-180', '180', '-searchrz', '-180', '180', '-dof', '12',
    #       '-interp', 'trilinear'])
    


#run test 



def main(input_path, output_path, rdir, betPath=pathlib.Path('/module/src/skull-strip/'), gtPath='gt', device='cuda', BS=200, modelPath=None):
    log.info("f{input_path.stem=}")
        
    # if os.path.isfile(input_path):
    #     f = os.path.basename(input_path)
    #     fileList = [f]
    #     fileName = [f.split('.nii')[0]]
    # else:
    #     fileList, fileName = imageList(input_path)
    
    device = checkDevice(device)
    model  = loadModel(modelPath, device)
    
    # print('Total number:', len(fileName))
    
    output_path_dict = dict()
    
    # for i in range(len(fileName)):
    # result = None
    
    # for i, fname in enumerate(fileName):
        
    # if not exists(os.path.join(input_path,'{}.nii.gz'.format(fileName[i]))):
    #     print(input_path, 'not exists')
    
    # n = nib.load(input_path)
    input_name = input_path.name.split('.')[0]
    
    # skull_strip(input_path, betPath / input_name, running_dir="/home/cirrus/projects/vision/Bisque_Module_NPH/Modules/NPHSegmentation/src/")
    # skull_strip(input_path, betPath / input_name, running_dir=rdir)
    skull_strip(input_path, betPath / input_name)
    
    resultName = runTest(input_name, output_path, input_path, betPath, device, BS, model) # Filename
    # maxArea, maxPos, finalimg =segVent(fileName[i], outputPath, resultName)
    # maxArea, maxPos, finalimg, outputName = segVent(input_name, output_path, resultName) # outputName is filename

    # output_path_dict["Segmented Image"] = os.path.join(output_path, outputName) 
    # result = os.path.join(output_path, outputName) 
    result = os.path.join(output_path, resultName) 
    
    # currently overwrites the previous entry, fix so it handles multiple images
        

    return result
        # correct, total, TP, FP, FN            = diceScore(fileName[i],finalimg,gtPath)
        
        # with open('CSFmax.txt',"a+") as file:
        #     file.write('{},{},{}\n'.format(fileName[i], maxPos, maxArea))

    
#             print(fileName)


if __name__== "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--modelPath', default='model_backup/epoch149_ResNet2D5Class.pt')
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
