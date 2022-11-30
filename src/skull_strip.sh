touch fslmaths_exists.txt
echo $(which fslmaths) >> fslmaths_exists.txt

img=$1
echo ${img}
intensity=0.01
outfile=$2
echo ${outfile}
tmpfile=`mktemp`

echo "img=$1" >> fslmaths_exists.txt
echo "outfile=$2" >> fslmaths_exists.txt
echo "${outfile}_Mask" >> fslmaths_exists.txt

# Thresholding Image to 0-100
fslmaths "${img}" -thr 0.000000 -uthr 100.000000  "${outfile}"  >>errors.txt 2>&1
# Creating 0 - 100 mask to remask after filling
fslmaths "${outfile}"  -bin   "${tmpfile}"  >>errors.txt 2>&1;
fslmaths "${tmpfile}.nii.gz" -bin -fillh "${tmpfile}"  >>errors.txt 2>&1
# Presmoothing image
fslmaths "${outfile}"  -s 1 "${outfile}"  >>errors.txt 2>&1;
# Remasking Smoothed Image
fslmaths "${outfile}" -mas "${tmpfile}" "${outfile}"  >>errors.txt 2>&1
# Running bet2
bet2 "${outfile}" "${outfile}" -f ${intensity} -v 
# Using fslfill to fill in any holes in mask 
fslmaths "${outfile}" -bin -fillh "${outfile}_Mask"  >>errors.txt 2>&1
# Using the filled mask to mask original image
fslmaths "${img}" -mas "${outfile}_Mask"  "${outfile}"  >>errors.txt 2>&1


######################
## If no pre-smoothing
######################
# outfile_nosmooth="Head_Image_1_SS_0.01_nopresmooth"
# fslmaths "$img" -thr 0.000000 -uthr 100.000000  "${outfile_nosmooth}" 
# # Creating 0 - 100 mask to remask after filling
# fslmaths "${outfile_nosmooth}"  -bin   "${tmpfile}"; 
# fslmaths "${tmpfile}" -bin -fillh "${tmpfile}" 
# # Running bet2
# bet2 "${outfile_nosmooth}" "${outfile_nosmooth}" -f ${intensity} -v 
# # Using fslfill to fill in any holes in mask 
# fslmaths "${outfile_nosmooth}" -bin -fillh "${outfile_nosmooth}_Mask" 
# # Using the filled mask to mask original image
# fslmaths "$img" -mas "${outfile_nosmooth}_Mask"  "${outfile_nosmooth}" 
