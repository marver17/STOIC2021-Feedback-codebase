import numpy as np
import SimpleITK as sitk
from algorithm.lungmask import mask
import re
import os 
from algorithm.lesion_segmentation import lesion_segmentation



def crop(mask,image):
    mask[mask == 2] = 1
    label = sitk.GetImageFromArray(mask,sitk.sitkInt8)
    lsif = sitk.LabelShapeStatisticsImageFilter()
    lsif.Execute(label)
    x1,y1,z1,x2,y2,z2 = lsif.GetBoundingBox(1)
    L = np.array([x1,y1,z1] , dtype='int').tolist()
    x,y,z = label.GetSize()
    U =[x - x1 - x2,y - y1- y2,z - z1 - z2]
    crop = sitk.CropImageFilter()
    crop.SetLowerBoundaryCropSize(L)
    crop.SetUpperBoundaryCropSize(U)
    label_crop  = crop.Execute(label)
    image_crop = crop.Execute(image)
    for k in image.GetMetaDataKeys():
        image_crop.SetMetaData(k,image.GetMetaData(k))
    return label_crop,image_crop





def preprocess(input_image: sitk.Image,input_path) -> sitk.Image:
    label = mask.apply(input_image)
    label_crop,image_crop = crop(label,input_image)
    file_path = os.path.splitext(input_path)[0]
    file_name = file_path.split('/')[-1]
    path_crop = "./algorithm/temp_file/" + file_name +" _crop.mha"
    sitk.WriteImage(image_crop,path_crop)
    lesion_segmentation(path_crop,"./algorithm/temp_file/")
    return sitk.GetArrayFromImage(label_crop),image_crop





def age_sex_info(itk_image: sitk.Image) -> int :
    age = itk_image.GetMetaData('PatientAge')
    if any(c.isalpha() for c in age)  == True :
        age = int(re.findall('\d+', age )[0])
    else:
        age = int(age)

    sex = itk_image.GetMetaData('PatientSex')
    if sex == "F" :
        sex = 0
    else:
        sex = 1
    return age, sex

def componenti_connesse(sitk_image):
    img = sitk.Cast(sitk_image,sitk.sitkInt8)
    con = sitk.ConnectedComponent(img)
    label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
    label_shape_filter.Execute(con)
    com_connesse = label_shape_filter.GetNumberOfLabels()
    return com_connesse         

 

def extract_all_(sitk_image,input_path):
        lung_mask,image_crop = preprocess(sitk_image,input_path)
        age,sex = age_sex_info(image_crop)
        vox_dim = np.prod(np.array(image_crop.GetSpacing()))
        file_path = os.path.splitext(input_path)[0]
        file_name = file_path.split('/')[-1]
        print(file_name)
        #lesion_mask_sitk = sitk.ReadImage("./algorithm/temp_file/" + file_name + "_crop_seg.nii.gz")
        lesion_mask_sitk = sitk.ReadImage("./algorithm/temp_file/4133 _crop_seg.nii.gz")
        lesion_mask = sitk.GetArrayFromImage(lesion_mask_sitk)
        vol_lesion = np.sum(np.bool_(lesion_mask))
        n_con  = componenti_connesse(lesion_mask_sitk)
        if vol_lesion != 0:
            vol_masklung = np.sum(np.bool_(lung_mask))
            vol_lesion = np.sum(np.bool_(lesion_mask))/vol_masklung
            img = sitk.GetArrayFromImage(image_crop)[lesion_mask == 1]
            mean_intensity = np.mean(img)
            std_intensity = np.std(img)
            check_lesion = 1
            print("fine")
            return np.array([age,vol_lesion,mean_intensity,n_con]),check_lesion
        else :
            return [age,0,0,0],check_lesion
        

