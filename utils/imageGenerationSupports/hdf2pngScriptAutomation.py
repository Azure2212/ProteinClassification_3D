import os 
import numpy as np
import h5py
from PIL import Image  

pdfData_path = "/data2/TestEman2/HDF/HDF_90_12/HDF90"
output_folder = "/data2/atran16/ProteinClassification_AnhTuanTran/3D_PDB_Dataset/90_12/TrainProteinPNG90"

all_files = os.listdir(pdfData_path)
myListP = [file.split(".")[0] for file in all_files if file.endswith(".hdf")]
myListP = sorted(myListP)
print(myListP)

for nameP in myListP:
    file_path = f"{pdfData_path}/{nameP}.hdf"
    
    with h5py.File(file_path, "r") as hdf_file:
        images_group = hdf_file["MDF/images"]
        keys = sorted(images_group.keys(), key=lambda x: int(x))
        stack_list = [images_group[k]["image"][:] for k in keys]
        stack = np.stack(stack_list, axis=0)

    outputFolder2Save = f"{output_folder}/{nameP}"
    os.makedirs(outputFolder2Save, exist_ok=True)  
    
    for idx, img in enumerate(stack):
        img_8bit = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
        im = Image.fromarray(img_8bit)
        
        
        filename = os.path.join(outputFolder2Save, f"{idx:03d}.png")
        im.save(filename)

    print(f"conver {nameP} successful!")
print("Finish!")
