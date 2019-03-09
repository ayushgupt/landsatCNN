import json
import sys
import rasterio
from rasterio.mask import mask
from json import loads
import sys
import os
from os import listdir
from os.path import isfile, join
# from rasterio.tools.mask import mask

# tiffFileName=sys.argv[1]
# folder_containing_tifffiles = "data/copyDummyData"
folder_containing_tifffiles = sys.argv[1]
output_directory = folder_containing_tifffiles+'_cutFiles'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)
allTiffFiles = [f for f in listdir(folder_containing_tifffiles) if isfile(join(folder_containing_tifffiles, f))]

xaa_file = "./../../../geojsonFile/xaa.json"
xab_file = "./../../../geojsonFile/xab.json"
xac_file = "./../../../geojsonFile/xac.json"

jsonFileList = [xaa_file,xab_file,xac_file]

for tiffFileName in allTiffFiles:
    for jsonFileName in jsonFileList:
        stateData = json.loads(open(jsonFileName).read())
        print('tiffFileName',tiffFileName)
        print('jsonFileName',jsonFileName)
        for currVillageFeature in stateData["features"]:
            try:
                vCode2011=currVillageFeature["properties"]["village_code_2011"]
                vCode2001=currVillageFeature["properties"]["village_code_2001"]
                vId=currVillageFeature["properties"]["ID"]
                geoms=currVillageFeature["geometry"]
                listGeom=[]
                listGeom.append(geoms)
                geoms=listGeom
                with rasterio.open(folder_containing_tifffiles+'/'+tiffFileName) as src:
                    out_image, out_transform = mask(src, geoms, crop=True)

                out_meta = src.meta.copy()
                # save the resulting raster  
                out_meta.update({"driver": "GTiff",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                "transform": out_transform})
                saveFileName=output_directory+'/'+tiffFileName[:-4]+"@"+str(vCode2001)+"@"+vId+"@"+str(vCode2011)+".tif"
                print(vCode2011)
                with rasterio.open(saveFileName, "w", **out_meta) as dest:
                    dest.write(out_image)
            except:
                continue