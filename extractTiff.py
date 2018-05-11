import json
import sys
import rasterio
from rasterio.tools.mask import mask

tiffFileName=sys.argv[1]
jsonFileName=sys.argv[2]
startIdStr=sys.argv[3]
dirName=sys.argv[4]
# dirName="../croppedImages/";

stateData = json.loads(open(jsonFileName).read())



for currVillageFeature in stateData["features"]:
  try:
    #currVillageFeature=stateData["features"][0]
    vCode2011=currVillageFeature["properties"]["village_code_2011"]
    vCode2001=currVillageFeature["properties"]["village_code_2001"]
    vId=currVillageFeature["properties"]["ID"]
    if(vId[:2]!=startIdStr):
        continue
    geoms=currVillageFeature["geometry"]
    listGeom=[]
    listGeom.append(geoms)
    geoms=listGeom
    with rasterio.open(tiffFileName) as src:
      out_image, out_transform = mask(src, geoms, crop=True)
    
    out_meta = src.meta.copy()
    print(vCode2011)

    # save the resulting raster  
    out_meta.update({"driver": "GTiff",
        "height": out_image.shape[1],
        "width": out_image.shape[2],
    "transform": out_transform})

    with rasterio.open(dirName+tiffFileName[:-4]+"@"+str(vCode2001)+"@"+vId+"@"+str(vCode2011)+".tif", "w", **out_meta) as dest:
      dest.write(out_image)
  except:
    continue