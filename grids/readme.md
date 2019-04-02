[Link to Goa_Fusion_Table](https://www.google.com/fusiontables/DataSource?docid=1vFIAYx9Da2dGRS2JSC5Etal7qPY4yABGaESA7iFR)

# Pipeline
1. Make Grid Csv with admin_data  
2. Make json file using makeJson.py (python3 makeJson.py India_Goa_2000_m.csv)  
3. Break folder containing tiff using folder containing json using extractGridsTiffs.py  
4. Make RGB images from tiff images using makeRgbLandsat.py (python3 makeRgbLandsat.py apr_19_cutFiles apr_19_cutFiles_rgb)    
5. Make SOL csv from cutFiles using makeSolCsv.py (python3 makeSolCsv.py apr_19_cutFiles)  
