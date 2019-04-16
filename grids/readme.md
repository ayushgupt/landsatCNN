[Link to Goa_Fusion_Table](https://www.google.com/fusiontables/DataSource?docid=1vFIAYx9Da2dGRS2JSC5Etal7qPY4yABGaESA7iFR)  
[GEE Tool to get Grids Points in CSV](https://goo.gl/bJv9BF)

# Fusion Tables
[Jaipur epoch 0,5,10](https://www.google.com/fusiontables/DataSource?docid=1cTqMW1Y_VGNqS6oHz3NZOH0Ef-65tnfD9KaJZrwt)  
[Jaipur epoch 23](https://www.google.com/fusiontables/DataSource?docid=1zuqeIKAhVliemXXZADRCatMABbO6T-8Ppq6QXFzk)  
[Giridih epoch 23](https://www.google.com/fusiontables/DataSource?docid=1NHi6Sj5TP8wiu_m06Qr1YjQ_LLH_k9hTVLdOECKB)  
[Giridih epoch 5](https://www.google.com/fusiontables/DataSource?docid=1g4ndgSpH0TE0ckg1Ek3UHCFXG4WNl9iz3mPn2beX)

| District 	| NBU   	| BU  	| Epoch 	|
|:--------:	|------	|------	|-------	|
| Jaipur   	| 1116 	| 1930 	| 0     	|
| Jaipur   	| 1030 	| 2016 	| 5     	|
| Jaipur   	| 1189 	| 1858 	| 10    	|
| Jaipur   	| 1076 	| 1970 	| 23    	|
| Giridih  	| 986  	| 312  	| 5     	|
| Giridih  	| 1105 	| 193  	| 23    	|


# Pipeline
1. Make Grid Csv with admin_data  
2. Make json file using makeJson.py (python3 makeJson.py India_Goa_2000_m.csv)  
3. Break folder containing tiff using folder containing json using extractGridsTiffs.py  
4. Make RGB images from tiff images using makeRgbLandsat.py (python3 makeRgbLandsat.py apr_19_cutFiles apr_19_cutFiles_rgb)    
5. Make SOL csv from cutFiles using makeSolCsv.py (python3 makeSolCsv.py apr_19_cutFiles)  



# History of command line 
 2019  sudo cp -r /my_mnt_dir/giridih ./  
 2020  clear  
 2021  ls  
 2022  sudo chmod 777 giridih/  
 2023  ls  
 2024  mv giridih/ data/  
 2025  ls  
 2026  cd ..  
 2027  python3 makeJson.py giridih/India_Jharkhand_Giridih\ syst_1970_m.csv   
 2028  ls giridih/  
 2029  mkdir giridih/json  
 2030  mv giridih/India_Jharkhand_Giridih\ syst_1970_m.json giridih/json/  
 2031  ls giridih/  
 2032  cat extractGridsTiffs.py  
 2033  conda deactivate  
 2034  python3 extractGridsTiffs.py giridih/data giridih/json  
 2035  python3 makeRgbLandsat.py giridih/data_cutFiles giridih/data_cutFiles_rgb  
 2036  ls giridih/  
 2037  mkdir jaipur/  
 2038  ls jaipur/  
 2039  mkdir giridih/data_cutFiles_rgb_1  
 2040  mv -r giridih/data_cutFiles_rgb giridih/data_cutFiles_rgb_1/  
 2041  mv giridih/data_cutFiles_rgb giridih/data_cutFiles_rgb_1/  
 2042  ls  
 2043  ls giridih/  
 2044  conda activate  
 2045  python3 testResnet18_csv.py splitData_25_100/train_checkpoints/epoch_23.ckpt jaipur/data_2011_rgb_1/  
 2046  mv splitData_25_100/train_checkpoints/epoch_23_predict.csv jaipur/  
 2047  python3 testResnet18_csv.py splitData_25_100/train_checkpoints/epoch_23.ckpt giridih/data_cutFiles_rgb_1/  
 2048  python3 json_merger.py jaipur/epoch_23_predict.csv jaipur/json/India_Rajasthan_Jaipur\ syst_1970_m.json   
 2049  cat jaipur/epoch_23_predict.csv | head -10  
 2050  python3 json_merger.py jaipur/epoch_23_predict.csv jaipur/json/India_Rajasthan_Jaipur\ syst_1970_m.json   
 2051  mv splitData_25_100/train_checkpoints/epoch_23_predict.csv giridih/  
 2052  python3 json_merger.py giridih/epoch_23_predict.csv giridih/json/India_Jharkhand_Giridih\ syst_1970_m.json   
 2053  cd giridih/json/  
 2054  tokml India_Jharkhand_Giridih\ syst_1970_m_withPred.json > India_Jharkhand_Giridih\ syst_1970_m_withPred.kml  
 2055  cd ../../jaipur/json/  
 2056  tokml India_Rajasthan_Jaipur\ syst_1970_m_withPred.json > India_Rajasthan_Jaipur\ syst_1970_m_withPred.kml  
