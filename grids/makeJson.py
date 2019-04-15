import pandas as pd
import numpy as np
import sys
coordinateCsvFile = sys.argv[1]
jsonFileName = coordinateCsvFile[:-3]+'json'

coordinateDf=pd.read_csv(coordinateCsvFile)
# coordinateDf.head()


latLongArray=coordinateDf[['latitude','longitude','adm1_name','adm2_name']].values
# print(latLongArray.shape)


from collections import defaultdict
pointList = defaultdict(lambda: 'not_present')
for lat,long,adm1_name,adm2_name in latLongArray:
    pointList[(lat,long)]=(adm1_name,adm2_name)
    
    
    
latitudeUniqueValues=([t[0] for t in latLongArray])
myset = set(latitudeUniqueValues)
latitudeUniqueValues=list(myset)
latitudeUniqueValues.sort()
longitudeUniqueValues=([t[1] for t in latLongArray])
myset = set(longitudeUniqueValues)
longitudeUniqueValues=list(myset)
longitudeUniqueValues.sort()


temp=[]
for lat_index in range(len(latitudeUniqueValues)-1):
    for long_index in range(len(longitudeUniqueValues)-1):
        top_left_Point=(latitudeUniqueValues[lat_index],longitudeUniqueValues[long_index])
        top_rigt_Point=(latitudeUniqueValues[lat_index],longitudeUniqueValues[long_index+1])
        bot_left_Point=(latitudeUniqueValues[lat_index+1],longitudeUniqueValues[long_index])
        bot_rigt_Point=(latitudeUniqueValues[lat_index+1],longitudeUniqueValues[long_index+1])
        if(pointList[top_left_Point]!='not_present' and pointList[top_rigt_Point]!='not_present' and 
          pointList[bot_left_Point]!='not_present' and pointList[bot_rigt_Point]!='not_present'):
            temp.append(1)
        
        
# print(len(temp))



initialStr='{"type": "FeatureCollection","features": ['
endStr=']}'


featureStr_List=[]
count_grid=0
for lat_index in range(len(latitudeUniqueValues)-1):
    for long_index in range(len(longitudeUniqueValues)-1):
        top_left_Point=(latitudeUniqueValues[lat_index],longitudeUniqueValues[long_index])
        top_rigt_Point=(latitudeUniqueValues[lat_index],longitudeUniqueValues[long_index+1])
        bot_left_Point=(latitudeUniqueValues[lat_index+1],longitudeUniqueValues[long_index])
        bot_rigt_Point=(latitudeUniqueValues[lat_index+1],longitudeUniqueValues[long_index+1])
        if(pointList[top_left_Point]!='not_present' and pointList[top_rigt_Point]!='not_present' and 
          pointList[bot_left_Point]!='not_present' and pointList[bot_rigt_Point]!='not_present'):
            featureStr = '{ "type": "Feature", "properties": { "ID": "'
            featureStr+= str(pointList[top_left_Point][0])+'@'+str(pointList[top_left_Point][1])+'@'+str(count_grid)
            count_grid+=1
            featureStr+='"}, "geometry": { "type": "Polygon", "coordinates": [ ['
            featureStr +='['+str(top_left_Point[1])+','+str(top_left_Point[0])+'],' 
            featureStr +='['+str(top_rigt_Point[1])+','+str(top_rigt_Point[0])+'],' 
            featureStr +='['+str(bot_rigt_Point[1])+','+str(bot_rigt_Point[0])+'],' 
            featureStr +='['+str(bot_left_Point[1])+','+str(bot_left_Point[0])+'],' 
            featureStr +='['+str(top_left_Point[1])+','+str(top_left_Point[0])+']' 
            featureStr+='] ] } }'
            featureStr_List.append(featureStr)
    

outputJson=initialStr+(','.join(featureStr_List))+endStr




f = open(jsonFileName, "w")
f.write(outputJson)
f.close()
