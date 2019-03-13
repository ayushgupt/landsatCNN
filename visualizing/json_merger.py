


import pandas as pd
import numpy as np
import pickle

import json
import pandas as pd
with open('2011_Dist.geojson', 'r') as f:
	a1=json.load(f)

# f11=open('totalx.json','w')
# json_data = json.dumps(a1)
# f11.write(json_data)



file_list = ['districtCombined.csv']
col_list=['sureDoubtfulNegativeMaskSum','unsureDoubtfulMaskSum','slopePositive','slopeNegative','stderrSum','stderrMedian','stderrMean','confidentForPositive']

for filename in file_list:
	df2=pd.read_csv(filename)
	asset=filename.split('_')[0]
	print(asset)	
	for index, row in df2.iterrows():
		for i in range(len(a1["features"])):
			if(a1["features"][i]["properties"]["censuscode"]==row['District']):
				# print(index)
				for namers in col_list:
					finamers=asset+'_'+namers
					a1["features"][i]["properties"][finamers]=float(row[namers])


with open('total.json','w') as f:
    json_data = json.dumps(a1)
    f.write(json_data)


# file=open('2011_district.pkl','rb')
# df1=pickle.load(file)

# df = pd.read_csv('GeoJson.csv',sep='|',header=(0))
# print(df)

# df=pd.merge(df,df1,left_on='censuscode',right_on='district_id',how='inner')

# print(df)
# df.to_csv('georesults.csv',sep='|')
