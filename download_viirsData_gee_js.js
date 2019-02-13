var india_boundary = /* color: #d63000 */ee.Geometry.Polygon(
        [[[71.80341423173991, 32.71935965386061],
          [70.92450798173991, 29.713337663630544],
          [67.23310173173991, 26.14212021020959],
          [66.17841423173991, 21.970043291298335],
          [70.74872673173991, 19.006550348509975],
          [72.50653923173991, 11.379335044942009],
          [77.25263298173991, 5.644954486620402],
          [81.82294548173991, 9.997518966322861],
          [83.93232048173991, 14.633013880242805],
          [98.52216423173991, 22.620598491511196],
          [98.87372673173991, 30.170296252435413],
          [87.09638298173991, 28.94710683082738],
          [80.76825798173991, 31.678156886375124],
          [81.99872673173991, 35.62738277876477],
          [79.36200798173991, 37.043336067703805],
          [71.27607048173991, 37.3234271558105]]]);
		  
		  
		  
		  
var d1='2012-10-01';
var d2='2012-10-31';

var dataset = (ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG').select('avg_rad').filter(ee.Filter.date(d1,d2)) );

var listOfImages = dataset.toList(dataset.size());
print(listOfImages);
var img1 = listOfImages.get(0);
print(img1);


Export.image.toDrive({
  image: img1,
  description: 'India_'+d1+'_'+d2+'_500',
  folder: 'ayushGEE',
  scale: 500,
  region: india_boundary,
  maxPixels: 14261279780
});



		  