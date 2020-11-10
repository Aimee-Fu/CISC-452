
from urllib.request import urlopen
from io import StringIO
import csv
lis = []
data=urlopen("https://raw.githubusercontent.com/microsoft/r-server-hospital-length-of-stay/master/Data/LengthOfStay.csv").read().decode('ascii','ignore')
dataFile=StringIO(data)
csvReader=csv.reader(dataFile)
for row in csvReader:
    lis.append(row)

