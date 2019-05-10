# -*- coding: utf-8 -*-
"""
Created on Sun May  5 01:21:56 2019

@author: Pavel
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 23:26:36 2019

@author: Pavel
"""
#Import all needed packages
import numpy as np
import pandas as pd
import sys
pd.options.display.float_format = '{:.6f}'.format

# Importing all IO-Tables from year 1997 until 2011
edit = ' sep=',', header = 6, usecols=np.arange(1,45),skiprows = np.append(np.array([7,8],dtype=float),np.array([47]))'
io98 = pd.read_csv('C:/Users/Pavel/Desktop/IES/Bachelor Thesis/data/OECD IO Tables/csv import/IO 97-11 - IO - 98.csv', sep=',', header = 6, usecols=np.arange(2,45),skiprows = np.append(np.array([7,8],dtype=float),np.array([47])))
io99 = pd.read_csv('C:/Users/Pavel/Desktop/IES/Bachelor Thesis/data/OECD IO Tables/csv import/IO 97-11 - IO - 99.csv', sep=',', header = 6, usecols=np.arange(2,45),skiprows = np.append(np.array([7,8],dtype=float),np.array([47])))
io00 = pd.read_csv('C:/Users/Pavel/Desktop/IES/Bachelor Thesis/data/OECD IO Tables/csv import/IO 97-11 - IO - 00.csv', sep=',', header = 6, usecols=np.arange(2,45),skiprows = np.append(np.array([7,8],dtype=float),np.array([47])))
io01 = pd.read_csv('C:/Users/Pavel/Desktop/IES/Bachelor Thesis/data/OECD IO Tables/csv import/IO 97-11 - IO - 01.csv', sep=',', header = 6, usecols=np.arange(2,45),skiprows = np.append(np.array([7,8],dtype=float),np.array([47])))
io02 = pd.read_csv('C:/Users/Pavel/Desktop/IES/Bachelor Thesis/data/OECD IO Tables/csv import/IO 97-11 - IO - 02.csv', sep=',', header = 6, usecols=np.arange(2,45),skiprows = np.append(np.array([7,8],dtype=float),np.array([47])))
io03 = pd.read_csv('C:/Users/Pavel/Desktop/IES/Bachelor Thesis/data/OECD IO Tables/csv import/IO 97-11 - IO - 03.csv', sep=',', header = 6, usecols=np.arange(2,45),skiprows = np.append(np.array([7,8],dtype=float),np.array([47])))
io04 = pd.read_csv('C:/Users/Pavel/Desktop/IES/Bachelor Thesis/data/OECD IO Tables/csv import/IO 97-11 - IO - 04.csv', sep=',', header = 6, usecols=np.arange(2,45),skiprows = np.append(np.array([7,8],dtype=float),np.array([47])))
io05 = pd.read_csv('C:/Users/Pavel/Desktop/IES/Bachelor Thesis/data/OECD IO Tables/csv import/IO 97-11 - IO - 05.csv', sep=',', header = 6, usecols=np.arange(2,45),skiprows = np.append(np.array([7,8],dtype=float),np.array([47])))
io06 = pd.read_csv('C:/Users/Pavel/Desktop/IES/Bachelor Thesis/data/OECD IO Tables/csv import/IO 97-11 - IO - 06.csv', sep=',', header = 6, usecols=np.arange(2,45),skiprows = np.append(np.array([7,8],dtype=float),np.array([47])))
io07 = pd.read_csv('C:/Users/Pavel/Desktop/IES/Bachelor Thesis/data/OECD IO Tables/csv import/IO 97-11 - IO - 07.csv', sep=',', header = 6, usecols=np.arange(2,45),skiprows = np.append(np.array([7,8],dtype=float),np.array([47])))
io08 = pd.read_csv('C:/Users/Pavel/Desktop/IES/Bachelor Thesis/data/OECD IO Tables/csv import/IO 97-11 - IO - 08.csv', sep=',', header = 6, usecols=np.arange(2,45),skiprows = np.append(np.array([7,8],dtype=float),np.array([47])))
io09 = pd.read_csv('C:/Users/Pavel/Desktop/IES/Bachelor Thesis/data/OECD IO Tables/csv import/IO 97-11 - IO - 09.csv', sep=',', header = 6, usecols=np.arange(2,45),skiprows = np.append(np.array([7,8],dtype=float),np.array([47])))
io10 = pd.read_csv('C:/Users/Pavel/Desktop/IES/Bachelor Thesis/data/OECD IO Tables/csv import/IO 97-11 - IO - 10.csv', sep=',', header = 6, usecols=np.arange(2,45),skiprows = np.append(np.array([7,8],dtype=float),np.array([47])))
io11 = pd.read_csv('C:/Users/Pavel/Desktop/IES/Bachelor Thesis/data/OECD IO Tables/csv import/IO 97-11 - IO - 11.csv', sep=',', header = 6, usecols=np.arange(2,45),skiprows = np.append(np.array([7,8],dtype=float),np.array([47])))

# Importing all Employment tables from year 1998 until 2011
emp98 = pd.read_csv('C:/Users/Pavel/Desktop/IES/Bachelor Thesis/data/Employment/csv/Employment by sectors and regions - 1998.csv', sep=',', header = 4, skiprows = np.append(np.arange(5,28,dtype=float),np.array([29,35,39,47,54,60])))
emp99 = pd.read_csv('C:/Users/Pavel/Desktop/IES/Bachelor Thesis/data/Employment/csv/Employment by sectors and regions - 1999.csv', sep=',', header = 4, skiprows = np.append(np.array([35,41,45,53,60,66],dtype=float),np.arange(5,34,dtype=float)))
emp00 = pd.read_csv('C:/Users/Pavel/Desktop/IES/Bachelor Thesis/data/Employment/csv/Employment by sectors and regions - 2000.csv', sep=',', header = 4, skiprows = np.append(np.array([39,45,49,57,64,70],dtype=float),np.arange(5,38,dtype=float)))
emp01 = pd.read_csv('C:/Users/Pavel/Desktop/IES/Bachelor Thesis/data/Employment/csv/Employment by sectors and regions - 2001.csv', sep=',', header = 4, skiprows = np.append(np.array([30,36,40,48,55,61],dtype=float),np.arange(5,29,dtype=float)))
emp02 = pd.read_csv('C:/Users/Pavel/Desktop/IES/Bachelor Thesis/data/Employment/csv/Employment by sectors and regions - 2002.csv', sep=',', header = 4, skiprows = np.append(np.arange(5,26,dtype=float),np.array([27],dtype=float)))
emp03 = pd.read_csv('C:/Users/Pavel/Desktop/IES/Bachelor Thesis/data/Employment/csv/Employment by sectors and regions - 2003.csv', sep=',', header = 4, skiprows = np.append(np.array([13,19,23,31,38,44],dtype=float),np.arange(5,12,dtype=float)))
emp04 = pd.read_csv('C:/Users/Pavel/Desktop/IES/Bachelor Thesis/data/Employment/csv/Employment by sectors and regions - 2004.csv', sep=',', header = 4, skiprows = np.append(np.array([14,20,24,32,39,45],dtype=float),np.arange(5,13,dtype=float)))
emp05 = pd.read_csv('C:/Users/Pavel/Desktop/IES/Bachelor Thesis/data/Employment/csv/Employment by sectors and regions - 2005.csv', sep=',', header = 4, skiprows = np.append(np.array([16,22,26,34,41,47],dtype=float),np.arange(5,15,dtype=float)))
emp06 = pd.read_csv('C:/Users/Pavel/Desktop/IES/Bachelor Thesis/data/Employment/csv/Employment by sectors and regions - 2006.csv', sep=',', header = 4, skiprows = np.append(np.array([15,21,25,33,40,46],dtype=float),np.arange(5,14,dtype=float)))
emp07 = pd.read_csv('C:/Users/Pavel/Desktop/IES/Bachelor Thesis/data/Employment/csv/Employment by sectors and regions - 2007.csv', sep=',', header = 4, skiprows = np.append(np.array([17,23,27,35,42,48],dtype=float),np.arange(5,16,dtype=float)))
emp08 = pd.read_csv('C:/Users/Pavel/Desktop/IES/Bachelor Thesis/data/Employment/csv/Employment by sectors and regions - 2008.csv', sep=',', header = 4, skiprows = np.append(np.array([18,24,28,36,43,49],dtype=float),np.arange(5,17,dtype=float)))
emp09 = pd.read_csv('C:/Users/Pavel/Desktop/IES/Bachelor Thesis/data/Employment/csv/Employment by sectors and regions - 2009.csv', sep=',', header = 4, skiprows = np.append(np.array([19,25,29,37,44,50],dtype=float),np.arange(5,18,dtype=float)))
emp10 = pd.read_csv('C:/Users/Pavel/Desktop/IES/Bachelor Thesis/data/Employment/csv/Employment by sectors and regions - 2010.csv', sep=',', header = 4, skiprows = np.append(np.array([20,26,30,38,45,51],dtype=float),np.arange(5,19,dtype=float)))
emp11 = pd.read_csv('C:/Users/Pavel/Desktop/IES/Bachelor Thesis/data/Employment/csv/Employment by sectors and regions - 2011.csv', sep=',', header = 4, skiprows = np.append(np.array([22,28,32,40,47,53],dtype=float),np.arange(5,21,dtype=float)))
#indexing
emp98 = pd.read_csv('C:/Users/Pavel/Desktop/IES/Bachelor Thesis/data/Employment/csv/Employment by sectors and regions - 1998.csv', sep=',', header = 4, skiprows = np.append(np.arange(5,28,dtype=float),np.array([29,35,39,47,54,60]))).set_index(emp98.columns[0])
emp99 = pd.read_csv('C:/Users/Pavel/Desktop/IES/Bachelor Thesis/data/Employment/csv/Employment by sectors and regions - 1999.csv', sep=',', header = 4, skiprows = np.append(np.array([35,41,45,53,60,66],dtype=float),np.arange(5,34,dtype=float))).set_index(emp99.columns[0])
emp00 = pd.read_csv('C:/Users/Pavel/Desktop/IES/Bachelor Thesis/data/Employment/csv/Employment by sectors and regions - 2000.csv', sep=',', header = 4, skiprows = np.append(np.array([39,45,49,57,64,70],dtype=float),np.arange(5,38,dtype=float))).set_index(emp00.columns[0])
emp01 = pd.read_csv('C:/Users/Pavel/Desktop/IES/Bachelor Thesis/data/Employment/csv/Employment by sectors and regions - 2001.csv', sep=',', header = 4, skiprows = np.append(np.array([30,36,40,48,55,61],dtype=float),np.arange(5,29,dtype=float))).set_index(emp01.columns[0])
emp02 = pd.read_csv('C:/Users/Pavel/Desktop/IES/Bachelor Thesis/data/Employment/csv/Employment by sectors and regions - 2002.csv', sep=',', header = 4, skiprows = np.append(np.arange(5,26,dtype=float),np.array([27],dtype=float))).set_index(emp02.columns[0])
emp03 = pd.read_csv('C:/Users/Pavel/Desktop/IES/Bachelor Thesis/data/Employment/csv/Employment by sectors and regions - 2003.csv', sep=',', header = 4, skiprows = np.append(np.array([13,19,23,31,38,44],dtype=float),np.arange(5,12,dtype=float))).set_index(emp03.columns[0])
emp04 = pd.read_csv('C:/Users/Pavel/Desktop/IES/Bachelor Thesis/data/Employment/csv/Employment by sectors and regions - 2004.csv', sep=',', header = 4, skiprows = np.append(np.array([14,20,24,32,39,45],dtype=float),np.arange(5,13,dtype=float))).set_index(emp04.columns[0])
emp05 = pd.read_csv('C:/Users/Pavel/Desktop/IES/Bachelor Thesis/data/Employment/csv/Employment by sectors and regions - 2005.csv', sep=',', header = 4, skiprows = np.append(np.array([16,22,26,34,41,47],dtype=float),np.arange(5,15,dtype=float))).set_index(emp05.columns[0])
emp06 = pd.read_csv('C:/Users/Pavel/Desktop/IES/Bachelor Thesis/data/Employment/csv/Employment by sectors and regions - 2006.csv', sep=',', header = 4, skiprows = np.append(np.array([15,21,25,33,40,46],dtype=float),np.arange(5,14,dtype=float))).set_index(emp06.columns[0])
emp07 = pd.read_csv('C:/Users/Pavel/Desktop/IES/Bachelor Thesis/data/Employment/csv/Employment by sectors and regions - 2007.csv', sep=',', header = 4, skiprows = np.append(np.array([17,23,27,35,42,48],dtype=float),np.arange(5,16,dtype=float))).set_index(emp07.columns[0])
emp08 = pd.read_csv('C:/Users/Pavel/Desktop/IES/Bachelor Thesis/data/Employment/csv/Employment by sectors and regions - 2008.csv', sep=',', header = 4, skiprows = np.append(np.array([18,24,28,36,43,49],dtype=float),np.arange(5,17,dtype=float))).set_index(emp08.columns[0])
emp09 = pd.read_csv('C:/Users/Pavel/Desktop/IES/Bachelor Thesis/data/Employment/csv/Employment by sectors and regions - 2009.csv', sep=',', header = 4, skiprows = np.append(np.array([19,25,29,37,44,50],dtype=float),np.arange(5,18,dtype=float))).set_index(emp09.columns[0])
emp10 = pd.read_csv('C:/Users/Pavel/Desktop/IES/Bachelor Thesis/data/Employment/csv/Employment by sectors and regions - 2010.csv', sep=',', header = 4, skiprows = np.append(np.array([20,26,30,38,45,51],dtype=float),np.arange(5,19,dtype=float))).set_index(emp10.columns[0])
emp11 = pd.read_csv('C:/Users/Pavel/Desktop/IES/Bachelor Thesis/data/Employment/csv/Employment by sectors and regions - 2011.csv', sep=',', header = 4, skiprows = np.append(np.array([22,28,32,40,47,53],dtype=float),np.arange(5,21,dtype=float))).set_index(emp11.columns[0])

# Importing all distances
gansu = pd.read_csv('C:/Users/Pavel/Desktop/IES/Bachelor Thesis/data/Distances/csv/Distances between major cities in china - Gansu.csv', sep=',', usecols=np.arange(0,3), header=1)
guangdong = pd.read_csv('C:/Users/Pavel/Desktop/IES/Bachelor Thesis/data/Distances/csv/Distances between major cities in china - Guangdong.csv', sep=',', usecols=np.arange(0,3), header=1)
jiangsu = pd.read_csv('C:/Users/Pavel/Desktop/IES/Bachelor Thesis/data/Distances/csv/Distances between major cities in china - Jiangsu.csv', sep=',', usecols=np.arange(0,3), header=1)
qinghai = pd.read_csv('C:/Users/Pavel/Desktop/IES/Bachelor Thesis/data/Distances/csv/Distances between major cities in china - Qinghai.csv', sep=',', usecols=np.arange(0,3), header=1)


# Start of matrixes
m=34;	#  set the dimension (number of sectors) in the national transactions table 
n=9;		#  set the dimension (number of categories) of the final demand and final payments matrix 
k=9;	    #  set the final dimension of the regional table after aggregation  
s=33;	# set the number of sectors in the national table after elimination of non-existing sectors 


# find national transactions tables
rows = ['1','2','3','3.1','3.2','3.3','3.4','3.5','3.6','3.7','3.8','3.9','3.10','3.11','3.12','3.13','3.14','3.15','4','5','6','6.1','6.2','6.3','6.4','7','7.1','7.2','7.3','7.4','8','9','9.1','9.2']
ntio98 = pd.DataFrame(io98.iloc[0:m,0:m])
ntio98.index = rows
ntio99 = pd.DataFrame(io99.iloc[0:m,0:m])
ntio99.index = rows
ntio00 = pd.DataFrame(io00.iloc[0:m,0:m])
ntio00.index = rows
ntio01 = pd.DataFrame(io01.iloc[0:m,0:m])
ntio01.index = rows
ntio02 = pd.DataFrame(io02.iloc[0:m,0:m])
ntio02.index = rows
ntio03 = pd.DataFrame(io03.iloc[0:m,0:m])
ntio03.index = rows
ntio04 = pd.DataFrame(io04.iloc[0:m,0:m])
ntio04.index = rows
ntio05 = pd.DataFrame(io05.iloc[0:m,0:m])
ntio05.index = rows
ntio06 = pd.DataFrame(io06.iloc[0:m,0:m])
ntio06.index = rows
ntio07 = pd.DataFrame(io07.iloc[0:m,0:m])
ntio07.index = rows
ntio08 = pd.DataFrame(io08.iloc[0:m,0:m])
ntio08.index = rows
ntio09 = pd.DataFrame(io09.iloc[0:m,0:m])
ntio09.index = rows
ntio10 = pd.DataFrame(io10.iloc[0:m,0:m])
ntio10.index = rows
ntio11 = pd.DataFrame(io11.iloc[0:m,0:m])
ntio11.index = rows

# directory for io
dio = {'1':['1'],'2':['2'],'3':['3','3.1','3.2','3.3','3.4','3.5','3.6','3.7','3.8','3.9','3.10','3.11','3.12','3.13','3.14','3.15'],'4':['4'],'5':['5'],'6':['6','6.1','6.2','6.3','6.4'],'7':['7','7.1','7.2','7.3','7.4'],'8':['8'],'9':['9','9.1','9.2']}

# define function
def aggregateSectors(df,row,col,dio):
    subdf = df.loc[dio[row],dio[col]]
    return subdf.sum(axis=1).sum()

newntio98 = pd.DataFrame(0, columns = dio.keys(), index = dio.keys())
newntio99 = pd.DataFrame(0, columns = dio.keys(), index = dio.keys())
newntio00 = pd.DataFrame(0, columns = dio.keys(), index = dio.keys())
newntio01 = pd.DataFrame(0, columns = dio.keys(), index = dio.keys())
newntio02 = pd.DataFrame(0, columns = dio.keys(), index = dio.keys())
newntio03 = pd.DataFrame(0, columns = dio.keys(), index = dio.keys())
newntio04 = pd.DataFrame(0, columns = dio.keys(), index = dio.keys())
newntio05 = pd.DataFrame(0, columns = dio.keys(), index = dio.keys())
newntio06 = pd.DataFrame(0, columns = dio.keys(), index = dio.keys())
newntio07 = pd.DataFrame(0, columns = dio.keys(), index = dio.keys())
newntio08 = pd.DataFrame(0, columns = dio.keys(), index = dio.keys())
newntio09 = pd.DataFrame(0, columns = dio.keys(), index = dio.keys())
newntio10 = pd.DataFrame(0, columns = dio.keys(), index = dio.keys())
newntio11 = pd.DataFrame(0, columns = dio.keys(), index = dio.keys())

# try to loop to not make it by hand
d = {newdf : pd.DataFrame() for newdf in list_of_datasets}
list_of_datasets = [newntio98,newntio99,newntio00,newntio01,newntio02,newntio03,newntio04,newntio05,newntio06,newntio07,newntio08,newntio09,newntio10,newntio11]
list_of_ntio = [ntio98,ntio99,ntio00,ntio01,ntio02,ntio03,ntio04,ntio05,ntio06,ntio07,ntio08,ntio09,ntio10,ntio11]
storage = {}
for newdf in list_of_datasets:
    for ntio in list_of_ntio:
        for outcol in dio.keys(): 
            for incol in dio.keys():
               newdf.loc[outcol,incol] = aggregateSectors(ntio,incol,outcol,dio)
               storage(newdf) = newdf
dict(storage)      
newdf = newdf.transpose()  

#Done manually   
#year98 
for outcol in dio.keys(): 
    for incol in dio.keys():
        newntio98.loc[outcol,incol] = aggregateSectors(ntio98,incol,outcol,dio)
newntio98 = newntio98.transpose()

#year 99
for outcol in dio.keys(): 
    for incol in dio.keys():
        newntio99.loc[outcol,incol] = aggregateSectors(ntio99,incol,outcol,dio)
newntio99 = newntio99.transpose()

#year 00
for outcol in dio.keys(): 
            for incol in dio.keys():
               newntio00.loc[outcol,incol] = aggregateSectors(ntio00,incol,outcol,dio)
newntio00 = newntio00.transpose()

#year 01
for outcol in dio.keys(): 
            for incol in dio.keys():
               newntio01.loc[outcol,incol] = aggregateSectors(ntio01,incol,outcol,dio)
newntio01 = newntio01.transpose()

#year 02
for outcol in dio.keys(): 
            for incol in dio.keys():
               newntio02.loc[outcol,incol] = aggregateSectors(ntio02,incol,outcol,dio)
newntio02 = newntio02.transpose()

#year 03
for outcol in dio.keys(): 
            for incol in dio.keys():
               newntio03.loc[outcol,incol] = aggregateSectors(ntio03,incol,outcol,dio)
newntio03 = newntio03.transpose()

#year 04
for outcol in dio.keys(): 
            for incol in dio.keys():
               newntio04.loc[outcol,incol] = aggregateSectors(ntio04,incol,outcol,dio)
newntio04 = newntio04.transpose()

#year 05
for outcol in dio.keys(): 
            for incol in dio.keys():
               newntio05.loc[outcol,incol] = aggregateSectors(ntio05,incol,outcol,dio)
newntio05 = newntio05.transpose()

#year 06
for outcol in dio.keys(): 
            for incol in dio.keys():
               newntio06.loc[outcol,incol] = aggregateSectors(ntio06,incol,outcol,dio)
newntio06 = newntio06.transpose()

#year 07
for outcol in dio.keys(): 
            for incol in dio.keys():
               newntio07.loc[outcol,incol] = aggregateSectors(ntio07,incol,outcol,dio)
newntio07 = newntio07.transpose()

#year 08
for outcol in dio.keys(): 
            for incol in dio.keys():
               newntio08.loc[outcol,incol] = aggregateSectors(ntio08,incol,outcol,dio)
newntio08 = newntio08.transpose()

#year 09
for outcol in dio.keys(): 
            for incol in dio.keys():
               newntio09.loc[outcol,incol] = aggregateSectors(ntio09,incol,outcol,dio)
newntio09 = newntio09.transpose()

#year 10
for outcol in dio.keys(): 
            for incol in dio.keys():
               newntio10.loc[outcol,incol] = aggregateSectors(ntio10,incol,outcol,dio)
newntio10 = newntio10.transpose()

#year 11
for outcol in dio.keys(): 
            for incol in dio.keys():
               newntio11.loc[outcol,incol] = aggregateSectors(ntio11,incol,outcol,dio)
newntio11 = newntio11.transpose()

# directory for emp, we need to keep indexes
demp9802 = {'Total':['Total'], '1':['1'],'2':['2'],'3':['3'],'4':['4'],'5':['5'],'6':['6','6.1'],'7':['7','7.1','7.2'],'8':['8','8.1','8.2'],'9':['9','9.1','9.2']}
demp0311 = {'Total':['Total'], '1':['1'],'2':['2'],'3':['3'],'4':['4'],'5':['5'],'6':['6','6.1'],'7':['7','7.1','7.2','7.3','7.4'],'8':['8','8.1','8.2','8.3','8.4'],'9':['9','9.1']}
dempi98 = {'1998':['1998'], 'Beijing':['Beijing'], 'Tianjin':['Tianjin'], 'Hebei':['Hebei'], 'Shanxi':['Shanxi'], 'Inner Mongolia':['Inner Mongolia'],'Liaoning':['Liaoning'], 'Jilin':['Jilin'], 'Heilongjiang':['Heilongjiang'], 'Shanghai':['Shanghai'], 'Jiangsu':['Jiangsu'],'Zhejiang':['Zhejiang'], 'Anhui':['Anhui'], 'Fujian':['Fujian'], 'Jiangxi':['Jiangxi'], 'Shandong':['Shandong'], 'Henan':['Henan'],'Hubei':['Hubei'], 'Hunan':['Hunan'], 'Guangdong':['Guangdong'], 'Guangxi':['Guangxi'], 'Hainan':['Hainan'], 'Chongqing':['Chongqing'],'Sichuan':['Sichuan'], 'Guizhou':['Guizhou'], 'Yunnan':['Yunnan'], 'Tibet':['Tibet'], 'Shaanxi':['Shaanxi'], 'Gansu':['Gansu'],'Qinghai':['Qinghai'], 'Ningxia':['Ningxia'], 'Xinjiang':['Xinjiang']}
dempi99 = {'1999':['1999'], 'Beijing':['Beijing'], 'Tianjin':['Tianjin'], 'Hebei':['Hebei'], 'Shanxi':['Shanxi'], 'Inner Mongolia':['Inner Mongolia'],'Liaoning':['Liaoning'], 'Jilin':['Jilin'], 'Heilongjiang':['Heilongjiang'], 'Shanghai':['Shanghai'], 'Jiangsu':['Jiangsu'],'Zhejiang':['Zhejiang'], 'Anhui':['Anhui'], 'Fujian':['Fujian'], 'Jiangxi':['Jiangxi'], 'Shandong':['Shandong'], 'Henan':['Henan'],'Hubei':['Hubei'], 'Hunan':['Hunan'], 'Guangdong':['Guangdong'], 'Guangxi':['Guangxi'], 'Hainan':['Hainan'], 'Chongqing':['Chongqing'],'Sichuan':['Sichuan'], 'Guizhou':['Guizhou'], 'Yunnan':['Yunnan'], 'Tibet':['Tibet'], 'Shaanxi':['Shaanxi'], 'Gansu':['Gansu'],'Qinghai':['Qinghai'], 'Ningxia':['Ningxia'], 'Xinjiang':['Xinjiang']}
dempi00 = {'2000':['2000'], 'Beijing':['Beijing'], 'Tianjin':['Tianjin'], 'Hebei':['Hebei'], 'Shanxi':['Shanxi'], 'Inner Mongolia':['Inner Mongolia'],'Liaoning':['Liaoning'], 'Jilin':['Jilin'], 'Heilongjiang':['Heilongjiang'], 'Shanghai':['Shanghai'], 'Jiangsu':['Jiangsu'],'Zhejiang':['Zhejiang'], 'Anhui':['Anhui'], 'Fujian':['Fujian'], 'Jiangxi':['Jiangxi'], 'Shandong':['Shandong'], 'Henan':['Henan'],'Hubei':['Hubei'], 'Hunan':['Hunan'], 'Guangdong':['Guangdong'], 'Guangxi':['Guangxi'], 'Hainan':['Hainan'], 'Chongqing':['Chongqing'],'Sichuan':['Sichuan'], 'Guizhou':['Guizhou'], 'Yunnan':['Yunnan'], 'Tibet':['Tibet'], 'Shaanxi':['Shaanxi'], 'Gansu':['Gansu'],'Qinghai':['Qinghai'], 'Ningxia':['Ningxia'], 'Xinjiang':['Xinjiang']}
dempi01 = {'2001':['2001'], 'Beijing':['Beijing'], 'Tianjin':['Tianjin'], 'Hebei':['Hebei'], 'Shanxi':['Shanxi'], 'Inner Mongolia':['Inner Mongolia'],'Liaoning':['Liaoning'], 'Jilin':['Jilin'], 'Heilongjiang':['Heilongjiang'], 'Shanghai':['Shanghai'], 'Jiangsu':['Jiangsu'],'Zhejiang':['Zhejiang'], 'Anhui':['Anhui'], 'Fujian':['Fujian'], 'Jiangxi':['Jiangxi'], 'Shandong':['Shandong'], 'Henan':['Henan'],'Hubei':['Hubei'], 'Hunan':['Hunan'], 'Guangdong':['Guangdong'], 'Guangxi':['Guangxi'], 'Hainan':['Hainan'], 'Chongqing':['Chongqing'],'Sichuan':['Sichuan'], 'Guizhou':['Guizhou'], 'Yunnan':['Yunnan'], 'Tibet':['Tibet'], 'Shaanxi':['Shaanxi'], 'Gansu':['Gansu'],'Qinghai':['Qinghai'], 'Ningxia':['Ningxia'], 'Xinjiang':['Xinjiang']}
dempi02 = {'2002':['2002'], 'Beijing':['Beijing'], 'Tianjin':['Tianjin'], 'Hebei':['Hebei'], 'Shanxi':['Shanxi'], 'Inner Mongolia':['Inner Mongolia'],'Liaoning':['Liaoning'], 'Jilin':['Jilin'], 'Heilongjiang':['Heilongjiang'], 'Shanghai':['Shanghai'], 'Jiangsu':['Jiangsu'],'Zhejiang':['Zhejiang'], 'Anhui':['Anhui'], 'Fujian':['Fujian'], 'Jiangxi':['Jiangxi'], 'Shandong':['Shandong'], 'Henan':['Henan'],'Hubei':['Hubei'], 'Hunan':['Hunan'], 'Guangdong':['Guangdong'], 'Guangxi':['Guangxi'], 'Hainan':['Hainan'], 'Chongqing':['Chongqing'],'Sichuan':['Sichuan'], 'Guizhou':['Guizhou'], 'Yunnan':['Yunnan'], 'Tibet':['Tibet'], 'Shaanxi':['Shaanxi'], 'Gansu':['Gansu'],'Qinghai':['Qinghai'], 'Ningxia':['Ningxia'], 'Xinjiang':['Xinjiang']}
dempi03 = {'2003':['2003'], 'Beijing':['Beijing'], 'Tianjin':['Tianjin'], 'Hebei':['Hebei'], 'Shanxi':['Shanxi'], 'Inner Mongolia':['Inner Mongolia'],'Liaoning':['Liaoning'], 'Jilin':['Jilin'], 'Heilongjiang':['Heilongjiang'], 'Shanghai':['Shanghai'], 'Jiangsu':['Jiangsu'],'Zhejiang':['Zhejiang'], 'Anhui':['Anhui'], 'Fujian':['Fujian'], 'Jiangxi':['Jiangxi'], 'Shandong':['Shandong'], 'Henan':['Henan'],'Hubei':['Hubei'], 'Hunan':['Hunan'], 'Guangdong':['Guangdong'], 'Guangxi':['Guangxi'], 'Hainan':['Hainan'], 'Chongqing':['Chongqing'],'Sichuan':['Sichuan'], 'Guizhou':['Guizhou'], 'Yunnan':['Yunnan'], 'Tibet':['Tibet'], 'Shaanxi':['Shaanxi'], 'Gansu':['Gansu'],'Qinghai':['Qinghai'], 'Ningxia':['Ningxia'], 'Xinjiang':['Xinjiang']}
dempi04 = {'2004':['2004'], 'Beijing':['Beijing'], 'Tianjin':['Tianjin'], 'Hebei':['Hebei'], 'Shanxi':['Shanxi'], 'Inner Mongolia':['Inner Mongolia'],'Liaoning':['Liaoning'], 'Jilin':['Jilin'], 'Heilongjiang':['Heilongjiang'], 'Shanghai':['Shanghai'], 'Jiangsu':['Jiangsu'],'Zhejiang':['Zhejiang'], 'Anhui':['Anhui'], 'Fujian':['Fujian'], 'Jiangxi':['Jiangxi'], 'Shandong':['Shandong'], 'Henan':['Henan'],'Hubei':['Hubei'], 'Hunan':['Hunan'], 'Guangdong':['Guangdong'], 'Guangxi':['Guangxi'], 'Hainan':['Hainan'], 'Chongqing':['Chongqing'],'Sichuan':['Sichuan'], 'Guizhou':['Guizhou'], 'Yunnan':['Yunnan'], 'Tibet':['Tibet'], 'Shaanxi':['Shaanxi'], 'Gansu':['Gansu'],'Qinghai':['Qinghai'], 'Ningxia':['Ningxia'], 'Xinjiang':['Xinjiang']}
dempi05 = {'2005':['2005'], 'Beijing':['Beijing'], 'Tianjin':['Tianjin'], 'Hebei':['Hebei'], 'Shanxi':['Shanxi'], 'Inner Mongolia':['Inner Mongolia'],'Liaoning':['Liaoning'], 'Jilin':['Jilin'], 'Heilongjiang':['Heilongjiang'], 'Shanghai':['Shanghai'], 'Jiangsu':['Jiangsu'],'Zhejiang':['Zhejiang'], 'Anhui':['Anhui'], 'Fujian':['Fujian'], 'Jiangxi':['Jiangxi'], 'Shandong':['Shandong'], 'Henan':['Henan'],'Hubei':['Hubei'], 'Hunan':['Hunan'], 'Guangdong':['Guangdong'], 'Guangxi':['Guangxi'], 'Hainan':['Hainan'], 'Chongqing':['Chongqing'],'Sichuan':['Sichuan'], 'Guizhou':['Guizhou'], 'Yunnan':['Yunnan'], 'Tibet':['Tibet'], 'Shaanxi':['Shaanxi'], 'Gansu':['Gansu'],'Qinghai':['Qinghai'], 'Ningxia':['Ningxia'], 'Xinjiang':['Xinjiang']}
dempi06 = {'2006':['2006'], 'Beijing':['Beijing'], 'Tianjin':['Tianjin'], 'Hebei':['Hebei'], 'Shanxi':['Shanxi'], 'Inner Mongolia':['Inner Mongolia'],'Liaoning':['Liaoning'], 'Jilin':['Jilin'], 'Heilongjiang':['Heilongjiang'], 'Shanghai':['Shanghai'], 'Jiangsu':['Jiangsu'],'Zhejiang':['Zhejiang'], 'Anhui':['Anhui'], 'Fujian':['Fujian'], 'Jiangxi':['Jiangxi'], 'Shandong':['Shandong'], 'Henan':['Henan'],'Hubei':['Hubei'], 'Hunan':['Hunan'], 'Guangdong':['Guangdong'], 'Guangxi':['Guangxi'], 'Hainan':['Hainan'], 'Chongqing':['Chongqing'],'Sichuan':['Sichuan'], 'Guizhou':['Guizhou'], 'Yunnan':['Yunnan'], 'Tibet':['Tibet'], 'Shaanxi':['Shaanxi'], 'Gansu':['Gansu'],'Qinghai':['Qinghai'], 'Ningxia':['Ningxia'], 'Xinjiang':['Xinjiang']}
dempi07 = {'2007':['2007'], 'Beijing':['Beijing'], 'Tianjin':['Tianjin'], 'Hebei':['Hebei'], 'Shanxi':['Shanxi'], 'Inner Mongolia':['Inner Mongolia'],'Liaoning':['Liaoning'], 'Jilin':['Jilin'], 'Heilongjiang':['Heilongjiang'], 'Shanghai':['Shanghai'], 'Jiangsu':['Jiangsu'],'Zhejiang':['Zhejiang'], 'Anhui':['Anhui'], 'Fujian':['Fujian'], 'Jiangxi':['Jiangxi'], 'Shandong':['Shandong'], 'Henan':['Henan'],'Hubei':['Hubei'], 'Hunan':['Hunan'], 'Guangdong':['Guangdong'], 'Guangxi':['Guangxi'], 'Hainan':['Hainan'], 'Chongqing':['Chongqing'],'Sichuan':['Sichuan'], 'Guizhou':['Guizhou'], 'Yunnan':['Yunnan'], 'Tibet':['Tibet'], 'Shaanxi':['Shaanxi'], 'Gansu':['Gansu'],'Qinghai':['Qinghai'], 'Ningxia':['Ningxia'], 'Xinjiang':['Xinjiang']}
dempi08 = {'2008':['2008'], 'Beijing':['Beijing'], 'Tianjin':['Tianjin'], 'Hebei':['Hebei'], 'Shanxi':['Shanxi'], 'Inner Mongolia':['Inner Mongolia'],'Liaoning':['Liaoning'], 'Jilin':['Jilin'], 'Heilongjiang':['Heilongjiang'], 'Shanghai':['Shanghai'], 'Jiangsu':['Jiangsu'],'Zhejiang':['Zhejiang'], 'Anhui':['Anhui'], 'Fujian':['Fujian'], 'Jiangxi':['Jiangxi'], 'Shandong':['Shandong'], 'Henan':['Henan'],'Hubei':['Hubei'], 'Hunan':['Hunan'], 'Guangdong':['Guangdong'], 'Guangxi':['Guangxi'], 'Hainan':['Hainan'], 'Chongqing':['Chongqing'],'Sichuan':['Sichuan'], 'Guizhou':['Guizhou'], 'Yunnan':['Yunnan'], 'Tibet':['Tibet'], 'Shaanxi':['Shaanxi'], 'Gansu':['Gansu'],'Qinghai':['Qinghai'], 'Ningxia':['Ningxia'], 'Xinjiang':['Xinjiang']}
dempi09 = {'2009':['2009'], 'Beijing':['Beijing'], 'Tianjin':['Tianjin'], 'Hebei':['Hebei'], 'Shanxi':['Shanxi'], 'Inner Mongolia':['Inner Mongolia'],'Liaoning':['Liaoning'], 'Jilin':['Jilin'], 'Heilongjiang':['Heilongjiang'], 'Shanghai':['Shanghai'], 'Jiangsu':['Jiangsu'],'Zhejiang':['Zhejiang'], 'Anhui':['Anhui'], 'Fujian':['Fujian'], 'Jiangxi':['Jiangxi'], 'Shandong':['Shandong'], 'Henan':['Henan'],'Hubei':['Hubei'], 'Hunan':['Hunan'], 'Guangdong':['Guangdong'], 'Guangxi':['Guangxi'], 'Hainan':['Hainan'], 'Chongqing':['Chongqing'],'Sichuan':['Sichuan'], 'Guizhou':['Guizhou'], 'Yunnan':['Yunnan'], 'Tibet':['Tibet'], 'Shaanxi':['Shaanxi'], 'Gansu':['Gansu'],'Qinghai':['Qinghai'], 'Ningxia':['Ningxia'], 'Xinjiang':['Xinjiang']}
dempi10 = {'2010':['2010'], 'Beijing':['Beijing'], 'Tianjin':['Tianjin'], 'Hebei':['Hebei'], 'Shanxi':['Shanxi'], 'Inner Mongolia':['Inner Mongolia'],'Liaoning':['Liaoning'], 'Jilin':['Jilin'], 'Heilongjiang':['Heilongjiang'], 'Shanghai':['Shanghai'], 'Jiangsu':['Jiangsu'],'Zhejiang':['Zhejiang'], 'Anhui':['Anhui'], 'Fujian':['Fujian'], 'Jiangxi':['Jiangxi'], 'Shandong':['Shandong'], 'Henan':['Henan'],'Hubei':['Hubei'], 'Hunan':['Hunan'], 'Guangdong':['Guangdong'], 'Guangxi':['Guangxi'], 'Hainan':['Hainan'], 'Chongqing':['Chongqing'],'Sichuan':['Sichuan'], 'Guizhou':['Guizhou'], 'Yunnan':['Yunnan'], 'Tibet':['Tibet'], 'Shaanxi':['Shaanxi'], 'Gansu':['Gansu'],'Qinghai':['Qinghai'], 'Ningxia':['Ningxia'], 'Xinjiang':['Xinjiang']}
dempi11 = {'2011':['2011'], 'Beijing':['Beijing'], 'Tianjin':['Tianjin'], 'Hebei':['Hebei'], 'Shanxi':['Shanxi'], 'Inner Mongolia':['Inner Mongolia'],'Liaoning':['Liaoning'], 'Jilin':['Jilin'], 'Heilongjiang':['Heilongjiang'], 'Shanghai':['Shanghai'], 'Jiangsu':['Jiangsu'],'Zhejiang':['Zhejiang'], 'Anhui':['Anhui'], 'Fujian':['Fujian'], 'Jiangxi':['Jiangxi'], 'Shandong':['Shandong'], 'Henan':['Henan'],'Hubei':['Hubei'], 'Hunan':['Hunan'], 'Guangdong':['Guangdong'], 'Guangxi':['Guangxi'], 'Hainan':['Hainan'], 'Chongqing':['Chongqing'],'Sichuan':['Sichuan'], 'Guizhou':['Guizhou'], 'Yunnan':['Yunnan'], 'Tibet':['Tibet'], 'Shaanxi':['Shaanxi'], 'Gansu':['Gansu'],'Qinghai':['Qinghai'], 'Ningxia':['Ningxia'], 'Xinjiang':['Xinjiang']}

#define functions
def aggregateEmp1(df,row,col,demp9802,dempi98):
    subdf1 = df.loc[dempi98[row],demp9802[col]]
    return subdf1.sum(axis=1).sum()

def aggregateEmp2(df,row,col,demp9802,dempi99):
    subdf2 = df.loc[dempi99[row],demp9802[col]]
    return subdf2.sum(axis=1).sum()

def aggregateEmp3(df,row,col,demp9802,dempi00):
    subdf3 = df.loc[dempi00[row],demp9802[col]]
    return subdf3.sum(axis=1).sum()

def aggregateEmp4(df,row,col,demp9802,dempi01):
    subdf4 = df.loc[dempi01[row],demp9802[col]]
    return subdf4.sum(axis=1).sum()

def aggregateEmp5(df,row,col,demp9802,dempi02):
    subdf5 = df.loc[dempi02[row],demp9802[col]]
    return subdf5.sum(axis=1).sum()

def aggregateEmp6(df,row,col,demp0311,dempi03):
    subdf6 = df.loc[dempi03[row],demp0311[col]]
    return subdf6.sum(axis=1).sum()

def aggregateEmp7(df,row,col,demp0311,dempi04):
    subdf7 = df.loc[dempi04[row],demp0311[col]]
    return subdf7.sum(axis=1).sum()

def aggregateEmp8(df,row,col,demp0311,dempi05):
    subdf8 = df.loc[dempi05[row],demp0311[col]]
    return subdf8.sum(axis=1).sum()

def aggregateEmp9(df,row,col,demp0311,dempi06):
    subdf9 = df.loc[dempi06[row],demp0311[col]]
    return subdf9.sum(axis=1).sum()

def aggregateEmp10(df,row,col,demp0311,dempi07):
    subdf10 = df.loc[dempi07[row],demp0311[col]]
    return subdf10.sum(axis=1).sum()

def aggregateEmp11(df,row,col,demp0311,dempi08):
    subdf11 = df.loc[dempi08[row],demp0311[col]]
    return subdf11.sum(axis=1).sum()

def aggregateEmp12(df,row,col,demp0311,dempi09):
    subdf12 = df.loc[dempi09[row],demp0311[col]]
    return subdf12.sum(axis=1).sum()

def aggregateEmp13(df,row,col,demp0311,dempi10):
    subdf13 = df.loc[dempi10[row],demp0311[col]]
    return subdf13.sum(axis=1).sum()

def aggregateEmp14(df,row,col,demp0311,dempi11):
    subdf14 = df.loc[dempi11[row],demp0311[col]]
    return subdf14.sum(axis=1).sum()

#define newemps
newemp98 = pd.DataFrame(0, columns=demp9802.keys(), index = dempi98.keys())
newemp99 = pd.DataFrame(0, columns=demp9802.keys(), index = dempi99.keys())
newemp00 = pd.DataFrame(0, columns=demp9802.keys(), index = dempi00.keys())
newemp01 = pd.DataFrame(0, columns=demp9802.keys(), index = dempi01.keys())
newemp02 = pd.DataFrame(0, columns=demp9802.keys(), index = dempi02.keys())
newemp03 = pd.DataFrame(0, columns=demp0311.keys(), index = dempi03.keys())
newemp04 = pd.DataFrame(0, columns=demp0311.keys(), index = dempi04.keys())
newemp05 = pd.DataFrame(0, columns=demp0311.keys(), index = dempi05.keys())
newemp06 = pd.DataFrame(0, columns=demp0311.keys(), index = dempi06.keys())
newemp07 = pd.DataFrame(0, columns=demp0311.keys(), index = dempi07.keys())
newemp08 = pd.DataFrame(0, columns=demp0311.keys(), index = dempi08.keys())
newemp09 = pd.DataFrame(0, columns=demp0311.keys(), index = dempi09.keys())
newemp10 = pd.DataFrame(0, columns=demp0311.keys(), index = dempi10.keys())
newemp11 = pd.DataFrame(0, columns=demp0311.keys(), index = dempi11.keys())

#define loops
for outcol in demp9802.keys(): 
            for incol in dempi98.keys():
               newemp98.loc[incol,outcol] = aggregateEmp1(emp98,incol,outcol,demp9802,dempi98)

for outcol in demp9802.keys(): 
            for incol in dempi99.keys():
               newemp99.loc[incol,outcol] = aggregateEmp2(emp99,incol,outcol,demp9802,dempi99)

for outcol in demp9802.keys(): 
            for incol in dempi00.keys():
               newemp00.loc[incol,outcol] = aggregateEmp3(emp00,incol,outcol,demp9802,dempi00)

for outcol in demp9802.keys(): 
            for incol in dempi01.keys():
               newemp01.loc[incol,outcol] = aggregateEmp4(emp01,incol,outcol,demp9802,dempi01)

for outcol in demp9802.keys(): 
            for incol in dempi02.keys():
               newemp02.loc[incol,outcol] = aggregateEmp5(emp02,incol,outcol,demp9802,dempi02)

for outcol in demp0311.keys(): 
            for incol in dempi03.keys():
               newemp03.loc[incol,outcol] = aggregateEmp6(emp03,incol,outcol,demp0311,dempi03)

for outcol in demp0311.keys(): 
            for incol in dempi04.keys():
               newemp04.loc[incol,outcol] = aggregateEmp7(emp04,incol,outcol,demp0311,dempi04)

for outcol in demp0311.keys(): 
            for incol in dempi05.keys():
               newemp05.loc[incol,outcol] = aggregateEmp8(emp05,incol,outcol,demp0311,dempi05)

for outcol in demp0311.keys(): 
            for incol in dempi06.keys():
               newemp06.loc[incol,outcol] = aggregateEmp9(emp06,incol,outcol,demp0311,dempi06)

for outcol in demp0311.keys(): 
            for incol in dempi07.keys():
               newemp07.loc[incol,outcol] = aggregateEmp10(emp07,incol,outcol,demp0311,dempi07)

for outcol in demp0311.keys(): 
            for incol in dempi08.keys():
               newemp08.loc[incol,outcol] = aggregateEmp11(emp08,incol,outcol,demp0311,dempi08)

for outcol in demp0311.keys(): 
            for incol in dempi09.keys():
               newemp09.loc[incol,outcol] = aggregateEmp12(emp09,incol,outcol,demp0311,dempi09)

for outcol in demp0311.keys(): 
            for incol in dempi10.keys():
               newemp10.loc[incol,outcol] = aggregateEmp13(emp10,incol,outcol,demp0311,dempi10)

for outcol in demp0311.keys(): 
            for incol in dempi11.keys():
               newemp11.loc[incol,outcol] = aggregateEmp14(emp11,incol,outcol,demp0311,dempi11)

#directory for final consumption
dfc = {'10':['10','10.1','10.2','10.3','10.4'],'12':['12','12.1'],'13':['13','13.1']}
rows2 = rows.copy()
rows2.extend(['11','11','11','14'])
drows2 = {'1':['1'],'2':['2'],'3':['3','3.1','3.2','3.3','3.4','3.5','3.6','3.7','3.8','3.9','3.10','3.11','3.12','3.13','3.14','3.15'],'4':['4'],'5':['5'],'6':['6','6.1','6.2','6.3','6.4'],'7':['7','7.1','7.2','7.3','7.4'],'8':['8'],'9':['9','9.1','9.2']}


#final consumption ios98
fcio98 = pd.DataFrame(io98.loc[:,['10','10.1','10.2','10.3','10.4','12','12.1','13','13.1']])
fcio98.index = rows2

#define functions
def aggregatefcio(df,row,col,dfc,drows2):
    subdf = df.loc[drows2[row],dfc[col]]
    return subdf.sum(axis=1).sum()

#define newfci
newfci98 = pd.DataFrame(0, columns=dfc.keys(), index = drows2.keys())

for outcol in dfc.keys(): 
            for incol in drows2.keys():
               newfci98.loc[incol,outcol] = aggregatefcio(fcio98,incol,outcol,dfc,drows2)

#final consumption ios99               
fcio99 = pd.DataFrame(io99.loc[:,['10','10.1','10.2','10.3','10.4','12','12.1','13','13.1']])
fcio99.index = rows2

#define newfci
newfci99 = pd.DataFrame(0, columns=dfc.keys(), index = drows2.keys())

for outcol in dfc.keys(): 
            for incol in drows2.keys():
               newfci99.loc[incol,outcol] = aggregatefcio(fcio99,incol,outcol,dfc,drows2)
               
#final consumption ios00              
fcio00 = pd.DataFrame(io00.loc[:,['10','10.1','10.2','10.3','10.4','12','12.1','13','13.1']])
fcio00.index = rows2

#define newfci
newfci00 = pd.DataFrame(0, columns=dfc.keys(), index = drows2.keys())

for outcol in dfc.keys(): 
            for incol in drows2.keys():
               newfci00.loc[incol,outcol] = aggregatefcio(fcio00,incol,outcol,dfc,drows2)
       
#final consumption ios01            
fcio01 = pd.DataFrame(io01.loc[:,['10','10.1','10.2','10.3','10.4','12','12.1','13','13.1']])
fcio01.index = rows2

#define newfci
newfci01 = pd.DataFrame(0, columns=dfc.keys(), index = drows2.keys())

for outcol in dfc.keys(): 
            for incol in drows2.keys():
               newfci01.loc[incol,outcol] = aggregatefcio(fcio01,incol,outcol,dfc,drows2)
               
#final consumption ios02            
fcio02 = pd.DataFrame(io02.loc[:,['10','10.1','10.2','10.3','10.4','12','12.1','13','13.1']])
fcio02.index = rows2

#define newfci
newfci02 = pd.DataFrame(0, columns=dfc.keys(), index = drows2.keys())

for outcol in dfc.keys(): 
            for incol in drows2.keys():
               newfci02.loc[incol,outcol] = aggregatefcio(fcio02,incol,outcol,dfc,drows2)
               
#final consumption ios03            
fcio03 = pd.DataFrame(io03.loc[:,['10','10.1','10.2','10.3','10.4','12','12.1','13','13.1']])
fcio03.index = rows2

#define newfci
newfci03 = pd.DataFrame(0, columns=dfc.keys(), index = drows2.keys())

for outcol in dfc.keys(): 
            for incol in drows2.keys():
               newfci03.loc[incol,outcol] = aggregatefcio(fcio03,incol,outcol,dfc,drows2)
      
#final consumption ios04
fcio04 = pd.DataFrame(io04.loc[:,['10','10.1','10.2','10.3','10.4','12','12.1','13','13.1']])
fcio04.index = rows2

#define newfci
newfci04 = pd.DataFrame(0, columns=dfc.keys(), index = drows2.keys())

for outcol in dfc.keys(): 
            for incol in drows2.keys():
               newfci04.loc[incol,outcol] = aggregatefcio(fcio04,incol,outcol,dfc,drows2)

#final consumption ios05            
fcio05 = pd.DataFrame(io05.loc[:,['10','10.1','10.2','10.3','10.4','12','12.1','13','13.1']])
fcio05.index = rows2

#define newfci
newfci05 = pd.DataFrame(0, columns=dfc.keys(), index = drows2.keys())

for outcol in dfc.keys(): 
            for incol in drows2.keys():
               newfci05.loc[incol,outcol] = aggregatefcio(fcio05,incol,outcol,dfc,drows2)

#final consumption ios06            
fcio06 = pd.DataFrame(io06.loc[:,['10','10.1','10.2','10.3','10.4','12','12.1','13','13.1']])
fcio06.index = rows2

#define newfci
newfci06 = pd.DataFrame(0, columns=dfc.keys(), index = drows2.keys())

for outcol in dfc.keys(): 
            for incol in drows2.keys():
               newfci06.loc[incol,outcol] = aggregatefcio(fcio06,incol,outcol,dfc,drows2)

#final consumption ios07            
fcio07 = pd.DataFrame(io07.loc[:,['10','10.1','10.2','10.3','10.4','12','12.1','13','13.1']])
fcio07.index = rows2

#define newfci
newfci07 = pd.DataFrame(0, columns=dfc.keys(), index = drows2.keys())

for outcol in dfc.keys(): 
            for incol in drows2.keys():
               newfci07.loc[incol,outcol] = aggregatefcio(fcio07,incol,outcol,dfc,drows2)

#final consumption ios08           
fcio08 = pd.DataFrame(io08.loc[:,['10','10.1','10.2','10.3','10.4','12','12.1','13','13.1']])
fcio08.index = rows2

#define newfci
newfci08 = pd.DataFrame(0, columns=dfc.keys(), index = drows2.keys())

for outcol in dfc.keys(): 
            for incol in drows2.keys():
               newfci08.loc[incol,outcol] = aggregatefcio(fcio08,incol,outcol,dfc,drows2)

#final consumption ios09           
fcio09 = pd.DataFrame(io09.loc[:,['10','10.1','10.2','10.3','10.4','12','12.1','13','13.1']])
fcio09.index = rows2

#define newfci
newfci09 = pd.DataFrame(0, columns=dfc.keys(), index = drows2.keys())

for outcol in dfc.keys(): 
            for incol in drows2.keys():
               newfci09.loc[incol,outcol] = aggregatefcio(fcio09,incol,outcol,dfc,drows2)

#final consumption ios10           
fcio10 = pd.DataFrame(io10.loc[:,['10','10.1','10.2','10.3','10.4','12','12.1','13','13.1']])
fcio10.index = rows2

#define newfci
newfci10 = pd.DataFrame(0, columns=dfc.keys(), index = drows2.keys())

for outcol in dfc.keys(): 
            for incol in drows2.keys():
               newfci10.loc[incol,outcol] = aggregatefcio(fcio10,incol,outcol,dfc,drows2)

#final consumption ios11
fcio11 = pd.DataFrame(io11.loc[:,['10','10.1','10.2','10.3','10.4','12','12.1','13','13.1']])
fcio11.index = rows2

#define newfci
newfci11 = pd.DataFrame(0, columns=dfc.keys(), index = drows2.keys())

for outcol in dfc.keys(): 
            for incol in drows2.keys():
               newfci11.loc[incol,outcol] = aggregatefcio(fcio11,incol,outcol,dfc,drows2)

#divide all employments
div_emp98 = newemp98.loc[:,"1":"9"].div(newemp98["Total"], axis = 0)
SLQ_emp98 = div_emp98.loc["Beijing":,:].div(div_emp98.loc["1998"][:])

div_emp99 = newemp99.loc[:,"1":"9"].div(newemp99["Total"], axis = 0)
SLQ_emp99 = div_emp99.loc["Beijing":,:].div(div_emp99.loc["1999"][:])

div_emp00 = newemp00.loc[:,"1":"9"].div(newemp00["Total"], axis = 0)
SLQ_emp00 = div_emp00.loc["Beijing":,:].div(div_emp00.loc["2000"][:])

div_emp01 = newemp01.loc[:,"1":"9"].div(newemp01["Total"], axis = 0)
SLQ_emp01 = div_emp01.loc["Beijing":,:].div(div_emp01.loc["2001"][:])

div_emp02 = newemp02.loc[:,"1":"9"].div(newemp02["Total"], axis = 0)
SLQ_emp02 = div_emp02.loc["Beijing":,:].div(div_emp02.loc["2002"][:])

div_emp03 = newemp03.loc[:,"1":"9"].div(newemp03["Total"], axis = 0)
SLQ_emp03 = div_emp03.loc["Beijing":,:].div(div_emp03.loc["2003"][:])

div_emp04 = newemp04.loc[:,"1":"9"].div(newemp04["Total"], axis = 0)
SLQ_emp04 = div_emp04.loc["Beijing":,:].div(div_emp04.loc["2004"][:])

div_emp05 = newemp05.loc[:,"1":"9"].div(newemp05["Total"], axis = 0)
SLQ_emp05 = div_emp05.loc["Beijing":,:].div(div_emp05.loc["2005"][:])

div_emp06 = newemp06.loc[:,"1":"9"].div(newemp06["Total"], axis = 0)
SLQ_emp06 = div_emp06.loc["Beijing":,:].div(div_emp06.loc["2006"][:])

div_emp07 = newemp07.loc[:,"1":"9"].div(newemp07["Total"], axis = 0)
SLQ_emp07 = div_emp07.loc["Beijing":,:].div(div_emp07.loc["2007"][:])

div_emp08 = newemp08.loc[:,"1":"9"].div(newemp08["Total"], axis = 0)
SLQ_emp08 = div_emp08.loc["Beijing":,:].div(div_emp08.loc["2008"][:])

div_emp09 = newemp09.loc[:,"1":"9"].div(newemp09["Total"], axis = 0)
SLQ_emp09 = div_emp09.loc["Beijing":,:].div(div_emp09.loc["2009"][:])

div_emp10 = newemp10.loc[:,"1":"9"].div(newemp10["Total"], axis = 0)
SLQ_emp10 = div_emp10.loc["Beijing":,:].div(div_emp10.loc["2010"][:])

div_emp11 = newemp11.loc[:,"1":"9"].div(newemp11["Total"], axis = 0)
SLQ_emp11 = div_emp11.loc["Beijing":,:].div(div_emp11.loc["2011"][:])

#CILQ
#CILQ_emp98_Beijing = pd.DataFrame(0, columns = list(SLQ_emp98.columns.values), index = list(SLQ_emp98.columns.values))
de98B = {'1':['1'],'2':['2'],'3':['3'],'4':['4'],'5':['5'],'6':['6'],'7':['7'],'8':['8'],'9':['9']}
deB = {'Beijing':['Beijing']}

#def aggregateCILQ(df,row,col,de98B,deB):
 #   subdf = df.loc[deB[row],de98B[col]].div(df.loc['Beijing']['1':'9'],axis=1)
  #  return subdf

#for outcol in de98B.keys(): 
 #           for incol in deB.keys():
  #             CILQ_emp98_Beijing.loc[incol,outcol] = aggregateCILQ(SLQ_emp98,incol,outcol,de98B,deB)

#CILQ
CILQ_emp98_Guangdong_prep = pd.DataFrame(np.tile(np.array(list(SLQ_emp98.loc['Guangdong',:]), dtype = np.float),(n,1)),columns = de98B.keys(), index = de98B.keys())
CILQ_emp98_Guangdong = CILQ_emp98_Guangdong_prep.transpose().div(CILQ_emp98_Guangdong_prep, axis=0)

CILQ_emp98_Jiangsu_prep = pd.DataFrame(np.tile(np.array(list(SLQ_emp98.loc['Jiangsu',:]), dtype = np.float),(n,1)),columns = de98B.keys(), index = de98B.keys())
CILQ_emp98_Jiangsu = CILQ_emp98_Jiangsu_prep.transpose().div(CILQ_emp98_Jiangsu_prep, axis=0)

CILQ_emp98_Gansu_prep = pd.DataFrame(np.tile(np.array(list(SLQ_emp98.loc['Gansu',:]), dtype = np.float),(n,1)),columns = de98B.keys(), index = de98B.keys())
CILQ_emp98_Gansu = CILQ_emp98_Gansu_prep.transpose().div(CILQ_emp98_Gansu_prep, axis=0)

CILQ_emp98_Qinghai_prep = pd.DataFrame(np.tile(np.array(list(SLQ_emp98.loc['Qinghai',:]), dtype = np.float),(n,1)),columns = de98B.keys(), index = de98B.keys())
CILQ_emp98_Qinghai = CILQ_emp98_Qinghai_prep.transpose().div(CILQ_emp98_Qinghai_prep, axis=0)

CILQ_emp99_Guangdong_prep = pd.DataFrame(np.tile(np.array(list(SLQ_emp99.loc['Guangdong',:]), dtype = np.float),(n,1)),columns = de98B.keys(), index = de98B.keys())
CILQ_emp99_Guangdong = CILQ_emp99_Guangdong_prep.transpose().div(CILQ_emp99_Guangdong_prep, axis=0)

CILQ_emp99_Jiangsu_prep = pd.DataFrame(np.tile(np.array(list(SLQ_emp99.loc['Jiangsu',:]), dtype = np.float),(n,1)),columns = de98B.keys(), index = de98B.keys())
CILQ_emp99_Jiangsu = CILQ_emp99_Jiangsu_prep.transpose().div(CILQ_emp99_Jiangsu_prep, axis=0)

CILQ_emp99_Gansu_prep = pd.DataFrame(np.tile(np.array(list(SLQ_emp99.loc['Gansu',:]), dtype = np.float),(n,1)),columns = de98B.keys(), index = de98B.keys())
CILQ_emp99_Gansu = CILQ_emp99_Gansu_prep.transpose().div(CILQ_emp99_Gansu_prep, axis=0)

CILQ_emp99_Qinghai_prep = pd.DataFrame(np.tile(np.array(list(SLQ_emp99.loc['Qinghai',:]), dtype = np.float),(n,1)),columns = de98B.keys(), index = de98B.keys())
CILQ_emp99_Qinghai = CILQ_emp99_Qinghai_prep.transpose().div(CILQ_emp99_Qinghai_prep, axis=0)

CILQ_emp00_Guangdong_prep = pd.DataFrame(np.tile(np.array(list(SLQ_emp00.loc['Guangdong',:]), dtype = np.float),(n,1)),columns = de98B.keys(), index = de98B.keys())
CILQ_emp00_Guangdong = CILQ_emp00_Guangdong_prep.transpose().div(CILQ_emp00_Guangdong_prep, axis=0)

CILQ_emp00_Jiangsu_prep = pd.DataFrame(np.tile(np.array(list(SLQ_emp00.loc['Jiangsu',:]), dtype = np.float),(n,1)),columns = de98B.keys(), index = de98B.keys())
CILQ_emp00_Jiangsu = CILQ_emp00_Jiangsu_prep.transpose().div(CILQ_emp00_Jiangsu_prep, axis=0)

CILQ_emp00_Gansu_prep = pd.DataFrame(np.tile(np.array(list(SLQ_emp00.loc['Gansu',:]), dtype = np.float),(n,1)),columns = de98B.keys(), index = de98B.keys())
CILQ_emp00_Gansu = CILQ_emp00_Gansu_prep.transpose().div(CILQ_emp00_Gansu_prep, axis=0)

CILQ_emp00_Qinghai_prep = pd.DataFrame(np.tile(np.array(list(SLQ_emp00.loc['Qinghai',:]), dtype = np.float),(n,1)),columns = de98B.keys(), index = de98B.keys())
CILQ_emp00_Qinghai = CILQ_emp00_Qinghai_prep.transpose().div(CILQ_emp00_Qinghai_prep, axis=0)

CILQ_emp01_Guangdong_prep = pd.DataFrame(np.tile(np.array(list(SLQ_emp01.loc['Guangdong',:]), dtype = np.float),(n,1)),columns = de98B.keys(), index = de98B.keys())
CILQ_emp01_Guangdong = CILQ_emp01_Guangdong_prep.transpose().div(CILQ_emp01_Guangdong_prep, axis=0)

CILQ_emp01_Jiangsu_prep = pd.DataFrame(np.tile(np.array(list(SLQ_emp01.loc['Jiangsu',:]), dtype = np.float),(n,1)),columns = de98B.keys(), index = de98B.keys())
CILQ_emp01_Jiangsu = CILQ_emp01_Jiangsu_prep.transpose().div(CILQ_emp01_Jiangsu_prep, axis=0)

CILQ_emp01_Gansu_prep = pd.DataFrame(np.tile(np.array(list(SLQ_emp01.loc['Gansu',:]), dtype = np.float),(n,1)),columns = de98B.keys(), index = de98B.keys())
CILQ_emp01_Gansu = CILQ_emp01_Gansu_prep.transpose().div(CILQ_emp01_Gansu_prep, axis=0)

CILQ_emp01_Qinghai_prep = pd.DataFrame(np.tile(np.array(list(SLQ_emp01.loc['Qinghai',:]), dtype = np.float),(n,1)),columns = de98B.keys(), index = de98B.keys())
CILQ_emp01_Qinghai = CILQ_emp01_Qinghai_prep.transpose().div(CILQ_emp01_Qinghai_prep, axis=0)

CILQ_emp02_Guangdong_prep = pd.DataFrame(np.tile(np.array(list(SLQ_emp02.loc['Guangdong',:]), dtype = np.float),(n,1)),columns = de98B.keys(), index = de98B.keys())
CILQ_emp02_Guangdong = CILQ_emp02_Guangdong_prep.transpose().div(CILQ_emp02_Guangdong_prep, axis=0)

CILQ_emp02_Jiangsu_prep = pd.DataFrame(np.tile(np.array(list(SLQ_emp02.loc['Jiangsu',:]), dtype = np.float),(n,1)),columns = de98B.keys(), index = de98B.keys())
CILQ_emp02_Jiangsu = CILQ_emp02_Jiangsu_prep.transpose().div(CILQ_emp02_Jiangsu_prep, axis=0)

CILQ_emp02_Gansu_prep = pd.DataFrame(np.tile(np.array(list(SLQ_emp02.loc['Gansu',:]), dtype = np.float),(n,1)),columns = de98B.keys(), index = de98B.keys())
CILQ_emp02_Gansu = CILQ_emp02_Gansu_prep.transpose().div(CILQ_emp02_Gansu_prep, axis=0)

CILQ_emp02_Qinghai_prep = pd.DataFrame(np.tile(np.array(list(SLQ_emp02.loc['Qinghai',:]), dtype = np.float),(n,1)),columns = de98B.keys(), index = de98B.keys())
CILQ_emp02_Qinghai = CILQ_emp02_Qinghai_prep.transpose().div(CILQ_emp02_Qinghai_prep, axis=0)

CILQ_emp03_Guangdong_prep = pd.DataFrame(np.tile(np.array(list(SLQ_emp03.loc['Guangdong',:]), dtype = np.float),(n,1)),columns = de98B.keys(), index = de98B.keys())
CILQ_emp03_Guangdong = CILQ_emp03_Guangdong_prep.transpose().div(CILQ_emp03_Guangdong_prep, axis=0)

CILQ_emp03_Jiangsu_prep = pd.DataFrame(np.tile(np.array(list(SLQ_emp03.loc['Jiangsu',:]), dtype = np.float),(n,1)),columns = de98B.keys(), index = de98B.keys())
CILQ_emp03_Jiangsu = CILQ_emp03_Jiangsu_prep.transpose().div(CILQ_emp03_Jiangsu_prep, axis=0)

CILQ_emp03_Gansu_prep = pd.DataFrame(np.tile(np.array(list(SLQ_emp03.loc['Gansu',:]), dtype = np.float),(n,1)),columns = de98B.keys(), index = de98B.keys())
CILQ_emp03_Gansu = CILQ_emp03_Gansu_prep.transpose().div(CILQ_emp03_Gansu_prep, axis=0)

CILQ_emp03_Qinghai_prep = pd.DataFrame(np.tile(np.array(list(SLQ_emp03.loc['Qinghai',:]), dtype = np.float),(n,1)),columns = de98B.keys(), index = de98B.keys())
CILQ_emp03_Qinghai = CILQ_emp03_Qinghai_prep.transpose().div(CILQ_emp03_Qinghai_prep, axis=0)

CILQ_emp04_Guangdong_prep = pd.DataFrame(np.tile(np.array(list(SLQ_emp04.loc['Guangdong',:]), dtype = np.float),(n,1)),columns = de98B.keys(), index = de98B.keys())
CILQ_emp04_Guangdong = CILQ_emp04_Guangdong_prep.transpose().div(CILQ_emp04_Guangdong_prep, axis=0)

CILQ_emp04_Jiangsu_prep = pd.DataFrame(np.tile(np.array(list(SLQ_emp04.loc['Jiangsu',:]), dtype = np.float),(n,1)),columns = de98B.keys(), index = de98B.keys())
CILQ_emp04_Jiangsu = CILQ_emp04_Jiangsu_prep.transpose().div(CILQ_emp04_Jiangsu_prep, axis=0)

CILQ_emp04_Gansu_prep = pd.DataFrame(np.tile(np.array(list(SLQ_emp04.loc['Gansu',:]), dtype = np.float),(n,1)),columns = de98B.keys(), index = de98B.keys())
CILQ_emp04_Gansu = CILQ_emp04_Gansu_prep.transpose().div(CILQ_emp04_Gansu_prep, axis=0)

CILQ_emp04_Qinghai_prep = pd.DataFrame(np.tile(np.array(list(SLQ_emp04.loc['Qinghai',:]), dtype = np.float),(n,1)),columns = de98B.keys(), index = de98B.keys())
CILQ_emp04_Qinghai = CILQ_emp04_Qinghai_prep.transpose().div(CILQ_emp04_Qinghai_prep, axis=0)

CILQ_emp05_Guangdong_prep = pd.DataFrame(np.tile(np.array(list(SLQ_emp05.loc['Guangdong',:]), dtype = np.float),(n,1)),columns = de98B.keys(), index = de98B.keys())
CILQ_emp05_Guangdong = CILQ_emp05_Guangdong_prep.transpose().div(CILQ_emp05_Guangdong_prep, axis=0)

CILQ_emp05_Jiangsu_prep = pd.DataFrame(np.tile(np.array(list(SLQ_emp05.loc['Jiangsu',:]), dtype = np.float),(n,1)),columns = de98B.keys(), index = de98B.keys())
CILQ_emp05_Jiangsu = CILQ_emp05_Jiangsu_prep.transpose().div(CILQ_emp05_Jiangsu_prep, axis=0)

CILQ_emp05_Gansu_prep = pd.DataFrame(np.tile(np.array(list(SLQ_emp05.loc['Gansu',:]), dtype = np.float),(n,1)),columns = de98B.keys(), index = de98B.keys())
CILQ_emp05_Gansu = CILQ_emp05_Gansu_prep.transpose().div(CILQ_emp05_Gansu_prep, axis=0)

CILQ_emp05_Qinghai_prep = pd.DataFrame(np.tile(np.array(list(SLQ_emp05.loc['Qinghai',:]), dtype = np.float),(n,1)),columns = de98B.keys(), index = de98B.keys())
CILQ_emp05_Qinghai = CILQ_emp05_Qinghai_prep.transpose().div(CILQ_emp05_Qinghai_prep, axis=0)

CILQ_emp06_Guangdong_prep = pd.DataFrame(np.tile(np.array(list(SLQ_emp06.loc['Guangdong',:]), dtype = np.float),(n,1)),columns = de98B.keys(), index = de98B.keys())
CILQ_emp06_Guangdong = CILQ_emp06_Guangdong_prep.transpose().div(CILQ_emp06_Guangdong_prep, axis=0)

CILQ_emp06_Jiangsu_prep = pd.DataFrame(np.tile(np.array(list(SLQ_emp06.loc['Jiangsu',:]), dtype = np.float),(n,1)),columns = de98B.keys(), index = de98B.keys())
CILQ_emp06_Jiangsu = CILQ_emp06_Jiangsu_prep.transpose().div(CILQ_emp06_Jiangsu_prep, axis=0)

CILQ_emp06_Gansu_prep = pd.DataFrame(np.tile(np.array(list(SLQ_emp06.loc['Gansu',:]), dtype = np.float),(n,1)),columns = de98B.keys(), index = de98B.keys())
CILQ_emp06_Gansu = CILQ_emp06_Gansu_prep.transpose().div(CILQ_emp06_Gansu_prep, axis=0)

CILQ_emp06_Qinghai_prep = pd.DataFrame(np.tile(np.array(list(SLQ_emp06.loc['Qinghai',:]), dtype = np.float),(n,1)),columns = de98B.keys(), index = de98B.keys())
CILQ_emp06_Qinghai = CILQ_emp06_Qinghai_prep.transpose().div(CILQ_emp06_Qinghai_prep, axis=0)

CILQ_emp07_Guangdong_prep = pd.DataFrame(np.tile(np.array(list(SLQ_emp07.loc['Guangdong',:]), dtype = np.float),(n,1)),columns = de98B.keys(), index = de98B.keys())
CILQ_emp07_Guangdong = CILQ_emp07_Guangdong_prep.transpose().div(CILQ_emp07_Guangdong_prep, axis=0)

CILQ_emp07_Jiangsu_prep = pd.DataFrame(np.tile(np.array(list(SLQ_emp07.loc['Jiangsu',:]), dtype = np.float),(n,1)),columns = de98B.keys(), index = de98B.keys())
CILQ_emp07_Jiangsu = CILQ_emp07_Jiangsu_prep.transpose().div(CILQ_emp07_Jiangsu_prep, axis=0)

CILQ_emp07_Gansu_prep = pd.DataFrame(np.tile(np.array(list(SLQ_emp07.loc['Gansu',:]), dtype = np.float),(n,1)),columns = de98B.keys(), index = de98B.keys())
CILQ_emp07_Gansu = CILQ_emp07_Gansu_prep.transpose().div(CILQ_emp07_Gansu_prep, axis=0)

CILQ_emp07_Qinghai_prep = pd.DataFrame(np.tile(np.array(list(SLQ_emp07.loc['Qinghai',:]), dtype = np.float),(n,1)),columns = de98B.keys(), index = de98B.keys())
CILQ_emp07_Qinghai = CILQ_emp07_Qinghai_prep.transpose().div(CILQ_emp07_Qinghai_prep, axis=0)

CILQ_emp08_Guangdong_prep = pd.DataFrame(np.tile(np.array(list(SLQ_emp08.loc['Guangdong',:]), dtype = np.float),(n,1)),columns = de98B.keys(), index = de98B.keys())
CILQ_emp08_Guangdong = CILQ_emp08_Guangdong_prep.transpose().div(CILQ_emp08_Guangdong_prep, axis=0)

CILQ_emp08_Jiangsu_prep = pd.DataFrame(np.tile(np.array(list(SLQ_emp08.loc['Jiangsu',:]), dtype = np.float),(n,1)),columns = de98B.keys(), index = de98B.keys())
CILQ_emp08_Jiangsu = CILQ_emp08_Jiangsu_prep.transpose().div(CILQ_emp08_Jiangsu_prep, axis=0)

CILQ_emp08_Gansu_prep = pd.DataFrame(np.tile(np.array(list(SLQ_emp08.loc['Gansu',:]), dtype = np.float),(n,1)),columns = de98B.keys(), index = de98B.keys())
CILQ_emp08_Gansu = CILQ_emp08_Gansu_prep.transpose().div(CILQ_emp08_Gansu_prep, axis=0)

CILQ_emp08_Qinghai_prep = pd.DataFrame(np.tile(np.array(list(SLQ_emp08.loc['Qinghai',:]), dtype = np.float),(n,1)),columns = de98B.keys(), index = de98B.keys())
CILQ_emp08_Qinghai = CILQ_emp08_Qinghai_prep.transpose().div(CILQ_emp08_Qinghai_prep, axis=0)

CILQ_emp09_Guangdong_prep = pd.DataFrame(np.tile(np.array(list(SLQ_emp09.loc['Guangdong',:]), dtype = np.float),(n,1)),columns = de98B.keys(), index = de98B.keys())
CILQ_emp09_Guangdong = CILQ_emp09_Guangdong_prep.transpose().div(CILQ_emp09_Guangdong_prep, axis=0)

CILQ_emp09_Jiangsu_prep = pd.DataFrame(np.tile(np.array(list(SLQ_emp09.loc['Jiangsu',:]), dtype = np.float),(n,1)),columns = de98B.keys(), index = de98B.keys())
CILQ_emp09_Jiangsu = CILQ_emp09_Jiangsu_prep.transpose().div(CILQ_emp09_Jiangsu_prep, axis=0)

CILQ_emp09_Gansu_prep = pd.DataFrame(np.tile(np.array(list(SLQ_emp09.loc['Gansu',:]), dtype = np.float),(n,1)),columns = de98B.keys(), index = de98B.keys())
CILQ_emp09_Gansu = CILQ_emp09_Gansu_prep.transpose().div(CILQ_emp09_Gansu_prep, axis=0)

CILQ_emp09_Qinghai_prep = pd.DataFrame(np.tile(np.array(list(SLQ_emp09.loc['Qinghai',:]), dtype = np.float),(n,1)),columns = de98B.keys(), index = de98B.keys())
CILQ_emp09_Qinghai = CILQ_emp09_Qinghai_prep.transpose().div(CILQ_emp09_Qinghai_prep, axis=0)

CILQ_emp10_Guangdong_prep = pd.DataFrame(np.tile(np.array(list(SLQ_emp10.loc['Guangdong',:]), dtype = np.float),(n,1)),columns = de98B.keys(), index = de98B.keys())
CILQ_emp10_Guangdong = CILQ_emp10_Guangdong_prep.transpose().div(CILQ_emp10_Guangdong_prep, axis=0)

CILQ_emp10_Jiangsu_prep = pd.DataFrame(np.tile(np.array(list(SLQ_emp10.loc['Jiangsu',:]), dtype = np.float),(n,1)),columns = de98B.keys(), index = de98B.keys())
CILQ_emp10_Jiangsu = CILQ_emp10_Jiangsu_prep.transpose().div(CILQ_emp10_Jiangsu_prep, axis=0)

CILQ_emp10_Gansu_prep = pd.DataFrame(np.tile(np.array(list(SLQ_emp10.loc['Gansu',:]), dtype = np.float),(n,1)),columns = de98B.keys(), index = de98B.keys())
CILQ_emp10_Gansu = CILQ_emp10_Gansu_prep.transpose().div(CILQ_emp10_Gansu_prep, axis=0)

CILQ_emp10_Qinghai_prep = pd.DataFrame(np.tile(np.array(list(SLQ_emp10.loc['Qinghai',:]), dtype = np.float),(n,1)),columns = de98B.keys(), index = de98B.keys())
CILQ_emp10_Qinghai = CILQ_emp10_Qinghai_prep.transpose().div(CILQ_emp10_Qinghai_prep, axis=0)

CILQ_emp11_Guangdong_prep = pd.DataFrame(np.tile(np.array(list(SLQ_emp11.loc['Guangdong',:]), dtype = np.float),(n,1)),columns = de98B.keys(), index = de98B.keys())
CILQ_emp11_Guangdong = CILQ_emp11_Guangdong_prep.transpose().div(CILQ_emp11_Guangdong_prep, axis=0)

CILQ_emp11_Jiangsu_prep = pd.DataFrame(np.tile(np.array(list(SLQ_emp11.loc['Jiangsu',:]), dtype = np.float),(n,1)),columns = de98B.keys(), index = de98B.keys())
CILQ_emp11_Jiangsu = CILQ_emp11_Jiangsu_prep.transpose().div(CILQ_emp11_Jiangsu_prep, axis=0)

CILQ_emp11_Gansu_prep = pd.DataFrame(np.tile(np.array(list(SLQ_emp11.loc['Gansu',:]), dtype = np.float),(n,1)),columns = de98B.keys(), index = de98B.keys())
CILQ_emp11_Gansu = CILQ_emp11_Gansu_prep.transpose().div(CILQ_emp11_Gansu_prep, axis=0)

CILQ_emp11_Qinghai_prep = pd.DataFrame(np.tile(np.array(list(SLQ_emp11.loc['Qinghai',:]), dtype = np.float),(n,1)),columns = de98B.keys(), index = de98B.keys())
CILQ_emp11_Qinghai = CILQ_emp11_Qinghai_prep.transpose().div(CILQ_emp11_Qinghai_prep, axis=0)

#FLQ


#removing CILQ >1
CILQ_emp98_Guangdong[:] = CILQ_emp98_Guangdong[:].apply(lambda x: [y if y <= 1 else 1 for y in x])
CILQ_emp98_Jiangsu[:] = CILQ_emp98_Jiangsu[:].apply(lambda x: [y if y <= 1 else 1 for y in x])
CILQ_emp98_Gansu[:] = CILQ_emp98_Gansu[:].apply(lambda x: [y if y <= 1 else 1 for y in x])
CILQ_emp98_Qinghai[:] = CILQ_emp98_Qinghai[:].apply(lambda x: [y if y <= 1 else 1 for y in x])

CILQ_emp99_Guangdong[:] = CILQ_emp99_Guangdong[:].apply(lambda x: [y if y <= 1 else 1 for y in x])
CILQ_emp99_Jiangsu[:] = CILQ_emp99_Jiangsu[:].apply(lambda x: [y if y <= 1 else 1 for y in x])
CILQ_emp99_Gansu[:] = CILQ_emp99_Gansu[:].apply(lambda x: [y if y <= 1 else 1 for y in x])
CILQ_emp99_Qinghai[:] = CILQ_emp99_Qinghai[:].apply(lambda x: [y if y <= 1 else 1 for y in x])

CILQ_emp00_Guangdong[:] = CILQ_emp00_Guangdong[:].apply(lambda x: [y if y <= 1 else 1 for y in x])
CILQ_emp00_Jiangsu[:] = CILQ_emp00_Jiangsu[:].apply(lambda x: [y if y <= 1 else 1 for y in x])
CILQ_emp00_Gansu[:] = CILQ_emp00_Gansu[:].apply(lambda x: [y if y <= 1 else 1 for y in x])
CILQ_emp00_Qinghai[:] = CILQ_emp00_Qinghai[:].apply(lambda x: [y if y <= 1 else 1 for y in x])

CILQ_emp01_Guangdong[:] = CILQ_emp01_Guangdong[:].apply(lambda x: [y if y <= 1 else 1 for y in x])
CILQ_emp01_Jiangsu[:] = CILQ_emp01_Jiangsu[:].apply(lambda x: [y if y <= 1 else 1 for y in x])
CILQ_emp01_Gansu[:] = CILQ_emp01_Gansu[:].apply(lambda x: [y if y <= 1 else 1 for y in x])
CILQ_emp01_Qinghai[:] = CILQ_emp01_Qinghai[:].apply(lambda x: [y if y <= 1 else 1 for y in x])

CILQ_emp02_Guangdong[:] = CILQ_emp02_Guangdong[:].apply(lambda x: [y if y <= 1 else 1 for y in x])
CILQ_emp02_Jiangsu[:] = CILQ_emp02_Jiangsu[:].apply(lambda x: [y if y <= 1 else 1 for y in x])
CILQ_emp02_Gansu[:] = CILQ_emp02_Gansu[:].apply(lambda x: [y if y <= 1 else 1 for y in x])
CILQ_emp02_Qinghai[:] = CILQ_emp02_Qinghai[:].apply(lambda x: [y if y <= 1 else 1 for y in x])

CILQ_emp03_Guangdong[:] = CILQ_emp03_Guangdong[:].apply(lambda x: [y if y <= 1 else 1 for y in x])
CILQ_emp03_Jiangsu[:] = CILQ_emp03_Jiangsu[:].apply(lambda x: [y if y <= 1 else 1 for y in x])
CILQ_emp03_Gansu[:] = CILQ_emp03_Gansu[:].apply(lambda x: [y if y <= 1 else 1 for y in x])
CILQ_emp03_Qinghai[:] = CILQ_emp03_Qinghai[:].apply(lambda x: [y if y <= 1 else 1 for y in x])

CILQ_emp04_Guangdong[:] = CILQ_emp04_Guangdong[:].apply(lambda x: [y if y <= 1 else 1 for y in x])
CILQ_emp04_Jiangsu[:] = CILQ_emp04_Jiangsu[:].apply(lambda x: [y if y <= 1 else 1 for y in x])
CILQ_emp04_Gansu[:] = CILQ_emp04_Gansu[:].apply(lambda x: [y if y <= 1 else 1 for y in x])
CILQ_emp04_Qinghai[:] = CILQ_emp04_Qinghai[:].apply(lambda x: [y if y <= 1 else 1 for y in x])

CILQ_emp05_Guangdong[:] = CILQ_emp05_Guangdong[:].apply(lambda x: [y if y <= 1 else 1 for y in x])
CILQ_emp05_Jiangsu[:] = CILQ_emp05_Jiangsu[:].apply(lambda x: [y if y <= 1 else 1 for y in x])
CILQ_emp05_Gansu[:] = CILQ_emp05_Gansu[:].apply(lambda x: [y if y <= 1 else 1 for y in x])
CILQ_emp05_Qinghai[:] = CILQ_emp05_Qinghai[:].apply(lambda x: [y if y <= 1 else 1 for y in x])

CILQ_emp06_Guangdong[:] = CILQ_emp06_Guangdong[:].apply(lambda x: [y if y <= 1 else 1 for y in x])
CILQ_emp06_Jiangsu[:] = CILQ_emp06_Jiangsu[:].apply(lambda x: [y if y <= 1 else 1 for y in x])
CILQ_emp06_Gansu[:] = CILQ_emp06_Gansu[:].apply(lambda x: [y if y <= 1 else 1 for y in x])
CILQ_emp06_Qinghai[:] = CILQ_emp06_Qinghai[:].apply(lambda x: [y if y <= 1 else 1 for y in x])

CILQ_emp07_Guangdong[:] = CILQ_emp07_Guangdong[:].apply(lambda x: [y if y <= 1 else 1 for y in x])
CILQ_emp07_Jiangsu[:] = CILQ_emp07_Jiangsu[:].apply(lambda x: [y if y <= 1 else 1 for y in x])
CILQ_emp07_Gansu[:] = CILQ_emp07_Gansu[:].apply(lambda x: [y if y <= 1 else 1 for y in x])
CILQ_emp07_Qinghai[:] = CILQ_emp07_Qinghai[:].apply(lambda x: [y if y <= 1 else 1 for y in x])

CILQ_emp08_Guangdong[:] = CILQ_emp08_Guangdong[:].apply(lambda x: [y if y <= 1 else 1 for y in x])
CILQ_emp08_Jiangsu[:] = CILQ_emp08_Jiangsu[:].apply(lambda x: [y if y <= 1 else 1 for y in x])
CILQ_emp08_Gansu[:] = CILQ_emp08_Gansu[:].apply(lambda x: [y if y <= 1 else 1 for y in x])
CILQ_emp08_Qinghai[:] = CILQ_emp08_Qinghai[:].apply(lambda x: [y if y <= 1 else 1 for y in x])

CILQ_emp09_Guangdong[:] = CILQ_emp09_Guangdong[:].apply(lambda x: [y if y <= 1 else 1 for y in x])
CILQ_emp09_Jiangsu[:] = CILQ_emp09_Jiangsu[:].apply(lambda x: [y if y <= 1 else 1 for y in x])
CILQ_emp09_Gansu[:] = CILQ_emp09_Gansu[:].apply(lambda x: [y if y <= 1 else 1 for y in x])
CILQ_emp09_Qinghai[:] = CILQ_emp09_Qinghai[:].apply(lambda x: [y if y <= 1 else 1 for y in x])

CILQ_emp10_Guangdong[:] = CILQ_emp10_Guangdong[:].apply(lambda x: [y if y <= 1 else 1 for y in x])
CILQ_emp10_Jiangsu[:] = CILQ_emp10_Jiangsu[:].apply(lambda x: [y if y <= 1 else 1 for y in x])
CILQ_emp10_Gansu[:] = CILQ_emp10_Gansu[:].apply(lambda x: [y if y <= 1 else 1 for y in x])
CILQ_emp10_Qinghai[:] = CILQ_emp10_Qinghai[:].apply(lambda x: [y if y <= 1 else 1 for y in x])

CILQ_emp11_Guangdong[:] = CILQ_emp11_Guangdong[:].apply(lambda x: [y if y <= 1 else 1 for y in x])
CILQ_emp11_Jiangsu[:] = CILQ_emp11_Jiangsu[:].apply(lambda x: [y if y <= 1 else 1 for y in x])
CILQ_emp11_Gansu[:] = CILQ_emp11_Gansu[:].apply(lambda x: [y if y <= 1 else 1 for y in x])
CILQ_emp11_Qinghai[:] = CILQ_emp11_Qinghai[:].apply(lambda x: [y if y <= 1 else 1 for y in x])

# create A matrix out of newntio
Y98 = pd.DataFrame(io98.iloc[-1,0:m]).transpose()
def aggregateY(df,row,col,dio):
    subdf = df.loc[:,dio[col]]
    return subdf.sum(axis=1).sum()
newY98 = pd.DataFrame(0, columns=dio.keys(), index = ["37"])
for outcol in dio.keys(): 
            for incol in ["37"]:
               newY98.loc[incol,outcol] = aggregateY(Y98,incol,outcol,dio)
               
cFY98 = pd.DataFrame(np.zeros((n,1),dtype = np.array(list(newY98.loc['37',:]), dtype = np.float).dtype) + np.array(list(newY98.loc['37',:]), dtype = np.float),columns = dio.keys(), index = dio.keys())
A98 = newntio98.div(cFY98, axis=0)

#year 1999
Y99 = pd.DataFrame(io99.iloc[-1,0:m]).transpose()
newY99 = pd.DataFrame(0, columns=dio.keys(), index = ["37"])
for outcol in dio.keys(): 
            for incol in ["37"]:
               newY99.loc[incol,outcol] = aggregateY(Y99,incol,outcol,dio)
               
cFY99 = pd.DataFrame(np.zeros((n,1),dtype = np.array(list(newY99.loc['37',:]), dtype = np.float).dtype) + np.array(list(newY99.loc['37',:]), dtype = np.float),columns = dio.keys(), index = dio.keys())
A99 = newntio99.div(cFY99, axis=0)

#year 2000
Y00 = pd.DataFrame(io00.iloc[-1,0:m]).transpose()
newY00 = pd.DataFrame(0, columns=dio.keys(), index = ["37"])
for outcol in dio.keys(): 
            for incol in ["37"]:
               newY00.loc[incol,outcol] = aggregateY(Y00,incol,outcol,dio)
               
cFY00 = pd.DataFrame(np.zeros((n,1),dtype = np.array(list(newY00.loc['37',:]), dtype = np.float).dtype) + np.array(list(newY00.loc['37',:]), dtype = np.float),columns = dio.keys(), index = dio.keys())
A00 = newntio00.div(cFY00, axis=0)

#year 2001
Y01 = pd.DataFrame(io01.iloc[-1,0:m]).transpose()
newY01 = pd.DataFrame(0, columns=dio.keys(), index = ["37"])
for outcol in dio.keys(): 
            for incol in ["37"]:
               newY01.loc[incol,outcol] = aggregateY(Y01,incol,outcol,dio)
               
cFY01 = pd.DataFrame(np.zeros((n,1),dtype = np.array(list(newY01.loc['37',:]), dtype = np.float).dtype) + np.array(list(newY01.loc['37',:]), dtype = np.float),columns = dio.keys(), index = dio.keys())
A01 = newntio01.div(cFY01, axis=0)

#year 2002
Y02 = pd.DataFrame(io02.iloc[-1,0:m]).transpose()
newY02 = pd.DataFrame(0, columns=dio.keys(), index = ["37"])
for outcol in dio.keys(): 
            for incol in ["37"]:
               newY02.loc[incol,outcol] = aggregateY(Y02,incol,outcol,dio)
               
cFY02 = pd.DataFrame(np.zeros((n,1),dtype = np.array(list(newY02.loc['37',:]), dtype = np.float).dtype) + np.array(list(newY02.loc['37',:]), dtype = np.float),columns = dio.keys(), index = dio.keys())
A02 = newntio02.div(cFY02, axis=0)

#year 2003
Y03 = pd.DataFrame(io03.iloc[-1,0:m]).transpose()
newY03 = pd.DataFrame(0, columns=dio.keys(), index = ["37"])
for outcol in dio.keys(): 
            for incol in ["37"]:
               newY03.loc[incol,outcol] = aggregateY(Y03,incol,outcol,dio)
               
cFY03 = pd.DataFrame(np.zeros((n,1),dtype = np.array(list(newY03.loc['37',:]), dtype = np.float).dtype) + np.array(list(newY03.loc['37',:]), dtype = np.float),columns = dio.keys(), index = dio.keys())
A03 = newntio03.div(cFY03, axis=0)

#year 2004
Y04 = pd.DataFrame(io04.iloc[-1,0:m]).transpose()
newY04 = pd.DataFrame(0, columns=dio.keys(), index = ["37"])
for outcol in dio.keys(): 
            for incol in ["37"]:
               newY04.loc[incol,outcol] = aggregateY(Y04,incol,outcol,dio)
               
cFY04 = pd.DataFrame(np.zeros((n,1),dtype = np.array(list(newY04.loc['37',:]), dtype = np.float).dtype) + np.array(list(newY04.loc['37',:]), dtype = np.float),columns = dio.keys(), index = dio.keys())
A04 = newntio04.div(cFY04, axis=0)

#year 2005
Y05 = pd.DataFrame(io05.iloc[-1,0:m]).transpose()
newY05 = pd.DataFrame(0, columns=dio.keys(), index = ["37"])
for outcol in dio.keys(): 
            for incol in ["37"]:
               newY05.loc[incol,outcol] = aggregateY(Y05,incol,outcol,dio)
               
cFY05 = pd.DataFrame(np.zeros((n,1),dtype = np.array(list(newY05.loc['37',:]), dtype = np.float).dtype) + np.array(list(newY05.loc['37',:]), dtype = np.float),columns = dio.keys(), index = dio.keys())
A05 = newntio05.div(cFY05, axis=0)

#year 2006
Y06 = pd.DataFrame(io06.iloc[-1,0:m]).transpose()
newY06 = pd.DataFrame(0, columns=dio.keys(), index = ["37"])
for outcol in dio.keys(): 
            for incol in ["37"]:
               newY06.loc[incol,outcol] = aggregateY(Y06,incol,outcol,dio)
               
cFY06 = pd.DataFrame(np.zeros((n,1),dtype = np.array(list(newY06.loc['37',:]), dtype = np.float).dtype) + np.array(list(newY06.loc['37',:]), dtype = np.float),columns = dio.keys(), index = dio.keys())
A06 = newntio06.div(cFY06, axis=0)

#year 2007
Y07 = pd.DataFrame(io07.iloc[-1,0:m]).transpose()
newY07 = pd.DataFrame(0, columns=dio.keys(), index = ["37"])
for outcol in dio.keys(): 
            for incol in ["37"]:
               newY07.loc[incol,outcol] = aggregateY(Y07,incol,outcol,dio)
               
cFY07 = pd.DataFrame(np.zeros((n,1),dtype = np.array(list(newY07.loc['37',:]), dtype = np.float).dtype) + np.array(list(newY07.loc['37',:]), dtype = np.float),columns = dio.keys(), index = dio.keys())
A07 = newntio07.div(cFY07, axis=0)

#year 2008
Y08 = pd.DataFrame(io08.iloc[-1,0:m]).transpose()
newY08 = pd.DataFrame(0, columns=dio.keys(), index = ["37"])
for outcol in dio.keys(): 
            for incol in ["37"]:
               newY08.loc[incol,outcol] = aggregateY(Y08,incol,outcol,dio)
               
cFY08 = pd.DataFrame(np.zeros((n,1),dtype = np.array(list(newY08.loc['37',:]), dtype = np.float).dtype) + np.array(list(newY08.loc['37',:]), dtype = np.float),columns = dio.keys(), index = dio.keys())
A08 = newntio08.div(cFY08, axis=0)

#year 2009
Y09 = pd.DataFrame(io09.iloc[-1,0:m]).transpose()
newY09 = pd.DataFrame(0, columns=dio.keys(), index = ["37"])
for outcol in dio.keys(): 
            for incol in ["37"]:
               newY09.loc[incol,outcol] = aggregateY(Y09,incol,outcol,dio)
               
cFY09 = pd.DataFrame(np.zeros((n,1),dtype = np.array(list(newY09.loc['37',:]), dtype = np.float).dtype) + np.array(list(newY09.loc['37',:]), dtype = np.float),columns = dio.keys(), index = dio.keys())
A09 = newntio09.div(cFY09, axis=0)

#year 2010
Y10 = pd.DataFrame(io10.iloc[-1,0:m]).transpose()
newY10 = pd.DataFrame(0, columns=dio.keys(), index = ["37"])
for outcol in dio.keys(): 
            for incol in ["37"]:
               newY10.loc[incol,outcol] = aggregateY(Y10,incol,outcol,dio)
               
cFY10 = pd.DataFrame(np.zeros((n,1),dtype = np.array(list(newY10.loc['37',:]), dtype = np.float).dtype) + np.array(list(newY10.loc['37',:]), dtype = np.float),columns = dio.keys(), index = dio.keys())
A10 = newntio10.div(cFY10, axis=0)

#year 2011
Y11 = pd.DataFrame(io11.iloc[-1,0:m]).transpose()
newY11 = pd.DataFrame(0, columns=dio.keys(), index = ["37"])
for outcol in dio.keys(): 
            for incol in ["37"]:
               newY11.loc[incol,outcol] = aggregateY(Y11,incol,outcol,dio)
               
cFY11 = pd.DataFrame(np.zeros((n,1),dtype = np.array(list(newY11.loc['37',:]), dtype = np.float).dtype) + np.array(list(newY11.loc['37',:]), dtype = np.float),columns = dio.keys(), index = dio.keys())
A11 = newntio11.div(cFY11, axis=0)

#Creating regional A matrices
# regionalisation of A98
ntio98_Guangdong = A98.values*CILQ_emp98_Guangdong
ntio98_Jiangsu = A98.values*CILQ_emp98_Jiangsu
ntio98_Gansu = A98.values*CILQ_emp98_Gansu
ntio98_Qinghai = A98.values*CILQ_emp98_Qinghai

# regionalisation of A99
ntio99_Guangdong = A99.values*CILQ_emp99_Guangdong
ntio99_Jiangsu = A99.values*CILQ_emp99_Jiangsu
ntio99_Gansu = A99.values*CILQ_emp99_Gansu
ntio99_Qinghai = A99.values*CILQ_emp99_Qinghai

# regionalisation of A00
ntio00_Guangdong = A00.values*CILQ_emp00_Guangdong
ntio00_Jiangsu = A00.values*CILQ_emp00_Jiangsu
ntio00_Gansu = A00.values*CILQ_emp00_Gansu
ntio00_Qinghai = A00.values*CILQ_emp00_Qinghai

# regionalisation of A01
ntio01_Guangdong = A01.values*CILQ_emp01_Guangdong
ntio01_Jiangsu = A01.values*CILQ_emp01_Jiangsu
ntio01_Gansu = A01.values*CILQ_emp01_Gansu
ntio01_Qinghai = A01.values*CILQ_emp01_Qinghai

# regionalisation of A02
ntio02_Guangdong = A02.values*CILQ_emp02_Guangdong
ntio02_Jiangsu = A02.values*CILQ_emp02_Jiangsu
ntio02_Gansu = A02.values*CILQ_emp02_Gansu
ntio02_Qinghai = A02.values*CILQ_emp02_Qinghai

# regionalisation of A03
ntio03_Guangdong = A03.values*CILQ_emp03_Guangdong
ntio03_Jiangsu = A03.values*CILQ_emp03_Jiangsu
ntio03_Gansu = A03.values*CILQ_emp03_Gansu
ntio03_Qinghai = A03.values*CILQ_emp03_Qinghai

# regionalisation of A04
ntio04_Guangdong = A04.values*CILQ_emp04_Guangdong
ntio04_Jiangsu = A04.values*CILQ_emp04_Jiangsu
ntio04_Gansu = A04.values*CILQ_emp04_Gansu
ntio04_Qinghai = A04.values*CILQ_emp04_Qinghai

# regionalisation of A05
ntio05_Guangdong = A05.values*CILQ_emp05_Guangdong
ntio05_Jiangsu = A05.values*CILQ_emp05_Jiangsu
ntio05_Gansu = A05.values*CILQ_emp05_Gansu
ntio05_Qinghai = A05.values*CILQ_emp05_Qinghai

# regionalisation of A06
ntio06_Guangdong = A06.values*CILQ_emp06_Guangdong
ntio06_Jiangsu = A06.values*CILQ_emp06_Jiangsu
ntio06_Gansu = A06.values*CILQ_emp06_Gansu
ntio06_Qinghai = A06.values*CILQ_emp06_Qinghai

# regionalisation of A07
ntio07_Guangdong = A07.values*CILQ_emp07_Guangdong
ntio07_Jiangsu = A07.values*CILQ_emp07_Jiangsu
ntio07_Gansu = A07.values*CILQ_emp07_Gansu
ntio07_Qinghai = A07.values*CILQ_emp07_Qinghai

# regionalisation of A08
ntio08_Guangdong = A08.values*CILQ_emp08_Guangdong
ntio08_Jiangsu = A08.values*CILQ_emp08_Jiangsu
ntio08_Gansu = A08.values*CILQ_emp08_Gansu
ntio08_Qinghai = A08.values*CILQ_emp08_Qinghai

# regionalisation of A04
ntio09_Guangdong = A09.values*CILQ_emp09_Guangdong
ntio09_Jiangsu = A09.values*CILQ_emp09_Jiangsu
ntio09_Gansu = A09.values*CILQ_emp09_Gansu
ntio09_Qinghai = A09.values*CILQ_emp09_Qinghai

# regionalisation of A04
ntio10_Guangdong = A10.values*CILQ_emp10_Guangdong
ntio10_Jiangsu = A10.values*CILQ_emp10_Jiangsu
ntio10_Gansu = A10.values*CILQ_emp10_Gansu
ntio10_Qinghai = A10.values*CILQ_emp10_Qinghai

# regionalisation of A11
ntio11_Guangdong = A11.values*CILQ_emp11_Guangdong
ntio11_Jiangsu = A11.values*CILQ_emp11_Jiangsu
ntio11_Gansu = A11.values*CILQ_emp11_Gansu
ntio11_Qinghai = A11.values*CILQ_emp11_Qinghai

#Weighting regional A matrices
# Weighting A98
wntio98_Guangdong = np.matmul(np.matrix(ntio98_Guangdong),np.matrix(np.diag(np.array(newemp98.loc["Guangdong","1":"9"].div(newemp98.loc["Guangdong","Total"]), dtype = np.float))))
wntio98_Jiangsu = np.matmul(np.matrix(ntio98_Jiangsu),np.matrix(np.diag(np.array(newemp98.loc["Jiangsu","1":"9"].div(newemp98.loc["Jiangsu","Total"]), dtype = np.float))))
wntio98_Gansu = np.matmul(np.matrix(ntio98_Gansu),np.matrix(np.diag(np.array(newemp98.loc["Gansu","1":"9"].div(newemp98.loc["Gansu","Total"]), dtype = np.float))))
wntio98_Qinghai = np.matmul(np.matrix(ntio98_Qinghai),np.matrix(np.diag(np.array(newemp98.loc["Qinghai","1":"9"].div(newemp98.loc["Qinghai","Total"]), dtype = np.float))))

# Weighting A99
wntio99_Guangdong = np.matmul(np.matrix(ntio99_Guangdong),np.matrix(np.diag(np.array(newemp99.loc["Guangdong","1":"9"].div(newemp99.loc["Guangdong","Total"]), dtype = np.float))))
wntio99_Jiangsu = np.matmul(np.matrix(ntio99_Jiangsu),np.matrix(np.diag(np.array(newemp99.loc["Jiangsu","1":"9"].div(newemp99.loc["Jiangsu","Total"]), dtype = np.float))))
wntio99_Gansu = np.matmul(np.matrix(ntio99_Gansu),np.matrix(np.diag(np.array(newemp99.loc["Gansu","1":"9"].div(newemp99.loc["Gansu","Total"]), dtype = np.float))))
wntio99_Qinghai = np.matmul(np.matrix(ntio99_Qinghai),np.matrix(np.diag(np.array(newemp99.loc["Qinghai","1":"9"].div(newemp99.loc["Qinghai","Total"]), dtype = np.float))))

# Weighting A00
wntio00_Guangdong = np.matmul(np.matrix(ntio00_Guangdong),np.matrix(np.diag(np.array(newemp00.loc["Guangdong","1":"9"].div(newemp00.loc["Guangdong","Total"]), dtype = np.float))))
wntio00_Jiangsu = np.matmul(np.matrix(ntio00_Jiangsu),np.matrix(np.diag(np.array(newemp00.loc["Jiangsu","1":"9"].div(newemp00.loc["Jiangsu","Total"]), dtype = np.float))))
wntio00_Gansu = np.matmul(np.matrix(ntio00_Gansu),np.matrix(np.diag(np.array(newemp00.loc["Gansu","1":"9"].div(newemp00.loc["Gansu","Total"]), dtype = np.float))))
wntio00_Qinghai = np.matmul(np.matrix(ntio00_Qinghai),np.matrix(np.diag(np.array(newemp00.loc["Qinghai","1":"9"].div(newemp00.loc["Qinghai","Total"]), dtype = np.float))))

# Weighting A01
wntio01_Guangdong = np.matmul(np.matrix(ntio01_Guangdong),np.matrix(np.diag(np.array(newemp01.loc["Guangdong","1":"9"].div(newemp01.loc["Guangdong","Total"]), dtype = np.float))))
wntio01_Jiangsu = np.matmul(np.matrix(ntio01_Jiangsu),np.matrix(np.diag(np.array(newemp01.loc["Jiangsu","1":"9"].div(newemp01.loc["Jiangsu","Total"]), dtype = np.float))))
wntio01_Gansu = np.matmul(np.matrix(ntio01_Gansu),np.matrix(np.diag(np.array(newemp01.loc["Gansu","1":"9"].div(newemp01.loc["Gansu","Total"]), dtype = np.float))))
wntio01_Qinghai = np.matmul(np.matrix(ntio01_Qinghai),np.matrix(np.diag(np.array(newemp01.loc["Qinghai","1":"9"].div(newemp01.loc["Qinghai","Total"]), dtype = np.float))))

# Weighting A02
wntio02_Guangdong = np.matmul(np.matrix(ntio02_Guangdong),np.matrix(np.diag(np.array(newemp02.loc["Guangdong","1":"9"].div(newemp02.loc["Guangdong","Total"]), dtype = np.float))))
wntio02_Jiangsu = np.matmul(np.matrix(ntio02_Jiangsu),np.matrix(np.diag(np.array(newemp02.loc["Jiangsu","1":"9"].div(newemp02.loc["Jiangsu","Total"]), dtype = np.float))))
wntio02_Gansu = np.matmul(np.matrix(ntio02_Gansu),np.matrix(np.diag(np.array(newemp02.loc["Gansu","1":"9"].div(newemp02.loc["Gansu","Total"]), dtype = np.float))))
wntio02_Qinghai = np.matmul(np.matrix(ntio02_Qinghai),np.matrix(np.diag(np.array(newemp02.loc["Qinghai","1":"9"].div(newemp02.loc["Qinghai","Total"]), dtype = np.float))))

# Weighting A03
wntio03_Guangdong = np.matmul(np.matrix(ntio03_Guangdong),np.matrix(np.diag(np.array(newemp03.loc["Guangdong","1":"9"].div(newemp03.loc["Guangdong","Total"]), dtype = np.float))))
wntio03_Jiangsu = np.matmul(np.matrix(ntio03_Jiangsu),np.matrix(np.diag(np.array(newemp03.loc["Jiangsu","1":"9"].div(newemp03.loc["Jiangsu","Total"]), dtype = np.float))))
wntio03_Gansu = np.matmul(np.matrix(ntio03_Gansu),np.matrix(np.diag(np.array(newemp03.loc["Gansu","1":"9"].div(newemp03.loc["Gansu","Total"]), dtype = np.float))))
wntio03_Qinghai = np.matmul(np.matrix(ntio03_Qinghai),np.matrix(np.diag(np.array(newemp03.loc["Qinghai","1":"9"].div(newemp03.loc["Qinghai","Total"]), dtype = np.float))))

# Weighting A04
wntio04_Guangdong = np.matmul(np.matrix(ntio04_Guangdong),np.matrix(np.diag(np.array(newemp04.loc["Guangdong","1":"9"].div(newemp04.loc["Guangdong","Total"]), dtype = np.float))))
wntio04_Jiangsu = np.matmul(np.matrix(ntio04_Jiangsu),np.matrix(np.diag(np.array(newemp04.loc["Jiangsu","1":"9"].div(newemp04.loc["Jiangsu","Total"]), dtype = np.float))))
wntio04_Gansu = np.matmul(np.matrix(ntio04_Gansu),np.matrix(np.diag(np.array(newemp04.loc["Gansu","1":"9"].div(newemp04.loc["Gansu","Total"]), dtype = np.float))))
wntio04_Qinghai = np.matmul(np.matrix(ntio04_Qinghai),np.matrix(np.diag(np.array(newemp04.loc["Qinghai","1":"9"].div(newemp04.loc["Qinghai","Total"]), dtype = np.float))))

# Weighting A05
wntio05_Guangdong = np.matmul(np.matrix(ntio05_Guangdong),np.matrix(np.diag(np.array(newemp05.loc["Guangdong","1":"9"].div(newemp05.loc["Guangdong","Total"]), dtype = np.float))))
wntio05_Jiangsu = np.matmul(np.matrix(ntio05_Jiangsu),np.matrix(np.diag(np.array(newemp05.loc["Jiangsu","1":"9"].div(newemp05.loc["Jiangsu","Total"]), dtype = np.float))))
wntio05_Gansu = np.matmul(np.matrix(ntio05_Gansu),np.matrix(np.diag(np.array(newemp05.loc["Gansu","1":"9"].div(newemp05.loc["Gansu","Total"]), dtype = np.float))))
wntio05_Qinghai = np.matmul(np.matrix(ntio05_Qinghai),np.matrix(np.diag(np.array(newemp05.loc["Qinghai","1":"9"].div(newemp05.loc["Qinghai","Total"]), dtype = np.float))))

# Weighting A06
wntio06_Guangdong = np.matmul(np.matrix(ntio06_Guangdong),np.matrix(np.diag(np.array(newemp06.loc["Guangdong","1":"9"].div(newemp06.loc["Guangdong","Total"]), dtype = np.float))))
wntio06_Jiangsu = np.matmul(np.matrix(ntio06_Jiangsu),np.matrix(np.diag(np.array(newemp06.loc["Jiangsu","1":"9"].div(newemp06.loc["Jiangsu","Total"]), dtype = np.float))))
wntio06_Gansu = np.matmul(np.matrix(ntio06_Gansu),np.matrix(np.diag(np.array(newemp06.loc["Gansu","1":"9"].div(newemp06.loc["Gansu","Total"]), dtype = np.float))))
wntio06_Qinghai = np.matmul(np.matrix(ntio06_Qinghai),np.matrix(np.diag(np.array(newemp06.loc["Qinghai","1":"9"].div(newemp06.loc["Qinghai","Total"]), dtype = np.float))))

# Weighting A07
wntio07_Guangdong = np.matmul(np.matrix(ntio07_Guangdong),np.matrix(np.diag(np.array(newemp07.loc["Guangdong","1":"9"].div(newemp07.loc["Guangdong","Total"]), dtype = np.float))))
wntio07_Jiangsu = np.matmul(np.matrix(ntio07_Jiangsu),np.matrix(np.diag(np.array(newemp07.loc["Jiangsu","1":"9"].div(newemp07.loc["Jiangsu","Total"]), dtype = np.float))))
wntio07_Gansu = np.matmul(np.matrix(ntio07_Gansu),np.matrix(np.diag(np.array(newemp07.loc["Gansu","1":"9"].div(newemp07.loc["Gansu","Total"]), dtype = np.float))))
wntio07_Qinghai = np.matmul(np.matrix(ntio07_Qinghai),np.matrix(np.diag(np.array(newemp07.loc["Qinghai","1":"9"].div(newemp07.loc["Qinghai","Total"]), dtype = np.float))))

# Weighting A08
wntio08_Guangdong = np.matmul(np.matrix(ntio08_Guangdong),np.matrix(np.diag(np.array(newemp08.loc["Guangdong","1":"9"].div(newemp08.loc["Guangdong","Total"]), dtype = np.float))))
wntio08_Jiangsu = np.matmul(np.matrix(ntio08_Jiangsu),np.matrix(np.diag(np.array(newemp08.loc["Jiangsu","1":"9"].div(newemp08.loc["Jiangsu","Total"]), dtype = np.float))))
wntio08_Gansu = np.matmul(np.matrix(ntio08_Gansu),np.matrix(np.diag(np.array(newemp08.loc["Gansu","1":"9"].div(newemp08.loc["Gansu","Total"]), dtype = np.float))))
wntio08_Qinghai = np.matmul(np.matrix(ntio08_Qinghai),np.matrix(np.diag(np.array(newemp08.loc["Qinghai","1":"9"].div(newemp08.loc["Qinghai","Total"]), dtype = np.float))))

# Weighting A09
wntio09_Guangdong = np.matmul(np.matrix(ntio09_Guangdong),np.matrix(np.diag(np.array(newemp09.loc["Guangdong","1":"9"].div(newemp09.loc["Guangdong","Total"]), dtype = np.float))))
wntio09_Jiangsu = np.matmul(np.matrix(ntio09_Jiangsu),np.matrix(np.diag(np.array(newemp09.loc["Jiangsu","1":"9"].div(newemp09.loc["Jiangsu","Total"]), dtype = np.float))))
wntio09_Gansu = np.matmul(np.matrix(ntio09_Gansu),np.matrix(np.diag(np.array(newemp09.loc["Gansu","1":"9"].div(newemp09.loc["Gansu","Total"]), dtype = np.float))))
wntio09_Qinghai = np.matmul(np.matrix(ntio09_Qinghai),np.matrix(np.diag(np.array(newemp09.loc["Qinghai","1":"9"].div(newemp09.loc["Qinghai","Total"]), dtype = np.float))))

# Weighting A10
wntio10_Guangdong = np.matmul(np.matrix(ntio10_Guangdong),np.matrix(np.diag(np.array(newemp10.loc["Guangdong","1":"9"].div(newemp10.loc["Guangdong","Total"]), dtype = np.float))))
wntio10_Jiangsu = np.matmul(np.matrix(ntio10_Jiangsu),np.matrix(np.diag(np.array(newemp10.loc["Jiangsu","1":"9"].div(newemp10.loc["Jiangsu","Total"]), dtype = np.float))))
wntio10_Gansu = np.matmul(np.matrix(ntio10_Gansu),np.matrix(np.diag(np.array(newemp10.loc["Gansu","1":"9"].div(newemp10.loc["Gansu","Total"]), dtype = np.float))))
wntio10_Qinghai = np.matmul(np.matrix(ntio10_Qinghai),np.matrix(np.diag(np.array(newemp10.loc["Qinghai","1":"9"].div(newemp10.loc["Qinghai","Total"]), dtype = np.float))))

# Weighting A11
wntio11_Guangdong = np.matmul(np.matrix(ntio11_Guangdong),np.matrix(np.diag(np.array(newemp11.loc["Guangdong","1":"9"].div(newemp11.loc["Guangdong","Total"]), dtype = np.float))))
wntio11_Jiangsu = np.matmul(np.matrix(ntio11_Jiangsu),np.matrix(np.diag(np.array(newemp11.loc["Jiangsu","1":"9"].div(newemp11.loc["Jiangsu","Total"]), dtype = np.float))))
wntio11_Gansu = np.matmul(np.matrix(ntio11_Gansu),np.matrix(np.diag(np.array(newemp11.loc["Gansu","1":"9"].div(newemp11.loc["Gansu","Total"]), dtype = np.float))))
wntio11_Qinghai = np.matmul(np.matrix(ntio11_Qinghai),np.matrix(np.diag(np.array(newemp11.loc["Qinghai","1":"9"].div(newemp11.loc["Qinghai","Total"]), dtype = np.float))))


#diagonal SLQ, regional
SLQ_emp98_Guangdong = pd.DataFrame(np.diag(np.array(list(SLQ_emp98.loc['Guangdong',:]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())
SLQ_emp98_Jiangsu = pd.DataFrame(np.diag(np.array(list(SLQ_emp98.loc['Jiangsu',:]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())
SLQ_emp98_Gansu = pd.DataFrame(np.diag(np.array(list(SLQ_emp98.loc['Gansu',:]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())
SLQ_emp98_Qinghai = pd.DataFrame(np.diag(np.array(list(SLQ_emp98.loc['Qinghai',:]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())

SLQ_emp99_Guangdong = pd.DataFrame(np.diag(np.array(list(SLQ_emp99.loc['Guangdong',:]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())
SLQ_emp99_Jiangsu = pd.DataFrame(np.diag(np.array(list(SLQ_emp99.loc['Jiangsu',:]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())
SLQ_emp99_Gansu = pd.DataFrame(np.diag(np.array(list(SLQ_emp99.loc['Gansu',:]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())
SLQ_emp99_Qinghai = pd.DataFrame(np.diag(np.array(list(SLQ_emp99.loc['Qinghai',:]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())

SLQ_emp00_Guangdong = pd.DataFrame(np.diag(np.array(list(SLQ_emp00.loc['Guangdong',:]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())
SLQ_emp00_Jiangsu = pd.DataFrame(np.diag(np.array(list(SLQ_emp00.loc['Jiangsu',:]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())
SLQ_emp00_Gansu = pd.DataFrame(np.diag(np.array(list(SLQ_emp00.loc['Gansu',:]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())
SLQ_emp00_Qinghai = pd.DataFrame(np.diag(np.array(list(SLQ_emp00.loc['Qinghai',:]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())

SLQ_emp01_Guangdong = pd.DataFrame(np.diag(np.array(list(SLQ_emp01.loc['Guangdong',:]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())
SLQ_emp01_Jiangsu = pd.DataFrame(np.diag(np.array(list(SLQ_emp01.loc['Jiangsu',:]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())
SLQ_emp01_Gansu = pd.DataFrame(np.diag(np.array(list(SLQ_emp01.loc['Gansu',:]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())
SLQ_emp01_Qinghai = pd.DataFrame(np.diag(np.array(list(SLQ_emp01.loc['Qinghai',:]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())

SLQ_emp02_Guangdong = pd.DataFrame(np.diag(np.array(list(SLQ_emp02.loc['Guangdong',:]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())
SLQ_emp02_Jiangsu = pd.DataFrame(np.diag(np.array(list(SLQ_emp02.loc['Jiangsu',:]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())
SLQ_emp02_Gansu = pd.DataFrame(np.diag(np.array(list(SLQ_emp02.loc['Gansu',:]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())
SLQ_emp02_Qinghai = pd.DataFrame(np.diag(np.array(list(SLQ_emp02.loc['Qinghai',:]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())

SLQ_emp03_Guangdong = pd.DataFrame(np.diag(np.array(list(SLQ_emp03.loc['Guangdong',:]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())
SLQ_emp03_Jiangsu = pd.DataFrame(np.diag(np.array(list(SLQ_emp03.loc['Jiangsu',:]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())
SLQ_emp03_Gansu = pd.DataFrame(np.diag(np.array(list(SLQ_emp03.loc['Gansu',:]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())
SLQ_emp03_Qinghai = pd.DataFrame(np.diag(np.array(list(SLQ_emp03.loc['Qinghai',:]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())

SLQ_emp04_Guangdong = pd.DataFrame(np.diag(np.array(list(SLQ_emp04.loc['Guangdong',:]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())
SLQ_emp04_Jiangsu = pd.DataFrame(np.diag(np.array(list(SLQ_emp04.loc['Jiangsu',:]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())
SLQ_emp04_Gansu = pd.DataFrame(np.diag(np.array(list(SLQ_emp04.loc['Gansu',:]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())
SLQ_emp04_Qinghai = pd.DataFrame(np.diag(np.array(list(SLQ_emp04.loc['Qinghai',:]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())

SLQ_emp05_Guangdong = pd.DataFrame(np.diag(np.array(list(SLQ_emp05.loc['Guangdong',:]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())
SLQ_emp05_Jiangsu = pd.DataFrame(np.diag(np.array(list(SLQ_emp05.loc['Jiangsu',:]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())
SLQ_emp05_Gansu = pd.DataFrame(np.diag(np.array(list(SLQ_emp05.loc['Gansu',:]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())
SLQ_emp05_Qinghai = pd.DataFrame(np.diag(np.array(list(SLQ_emp05.loc['Qinghai',:]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())

SLQ_emp06_Guangdong = pd.DataFrame(np.diag(np.array(list(SLQ_emp06.loc['Guangdong',:]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())
SLQ_emp06_Jiangsu = pd.DataFrame(np.diag(np.array(list(SLQ_emp06.loc['Jiangsu',:]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())
SLQ_emp06_Gansu = pd.DataFrame(np.diag(np.array(list(SLQ_emp06.loc['Gansu',:]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())
SLQ_emp06_Qinghai = pd.DataFrame(np.diag(np.array(list(SLQ_emp06.loc['Qinghai',:]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())

SLQ_emp07_Guangdong = pd.DataFrame(np.diag(np.array(list(SLQ_emp07.loc['Guangdong',:]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())
SLQ_emp07_Jiangsu = pd.DataFrame(np.diag(np.array(list(SLQ_emp07.loc['Jiangsu',:]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())
SLQ_emp07_Gansu = pd.DataFrame(np.diag(np.array(list(SLQ_emp07.loc['Gansu',:]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())
SLQ_emp07_Qinghai = pd.DataFrame(np.diag(np.array(list(SLQ_emp07.loc['Qinghai',:]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())

SLQ_emp08_Guangdong = pd.DataFrame(np.diag(np.array(list(SLQ_emp08.loc['Guangdong',:]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())
SLQ_emp08_Jiangsu = pd.DataFrame(np.diag(np.array(list(SLQ_emp08.loc['Jiangsu',:]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())
SLQ_emp08_Gansu = pd.DataFrame(np.diag(np.array(list(SLQ_emp08.loc['Gansu',:]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())
SLQ_emp08_Qinghai = pd.DataFrame(np.diag(np.array(list(SLQ_emp08.loc['Qinghai',:]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())

SLQ_emp09_Guangdong = pd.DataFrame(np.diag(np.array(list(SLQ_emp09.loc['Guangdong',:]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())
SLQ_emp09_Jiangsu = pd.DataFrame(np.diag(np.array(list(SLQ_emp09.loc['Jiangsu',:]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())
SLQ_emp09_Gansu = pd.DataFrame(np.diag(np.array(list(SLQ_emp09.loc['Gansu',:]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())
SLQ_emp09_Qinghai = pd.DataFrame(np.diag(np.array(list(SLQ_emp09.loc['Qinghai',:]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())

SLQ_emp10_Guangdong = pd.DataFrame(np.diag(np.array(list(SLQ_emp10.loc['Guangdong',:]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())
SLQ_emp10_Jiangsu = pd.DataFrame(np.diag(np.array(list(SLQ_emp10.loc['Jiangsu',:]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())
SLQ_emp10_Gansu = pd.DataFrame(np.diag(np.array(list(SLQ_emp10.loc['Gansu',:]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())
SLQ_emp10_Qinghai = pd.DataFrame(np.diag(np.array(list(SLQ_emp10.loc['Qinghai',:]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())

SLQ_emp11_Guangdong = pd.DataFrame(np.diag(np.array(list(SLQ_emp11.loc['Guangdong',:]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())
SLQ_emp11_Jiangsu = pd.DataFrame(np.diag(np.array(list(SLQ_emp11.loc['Jiangsu',:]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())
SLQ_emp11_Gansu = pd.DataFrame(np.diag(np.array(list(SLQ_emp11.loc['Gansu',:]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())
SLQ_emp11_Qinghai = pd.DataFrame(np.diag(np.array(list(SLQ_emp11.loc['Qinghai',:]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())

#removing SLQ >1
SLQ_emp98_Guangdong[:] = SLQ_emp98_Guangdong[:].apply(lambda x: [y if y <= 1 else 1 for y in x])
SLQ_emp98_Jiangsu[:] = SLQ_emp98_Jiangsu[:].apply(lambda x: [y if y <= 1 else 1 for y in x])
SLQ_emp98_Gansu[:] = SLQ_emp98_Gansu[:].apply(lambda x: [y if y <= 1 else 1 for y in x])
SLQ_emp98_Qinghai[:] = SLQ_emp98_Qinghai[:].apply(lambda x: [y if y <= 1 else 1 for y in x])

SLQ_emp99_Guangdong[:] = SLQ_emp99_Guangdong[:].apply(lambda x: [y if y <= 1 else 1 for y in x])
SLQ_emp99_Jiangsu[:] = SLQ_emp99_Jiangsu[:].apply(lambda x: [y if y <= 1 else 1 for y in x])
SLQ_emp99_Gansu[:] = SLQ_emp99_Gansu[:].apply(lambda x: [y if y <= 1 else 1 for y in x])
SLQ_emp99_Qinghai[:] = SLQ_emp99_Qinghai[:].apply(lambda x: [y if y <= 1 else 1 for y in x])

SLQ_emp00_Guangdong[:] = SLQ_emp00_Guangdong[:].apply(lambda x: [y if y <= 1 else 1 for y in x])
SLQ_emp00_Jiangsu[:] = SLQ_emp00_Jiangsu[:].apply(lambda x: [y if y <= 1 else 1 for y in x])
SLQ_emp00_Gansu[:] = SLQ_emp00_Gansu[:].apply(lambda x: [y if y <= 1 else 1 for y in x])
SLQ_emp00_Qinghai[:] = SLQ_emp00_Qinghai[:].apply(lambda x: [y if y <= 1 else 1 for y in x])

SLQ_emp01_Guangdong[:] = SLQ_emp01_Guangdong[:].apply(lambda x: [y if y <= 1 else 1 for y in x])
SLQ_emp01_Jiangsu[:] = SLQ_emp01_Jiangsu[:].apply(lambda x: [y if y <= 1 else 1 for y in x])
SLQ_emp01_Gansu[:] = SLQ_emp01_Gansu[:].apply(lambda x: [y if y <= 1 else 1 for y in x])
SLQ_emp01_Qinghai[:] = SLQ_emp01_Qinghai[:].apply(lambda x: [y if y <= 1 else 1 for y in x])

SLQ_emp02_Guangdong[:] = SLQ_emp02_Guangdong[:].apply(lambda x: [y if y <= 1 else 1 for y in x])
SLQ_emp02_Jiangsu[:] = SLQ_emp02_Jiangsu[:].apply(lambda x: [y if y <= 1 else 1 for y in x])
SLQ_emp02_Gansu[:] = SLQ_emp02_Gansu[:].apply(lambda x: [y if y <= 1 else 1 for y in x])
SLQ_emp02_Qinghai[:] = SLQ_emp02_Qinghai[:].apply(lambda x: [y if y <= 1 else 1 for y in x])

SLQ_emp03_Guangdong[:] = SLQ_emp03_Guangdong[:].apply(lambda x: [y if y <= 1 else 1 for y in x])
SLQ_emp03_Jiangsu[:] = SLQ_emp03_Jiangsu[:].apply(lambda x: [y if y <= 1 else 1 for y in x])
SLQ_emp03_Gansu[:] = SLQ_emp03_Gansu[:].apply(lambda x: [y if y <= 1 else 1 for y in x])
SLQ_emp03_Qinghai[:] = SLQ_emp03_Qinghai[:].apply(lambda x: [y if y <= 1 else 1 for y in x])

SLQ_emp04_Guangdong[:] = SLQ_emp04_Guangdong[:].apply(lambda x: [y if y <= 1 else 1 for y in x])
SLQ_emp04_Jiangsu[:] = SLQ_emp04_Jiangsu[:].apply(lambda x: [y if y <= 1 else 1 for y in x])
SLQ_emp04_Gansu[:] = SLQ_emp04_Gansu[:].apply(lambda x: [y if y <= 1 else 1 for y in x])
SLQ_emp04_Qinghai[:] = SLQ_emp04_Qinghai[:].apply(lambda x: [y if y <= 1 else 1 for y in x])

SLQ_emp05_Guangdong[:] = SLQ_emp05_Guangdong[:].apply(lambda x: [y if y <= 1 else 1 for y in x])
SLQ_emp05_Jiangsu[:] = SLQ_emp05_Jiangsu[:].apply(lambda x: [y if y <= 1 else 1 for y in x])
SLQ_emp05_Gansu[:] = SLQ_emp05_Gansu[:].apply(lambda x: [y if y <= 1 else 1 for y in x])
SLQ_emp05_Qinghai[:] = SLQ_emp05_Qinghai[:].apply(lambda x: [y if y <= 1 else 1 for y in x])

SLQ_emp06_Guangdong[:] = SLQ_emp06_Guangdong[:].apply(lambda x: [y if y <= 1 else 1 for y in x])
SLQ_emp06_Jiangsu[:] = SLQ_emp06_Jiangsu[:].apply(lambda x: [y if y <= 1 else 1 for y in x])
SLQ_emp06_Gansu[:] = SLQ_emp06_Gansu[:].apply(lambda x: [y if y <= 1 else 1 for y in x])
SLQ_emp06_Qinghai[:] = SLQ_emp06_Qinghai[:].apply(lambda x: [y if y <= 1 else 1 for y in x])

SLQ_emp07_Guangdong[:] = SLQ_emp07_Guangdong[:].apply(lambda x: [y if y <= 1 else 1 for y in x])
SLQ_emp07_Jiangsu[:] = SLQ_emp07_Jiangsu[:].apply(lambda x: [y if y <= 1 else 1 for y in x])
SLQ_emp07_Gansu[:] = SLQ_emp07_Gansu[:].apply(lambda x: [y if y <= 1 else 1 for y in x])
SLQ_emp07_Qinghai[:] = SLQ_emp07_Qinghai[:].apply(lambda x: [y if y <= 1 else 1 for y in x])

SLQ_emp08_Guangdong[:] = SLQ_emp08_Guangdong[:].apply(lambda x: [y if y <= 1 else 1 for y in x])
SLQ_emp08_Jiangsu[:] = SLQ_emp08_Jiangsu[:].apply(lambda x: [y if y <= 1 else 1 for y in x])
SLQ_emp08_Gansu[:] = SLQ_emp08_Gansu[:].apply(lambda x: [y if y <= 1 else 1 for y in x])
SLQ_emp08_Qinghai[:] = SLQ_emp08_Qinghai[:].apply(lambda x: [y if y <= 1 else 1 for y in x])

SLQ_emp09_Guangdong[:] = SLQ_emp09_Guangdong[:].apply(lambda x: [y if y <= 1 else 1 for y in x])
SLQ_emp09_Jiangsu[:] = SLQ_emp09_Jiangsu[:].apply(lambda x: [y if y <= 1 else 1 for y in x])
SLQ_emp09_Gansu[:] = SLQ_emp09_Gansu[:].apply(lambda x: [y if y <= 1 else 1 for y in x])
SLQ_emp09_Qinghai[:] = SLQ_emp09_Qinghai[:].apply(lambda x: [y if y <= 1 else 1 for y in x])

SLQ_emp10_Guangdong[:] = SLQ_emp10_Guangdong[:].apply(lambda x: [y if y <= 1 else 1 for y in x])
SLQ_emp10_Jiangsu[:] = SLQ_emp10_Jiangsu[:].apply(lambda x: [y if y <= 1 else 1 for y in x])
SLQ_emp10_Gansu[:] = SLQ_emp10_Gansu[:].apply(lambda x: [y if y <= 1 else 1 for y in x])
SLQ_emp10_Qinghai[:] = SLQ_emp10_Qinghai[:].apply(lambda x: [y if y <= 1 else 1 for y in x])

SLQ_emp11_Guangdong[:] = SLQ_emp11_Guangdong[:].apply(lambda x: [y if y <= 1 else 1 for y in x])
SLQ_emp11_Jiangsu[:] = SLQ_emp11_Jiangsu[:].apply(lambda x: [y if y <= 1 else 1 for y in x])
SLQ_emp11_Gansu[:] = SLQ_emp11_Gansu[:].apply(lambda x: [y if y <= 1 else 1 for y in x])
SLQ_emp11_Qinghai[:] = SLQ_emp11_Qinghai[:].apply(lambda x: [y if y <= 1 else 1 for y in x])


#National Employment matrices and Invert them
Nemp98 = pd.DataFrame(np.linalg.inv(np.diag(np.array(list(newemp98.loc["1998","1":"9"]), dtype = np.float))),columns = de98B.keys(), index = de98B.keys())
Nemp99 = pd.DataFrame(np.linalg.inv(np.diag(np.array(list(newemp99.loc["1999","1":"9"]), dtype = np.float))),columns = de98B.keys(), index = de98B.keys())
Nemp00 = pd.DataFrame(np.linalg.inv(np.diag(np.array(list(newemp00.loc["2000","1":"9"]), dtype = np.float))),columns = de98B.keys(), index = de98B.keys())
Nemp01 = pd.DataFrame(np.linalg.inv(np.diag(np.array(list(newemp01.loc["2001","1":"9"]), dtype = np.float))),columns = de98B.keys(), index = de98B.keys())
Nemp02 = pd.DataFrame(np.linalg.inv(np.diag(np.array(list(newemp02.loc["2002","1":"9"]), dtype = np.float))),columns = de98B.keys(), index = de98B.keys())
Nemp03 = pd.DataFrame(np.linalg.inv(np.diag(np.array(list(newemp03.loc["2003","1":"9"]), dtype = np.float))),columns = de98B.keys(), index = de98B.keys())
Nemp04 = pd.DataFrame(np.linalg.inv(np.diag(np.array(list(newemp04.loc["2004","1":"9"]), dtype = np.float))),columns = de98B.keys(), index = de98B.keys())
Nemp05 = pd.DataFrame(np.linalg.inv(np.diag(np.array(list(newemp05.loc["2005","1":"9"]), dtype = np.float))),columns = de98B.keys(), index = de98B.keys())
Nemp06 = pd.DataFrame(np.linalg.inv(np.diag(np.array(list(newemp06.loc["2006","1":"9"]), dtype = np.float))),columns = de98B.keys(), index = de98B.keys())
Nemp07 = pd.DataFrame(np.linalg.inv(np.diag(np.array(list(newemp07.loc["2007","1":"9"]), dtype = np.float))),columns = de98B.keys(), index = de98B.keys())
Nemp08 = pd.DataFrame(np.linalg.inv(np.diag(np.array(list(newemp08.loc["2008","1":"9"]), dtype = np.float))),columns = de98B.keys(), index = de98B.keys())
Nemp09 = pd.DataFrame(np.linalg.inv(np.diag(np.array(list(newemp09.loc["2009","1":"9"]), dtype = np.float))),columns = de98B.keys(), index = de98B.keys())
Nemp10 = pd.DataFrame(np.linalg.inv(np.diag(np.array(list(newemp10.loc["2010","1":"9"]), dtype = np.float))),columns = de98B.keys(), index = de98B.keys())
Nemp11 = pd.DataFrame(np.linalg.inv(np.diag(np.array(list(newemp11.loc["2011","1":"9"]), dtype = np.float))),columns = de98B.keys(), index = de98B.keys())

#Regional Employment matrices
#Gansu
Remp98_Gansu = pd.DataFrame(np.diag(np.array(list(newemp98.loc["Gansu","1":"9"]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())
Remp99_Gansu = pd.DataFrame(np.diag(np.array(list(newemp99.loc["Gansu","1":"9"]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())
Remp00_Gansu = pd.DataFrame(np.diag(np.array(list(newemp00.loc["Gansu","1":"9"]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())
Remp01_Gansu = pd.DataFrame(np.diag(np.array(list(newemp01.loc["Gansu","1":"9"]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())
Remp02_Gansu = pd.DataFrame(np.diag(np.array(list(newemp02.loc["Gansu","1":"9"]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())
Remp03_Gansu = pd.DataFrame(np.diag(np.array(list(newemp03.loc["Gansu","1":"9"]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())
Remp04_Gansu = pd.DataFrame(np.diag(np.array(list(newemp04.loc["Gansu","1":"9"]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())
Remp05_Gansu = pd.DataFrame(np.diag(np.array(list(newemp05.loc["Gansu","1":"9"]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())
Remp06_Gansu = pd.DataFrame(np.diag(np.array(list(newemp06.loc["Gansu","1":"9"]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())
Remp07_Gansu = pd.DataFrame(np.diag(np.array(list(newemp07.loc["Gansu","1":"9"]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())
Remp08_Gansu = pd.DataFrame(np.diag(np.array(list(newemp08.loc["Gansu","1":"9"]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())
Remp09_Gansu = pd.DataFrame(np.diag(np.array(list(newemp09.loc["Gansu","1":"9"]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())
Remp10_Gansu = pd.DataFrame(np.diag(np.array(list(newemp10.loc["Gansu","1":"9"]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())
Remp11_Gansu = pd.DataFrame(np.diag(np.array(list(newemp11.loc["Gansu","1":"9"]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())

#Guangdong
Remp98_Guangdong = pd.DataFrame(np.diag(np.array(list(newemp98.loc["Guangdong","1":"9"]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())
Remp99_Guangdong = pd.DataFrame(np.diag(np.array(list(newemp99.loc["Guangdong","1":"9"]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())
Remp00_Guangdong = pd.DataFrame(np.diag(np.array(list(newemp00.loc["Guangdong","1":"9"]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())
Remp01_Guangdong = pd.DataFrame(np.diag(np.array(list(newemp01.loc["Guangdong","1":"9"]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())
Remp02_Guangdong = pd.DataFrame(np.diag(np.array(list(newemp02.loc["Guangdong","1":"9"]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())
Remp03_Guangdong = pd.DataFrame(np.diag(np.array(list(newemp03.loc["Guangdong","1":"9"]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())
Remp04_Guangdong = pd.DataFrame(np.diag(np.array(list(newemp04.loc["Guangdong","1":"9"]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())
Remp05_Guangdong = pd.DataFrame(np.diag(np.array(list(newemp05.loc["Guangdong","1":"9"]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())
Remp06_Guangdong = pd.DataFrame(np.diag(np.array(list(newemp06.loc["Guangdong","1":"9"]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())
Remp07_Guangdong = pd.DataFrame(np.diag(np.array(list(newemp07.loc["Guangdong","1":"9"]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())
Remp08_Guangdong = pd.DataFrame(np.diag(np.array(list(newemp08.loc["Guangdong","1":"9"]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())
Remp09_Guangdong = pd.DataFrame(np.diag(np.array(list(newemp09.loc["Guangdong","1":"9"]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())
Remp10_Guangdong = pd.DataFrame(np.diag(np.array(list(newemp10.loc["Guangdong","1":"9"]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())
Remp11_Guangdong = pd.DataFrame(np.diag(np.array(list(newemp11.loc["Guangdong","1":"9"]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())

#Jiangsu
Remp98_Jiangsu = pd.DataFrame(np.diag(np.array(list(newemp98.loc["Jiangsu","1":"9"]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())
Remp99_Jiangsu = pd.DataFrame(np.diag(np.array(list(newemp99.loc["Jiangsu","1":"9"]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())
Remp00_Jiangsu = pd.DataFrame(np.diag(np.array(list(newemp00.loc["Jiangsu","1":"9"]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())
Remp01_Jiangsu = pd.DataFrame(np.diag(np.array(list(newemp01.loc["Jiangsu","1":"9"]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())
Remp02_Jiangsu = pd.DataFrame(np.diag(np.array(list(newemp02.loc["Jiangsu","1":"9"]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())
Remp03_Jiangsu = pd.DataFrame(np.diag(np.array(list(newemp03.loc["Jiangsu","1":"9"]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())
Remp04_Jiangsu = pd.DataFrame(np.diag(np.array(list(newemp04.loc["Jiangsu","1":"9"]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())
Remp05_Jiangsu = pd.DataFrame(np.diag(np.array(list(newemp05.loc["Jiangsu","1":"9"]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())
Remp06_Jiangsu = pd.DataFrame(np.diag(np.array(list(newemp06.loc["Jiangsu","1":"9"]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())
Remp07_Jiangsu = pd.DataFrame(np.diag(np.array(list(newemp07.loc["Jiangsu","1":"9"]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())
Remp08_Jiangsu = pd.DataFrame(np.diag(np.array(list(newemp08.loc["Jiangsu","1":"9"]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())
Remp09_Jiangsu = pd.DataFrame(np.diag(np.array(list(newemp09.loc["Jiangsu","1":"9"]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())
Remp10_Jiangsu = pd.DataFrame(np.diag(np.array(list(newemp10.loc["Jiangsu","1":"9"]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())
Remp11_Jiangsu = pd.DataFrame(np.diag(np.array(list(newemp11.loc["Jiangsu","1":"9"]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())

#Qinghai
Remp98_Qinghai = pd.DataFrame(np.diag(np.array(list(newemp98.loc["Qinghai","1":"9"]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())
Remp99_Qinghai = pd.DataFrame(np.diag(np.array(list(newemp99.loc["Qinghai","1":"9"]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())
Remp00_Qinghai = pd.DataFrame(np.diag(np.array(list(newemp00.loc["Qinghai","1":"9"]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())
Remp01_Qinghai = pd.DataFrame(np.diag(np.array(list(newemp01.loc["Qinghai","1":"9"]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())
Remp02_Qinghai = pd.DataFrame(np.diag(np.array(list(newemp02.loc["Qinghai","1":"9"]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())
Remp03_Qinghai = pd.DataFrame(np.diag(np.array(list(newemp03.loc["Qinghai","1":"9"]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())
Remp04_Qinghai = pd.DataFrame(np.diag(np.array(list(newemp04.loc["Qinghai","1":"9"]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())
Remp05_Qinghai = pd.DataFrame(np.diag(np.array(list(newemp05.loc["Qinghai","1":"9"]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())
Remp06_Qinghai = pd.DataFrame(np.diag(np.array(list(newemp06.loc["Qinghai","1":"9"]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())
Remp07_Qinghai = pd.DataFrame(np.diag(np.array(list(newemp07.loc["Qinghai","1":"9"]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())
Remp08_Qinghai = pd.DataFrame(np.diag(np.array(list(newemp08.loc["Qinghai","1":"9"]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())
Remp09_Qinghai = pd.DataFrame(np.diag(np.array(list(newemp09.loc["Qinghai","1":"9"]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())
Remp10_Qinghai = pd.DataFrame(np.diag(np.array(list(newemp10.loc["Qinghai","1":"9"]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())
Remp11_Qinghai = pd.DataFrame(np.diag(np.array(list(newemp11.loc["Qinghai","1":"9"]), dtype = np.float)),columns = de98B.keys(), index = de98B.keys())


# Estimating regional Outputs
# regionalisation of X98
RX98_Guangdong = pd.DataFrame(np.matmul(np.matmul(np.matmul(np.matrix(Remp98_Guangdong),np.matrix(Nemp98)),np.matrix(SLQ_emp98_Guangdong)),np.matrix(newY98.transpose())))
RX98_Jiangsu = pd.DataFrame(np.matmul(np.matmul(np.matmul(np.matrix(Remp98_Jiangsu),np.matrix(Nemp98)),np.matrix(SLQ_emp98_Jiangsu)),np.matrix(newY98.transpose())))
RX98_Gansu = pd.DataFrame(np.matmul(np.matmul(np.matmul(np.matrix(Remp98_Gansu),np.matrix(Nemp98)),np.matrix(SLQ_emp98_Gansu)),np.matrix(newY98.transpose())))
RX98_Qinghai = pd.DataFrame(np.matmul(np.matmul(np.matmul(np.matrix(Remp98_Qinghai),np.matrix(Nemp98)),np.matrix(SLQ_emp98_Qinghai)),np.matrix(newY98.transpose())))

# regionalisation of X99
RX99_Guangdong = pd.DataFrame(np.matmul(np.matmul(np.matmul(np.matrix(Remp99_Guangdong),np.matrix(Nemp99)),np.matrix(SLQ_emp99_Guangdong)),np.matrix(newY99.transpose())))
RX99_Jiangsu = pd.DataFrame(np.matmul(np.matmul(np.matmul(np.matrix(Remp99_Jiangsu),np.matrix(Nemp99)),np.matrix(SLQ_emp99_Jiangsu)),np.matrix(newY99.transpose())))
RX99_Gansu = pd.DataFrame(np.matmul(np.matmul(np.matmul(np.matrix(Remp99_Gansu),np.matrix(Nemp99)),np.matrix(SLQ_emp99_Gansu)),np.matrix(newY99.transpose())))
RX99_Qinghai = pd.DataFrame(np.matmul(np.matmul(np.matmul(np.matrix(Remp99_Qinghai),np.matrix(Nemp99)),np.matrix(SLQ_emp99_Qinghai)),np.matrix(newY99.transpose())))

# regionalisation of X00
RX00_Guangdong = pd.DataFrame(np.matmul(np.matmul(np.matmul(np.matrix(Remp00_Guangdong),np.matrix(Nemp00)),np.matrix(SLQ_emp00_Guangdong)),np.matrix(newY00.transpose())))
RX00_Jiangsu = pd.DataFrame(np.matmul(np.matmul(np.matmul(np.matrix(Remp00_Jiangsu),np.matrix(Nemp00)),np.matrix(SLQ_emp00_Jiangsu)),np.matrix(newY00.transpose())))
RX00_Gansu = pd.DataFrame(np.matmul(np.matmul(np.matmul(np.matrix(Remp00_Gansu),np.matrix(Nemp00)),np.matrix(SLQ_emp00_Gansu)),np.matrix(newY00.transpose())))
RX00_Qinghai = pd.DataFrame(np.matmul(np.matmul(np.matmul(np.matrix(Remp00_Qinghai),np.matrix(Nemp00)),np.matrix(SLQ_emp00_Qinghai)),np.matrix(newY00.transpose())))

# regionalisation of X01
RX01_Guangdong = pd.DataFrame(np.matmul(np.matmul(np.matmul(np.matrix(Remp01_Guangdong),np.matrix(Nemp01)),np.matrix(SLQ_emp01_Guangdong)),np.matrix(newY01.transpose())))
RX01_Jiangsu = pd.DataFrame(np.matmul(np.matmul(np.matmul(np.matrix(Remp01_Jiangsu),np.matrix(Nemp01)),np.matrix(SLQ_emp01_Jiangsu)),np.matrix(newY01.transpose())))
RX01_Gansu = pd.DataFrame(np.matmul(np.matmul(np.matmul(np.matrix(Remp01_Gansu),np.matrix(Nemp01)),np.matrix(SLQ_emp01_Gansu)),np.matrix(newY01.transpose())))
RX01_Qinghai = pd.DataFrame(np.matmul(np.matmul(np.matmul(np.matrix(Remp01_Qinghai),np.matrix(Nemp01)),np.matrix(SLQ_emp01_Qinghai)),np.matrix(newY01.transpose())))

# regionalisation of X02
RX02_Guangdong = pd.DataFrame(np.matmul(np.matmul(np.matmul(np.matrix(Remp02_Guangdong),np.matrix(Nemp02)),np.matrix(SLQ_emp02_Guangdong)),np.matrix(newY02.transpose())))
RX02_Jiangsu = pd.DataFrame(np.matmul(np.matmul(np.matmul(np.matrix(Remp02_Jiangsu),np.matrix(Nemp02)),np.matrix(SLQ_emp02_Jiangsu)),np.matrix(newY02.transpose())))
RX02_Gansu = pd.DataFrame(np.matmul(np.matmul(np.matmul(np.matrix(Remp02_Gansu),np.matrix(Nemp02)),np.matrix(SLQ_emp02_Gansu)),np.matrix(newY02.transpose())))
RX02_Qinghai = pd.DataFrame(np.matmul(np.matmul(np.matmul(np.matrix(Remp02_Qinghai),np.matrix(Nemp02)),np.matrix(SLQ_emp02_Qinghai)),np.matrix(newY02.transpose())))

# regionalisation of X03
RX03_Guangdong = pd.DataFrame(np.matmul(np.matmul(np.matmul(np.matrix(Remp03_Guangdong),np.matrix(Nemp03)),np.matrix(SLQ_emp03_Guangdong)),np.matrix(newY03.transpose())))
RX03_Jiangsu = pd.DataFrame(np.matmul(np.matmul(np.matmul(np.matrix(Remp03_Jiangsu),np.matrix(Nemp03)),np.matrix(SLQ_emp03_Jiangsu)),np.matrix(newY03.transpose())))
RX03_Gansu = pd.DataFrame(np.matmul(np.matmul(np.matmul(np.matrix(Remp03_Gansu),np.matrix(Nemp03)),np.matrix(SLQ_emp03_Gansu)),np.matrix(newY03.transpose())))
RX03_Qinghai = pd.DataFrame(np.matmul(np.matmul(np.matmul(np.matrix(Remp03_Qinghai),np.matrix(Nemp03)),np.matrix(SLQ_emp03_Qinghai)),np.matrix(newY03.transpose())))

# regionalisation of X04
RX04_Guangdong = pd.DataFrame(np.matmul(np.matmul(np.matmul(np.matrix(Remp04_Guangdong),np.matrix(Nemp04)),np.matrix(SLQ_emp04_Guangdong)),np.matrix(newY04.transpose())))
RX04_Jiangsu = pd.DataFrame(np.matmul(np.matmul(np.matmul(np.matrix(Remp04_Jiangsu),np.matrix(Nemp04)),np.matrix(SLQ_emp04_Jiangsu)),np.matrix(newY04.transpose())))
RX04_Gansu = pd.DataFrame(np.matmul(np.matmul(np.matmul(np.matrix(Remp04_Gansu),np.matrix(Nemp04)),np.matrix(SLQ_emp04_Gansu)),np.matrix(newY04.transpose())))
RX04_Qinghai = pd.DataFrame(np.matmul(np.matmul(np.matmul(np.matrix(Remp04_Qinghai),np.matrix(Nemp04)),np.matrix(SLQ_emp04_Qinghai)),np.matrix(newY04.transpose())))

# regionalisation of X05
RX05_Guangdong = pd.DataFrame(np.matmul(np.matmul(np.matmul(np.matrix(Remp05_Guangdong),np.matrix(Nemp05)),np.matrix(SLQ_emp05_Guangdong)),np.matrix(newY05.transpose())))
RX05_Jiangsu = pd.DataFrame(np.matmul(np.matmul(np.matmul(np.matrix(Remp05_Jiangsu),np.matrix(Nemp05)),np.matrix(SLQ_emp05_Jiangsu)),np.matrix(newY05.transpose())))
RX05_Gansu = pd.DataFrame(np.matmul(np.matmul(np.matmul(np.matrix(Remp05_Gansu),np.matrix(Nemp05)),np.matrix(SLQ_emp05_Gansu)),np.matrix(newY05.transpose())))
RX05_Qinghai = pd.DataFrame(np.matmul(np.matmul(np.matmul(np.matrix(Remp05_Qinghai),np.matrix(Nemp05)),np.matrix(SLQ_emp05_Qinghai)),np.matrix(newY05.transpose())))

# regionalisation of X06
RX06_Guangdong = pd.DataFrame(np.matmul(np.matmul(np.matmul(np.matrix(Remp06_Guangdong),np.matrix(Nemp06)),np.matrix(SLQ_emp06_Guangdong)),np.matrix(newY06.transpose())))
RX06_Jiangsu = pd.DataFrame(np.matmul(np.matmul(np.matmul(np.matrix(Remp06_Jiangsu),np.matrix(Nemp06)),np.matrix(SLQ_emp06_Jiangsu)),np.matrix(newY06.transpose())))
RX06_Gansu = pd.DataFrame(np.matmul(np.matmul(np.matmul(np.matrix(Remp06_Gansu),np.matrix(Nemp06)),np.matrix(SLQ_emp06_Gansu)),np.matrix(newY06.transpose())))
RX06_Qinghai = pd.DataFrame(np.matmul(np.matmul(np.matmul(np.matrix(Remp06_Qinghai),np.matrix(Nemp06)),np.matrix(SLQ_emp06_Qinghai)),np.matrix(newY06.transpose())))

# regionalisation of X07
RX07_Guangdong = pd.DataFrame(np.matmul(np.matmul(np.matmul(np.matrix(Remp07_Guangdong),np.matrix(Nemp00)),np.matrix(SLQ_emp07_Guangdong)),np.matrix(newY07.transpose())))
RX07_Jiangsu = pd.DataFrame(np.matmul(np.matmul(np.matmul(np.matrix(Remp07_Jiangsu),np.matrix(Nemp00)),np.matrix(SLQ_emp07_Jiangsu)),np.matrix(newY07.transpose())))
RX07_Gansu = pd.DataFrame(np.matmul(np.matmul(np.matmul(np.matrix(Remp07_Gansu),np.matrix(Nemp00)),np.matrix(SLQ_emp07_Gansu)),np.matrix(newY07.transpose())))
RX07_Qinghai = pd.DataFrame(np.matmul(np.matmul(np.matmul(np.matrix(Remp07_Qinghai),np.matrix(Nemp00)),np.matrix(SLQ_emp07_Qinghai)),np.matrix(newY07.transpose())))

# regionalisation of X08
RX08_Guangdong = pd.DataFrame(np.matmul(np.matmul(np.matmul(np.matrix(Remp08_Guangdong),np.matrix(Nemp08)),np.matrix(SLQ_emp08_Guangdong)),np.matrix(newY08.transpose())))
RX08_Jiangsu = pd.DataFrame(np.matmul(np.matmul(np.matmul(np.matrix(Remp08_Jiangsu),np.matrix(Nemp08)),np.matrix(SLQ_emp08_Jiangsu)),np.matrix(newY08.transpose())))
RX08_Gansu = pd.DataFrame(np.matmul(np.matmul(np.matmul(np.matrix(Remp08_Gansu),np.matrix(Nemp08)),np.matrix(SLQ_emp08_Gansu)),np.matrix(newY08.transpose())))
RX08_Qinghai = pd.DataFrame(np.matmul(np.matmul(np.matmul(np.matrix(Remp08_Qinghai),np.matrix(Nemp08)),np.matrix(SLQ_emp08_Qinghai)),np.matrix(newY08.transpose())))

# regionalisation of X09
RX09_Guangdong = pd.DataFrame(np.matmul(np.matmul(np.matmul(np.matrix(Remp09_Guangdong),np.matrix(Nemp09)),np.matrix(SLQ_emp09_Guangdong)),np.matrix(newY09.transpose())))
RX09_Jiangsu = pd.DataFrame(np.matmul(np.matmul(np.matmul(np.matrix(Remp09_Jiangsu),np.matrix(Nemp09)),np.matrix(SLQ_emp09_Jiangsu)),np.matrix(newY09.transpose())))
RX09_Gansu = pd.DataFrame(np.matmul(np.matmul(np.matmul(np.matrix(Remp09_Gansu),np.matrix(Nemp09)),np.matrix(SLQ_emp09_Gansu)),np.matrix(newY09.transpose())))
RX09_Qinghai = pd.DataFrame(np.matmul(np.matmul(np.matmul(np.matrix(Remp09_Qinghai),np.matrix(Nemp09)),np.matrix(SLQ_emp09_Qinghai)),np.matrix(newY09.transpose())))

# regionalisation of X10
RX10_Guangdong = pd.DataFrame(np.matmul(np.matmul(np.matmul(np.matrix(Remp10_Guangdong),np.matrix(Nemp10)),np.matrix(SLQ_emp10_Guangdong)),np.matrix(newY10.transpose())))
RX10_Jiangsu = pd.DataFrame(np.matmul(np.matmul(np.matmul(np.matrix(Remp10_Jiangsu),np.matrix(Nemp10)),np.matrix(SLQ_emp10_Jiangsu)),np.matrix(newY10.transpose())))
RX10_Gansu = pd.DataFrame(np.matmul(np.matmul(np.matmul(np.matrix(Remp10_Gansu),np.matrix(Nemp10)),np.matrix(SLQ_emp10_Gansu)),np.matrix(newY10.transpose())))
RX10_Qinghai = pd.DataFrame(np.matmul(np.matmul(np.matmul(np.matrix(Remp10_Qinghai),np.matrix(Nemp10)),np.matrix(SLQ_emp10_Qinghai)),np.matrix(newY10.transpose())))

# regionalisation of X11
RX11_Guangdong = pd.DataFrame(np.matmul(np.matmul(np.matmul(np.matrix(Remp11_Guangdong),np.matrix(Nemp11)),np.matrix(SLQ_emp11_Guangdong)),np.matrix(newY11.transpose())))
RX11_Jiangsu = pd.DataFrame(np.matmul(np.matmul(np.matmul(np.matrix(Remp11_Jiangsu),np.matrix(Nemp11)),np.matrix(SLQ_emp11_Jiangsu)),np.matrix(newY11.transpose())))
RX11_Gansu = pd.DataFrame(np.matmul(np.matmul(np.matmul(np.matrix(Remp11_Gansu),np.matrix(Nemp11)),np.matrix(SLQ_emp11_Gansu)),np.matrix(newY11.transpose())))
RX11_Qinghai = pd.DataFrame(np.matmul(np.matmul(np.matmul(np.matrix(Remp11_Qinghai),np.matrix(Nemp11)),np.matrix(SLQ_emp11_Qinghai)),np.matrix(newY11.transpose())))


#Regional Transactions Matrices
# regionalisation of Z98
Z98_Guangdong = pd.DataFrame(np.matmul(np.matrix(wntio98_Guangdong),np.diag(np.array(RX98_Guangdong.transpose())[0])))
Z98_Jiangsu = pd.DataFrame(np.matmul(np.matrix(wntio98_Jiangsu),np.diag(np.array(RX98_Jiangsu.transpose())[0])))
Z98_Gansu = pd.DataFrame(np.matmul(np.matrix(wntio98_Gansu),np.diag(np.array(RX98_Gansu.transpose())[0])))
Z98_Qinghai = pd.DataFrame(np.matmul(np.matrix(wntio98_Qinghai),np.diag(np.array(RX98_Qinghai.transpose())[0])))

# regionalisation of Z99
Z99_Guangdong = pd.DataFrame(np.matmul(np.matrix(wntio99_Guangdong),np.diag(np.array(RX99_Guangdong.transpose())[0])))
Z99_Jiangsu = pd.DataFrame(np.matmul(np.matrix(wntio99_Jiangsu),np.diag(np.array(RX99_Jiangsu.transpose())[0])))
Z99_Gansu = pd.DataFrame(np.matmul(np.matrix(wntio99_Gansu),np.diag(np.array(RX99_Gansu.transpose())[0])))
Z99_Qinghai = pd.DataFrame(np.matmul(np.matrix(wntio99_Qinghai),np.diag(np.array(RX99_Qinghai.transpose())[0])))

# regionalisation of Z00
Z00_Guangdong = pd.DataFrame(np.matmul(np.matrix(wntio00_Guangdong),np.diag(np.array(RX00_Guangdong.transpose())[0])))
Z00_Jiangsu = pd.DataFrame(np.matmul(np.matrix(wntio00_Jiangsu),np.diag(np.array(RX00_Jiangsu.transpose())[0])))
Z00_Gansu = pd.DataFrame(np.matmul(np.matrix(wntio00_Gansu),np.diag(np.array(RX00_Gansu.transpose())[0])))
Z00_Qinghai = pd.DataFrame(np.matmul(np.matrix(wntio00_Qinghai),np.diag(np.array(RX00_Qinghai.transpose())[0])))

# regionalisation of Z01
Z01_Guangdong = pd.DataFrame(np.matmul(np.matrix(wntio01_Guangdong),np.diag(np.array(RX01_Guangdong.transpose())[0])))
Z01_Jiangsu = pd.DataFrame(np.matmul(np.matrix(wntio01_Jiangsu),np.diag(np.array(RX01_Jiangsu.transpose())[0])))
Z01_Gansu = pd.DataFrame(np.matmul(np.matrix(wntio01_Gansu),np.diag(np.array(RX01_Gansu.transpose())[0])))
Z01_Qinghai = pd.DataFrame(np.matmul(np.matrix(wntio01_Qinghai),np.diag(np.array(RX01_Qinghai.transpose())[0])))

# regionalisation of Z02
Z02_Guangdong = pd.DataFrame(np.matmul(np.matrix(wntio02_Guangdong),np.diag(np.array(RX02_Guangdong.transpose())[0])))
Z02_Jiangsu = pd.DataFrame(np.matmul(np.matrix(wntio02_Jiangsu),np.diag(np.array(RX02_Jiangsu.transpose())[0])))
Z02_Gansu = pd.DataFrame(np.matmul(np.matrix(wntio02_Gansu),np.diag(np.array(RX02_Gansu.transpose())[0])))
Z02_Qinghai = pd.DataFrame(np.matmul(np.matrix(wntio02_Qinghai),np.diag(np.array(RX02_Qinghai.transpose())[0])))

# regionalisation of Z03
Z03_Guangdong = pd.DataFrame(np.matmul(np.matrix(wntio03_Guangdong),np.diag(np.array(RX03_Guangdong.transpose())[0])))
Z03_Jiangsu = pd.DataFrame(np.matmul(np.matrix(wntio03_Jiangsu),np.diag(np.array(RX03_Jiangsu.transpose())[0])))
Z03_Gansu = pd.DataFrame(np.matmul(np.matrix(wntio03_Gansu),np.diag(np.array(RX03_Gansu.transpose())[0])))
Z03_Qinghai = pd.DataFrame(np.matmul(np.matrix(wntio03_Qinghai),np.diag(np.array(RX03_Qinghai.transpose())[0])))

# regionalisation of Z04
Z04_Guangdong = pd.DataFrame(np.matmul(np.matrix(wntio04_Guangdong),np.diag(np.array(RX04_Guangdong.transpose())[0])))
Z04_Jiangsu = pd.DataFrame(np.matmul(np.matrix(wntio04_Jiangsu),np.diag(np.array(RX04_Jiangsu.transpose())[0])))
Z04_Gansu = pd.DataFrame(np.matmul(np.matrix(wntio04_Gansu),np.diag(np.array(RX04_Gansu.transpose())[0])))
Z04_Qinghai = pd.DataFrame(np.matmul(np.matrix(wntio04_Qinghai),np.diag(np.array(RX04_Qinghai.transpose())[0])))

# regionalisation of Z05
Z05_Guangdong = pd.DataFrame(np.matmul(np.matrix(wntio05_Guangdong),np.diag(np.array(RX05_Guangdong.transpose())[0])))
Z05_Jiangsu = pd.DataFrame(np.matmul(np.matrix(wntio05_Jiangsu),np.diag(np.array(RX05_Jiangsu.transpose())[0])))
Z05_Gansu = pd.DataFrame(np.matmul(np.matrix(wntio05_Gansu),np.diag(np.array(RX05_Gansu.transpose())[0])))
Z05_Qinghai = pd.DataFrame(np.matmul(np.matrix(wntio05_Qinghai),np.diag(np.array(RX05_Qinghai.transpose())[0])))

# regionalisation of Z06
Z06_Guangdong = pd.DataFrame(np.matmul(np.matrix(wntio06_Guangdong),np.diag(np.array(RX06_Guangdong.transpose())[0])))
Z06_Jiangsu = pd.DataFrame(np.matmul(np.matrix(wntio06_Jiangsu),np.diag(np.array(RX06_Jiangsu.transpose())[0])))
Z06_Gansu = pd.DataFrame(np.matmul(np.matrix(wntio06_Gansu),np.diag(np.array(RX06_Gansu.transpose())[0])))
Z06_Qinghai = pd.DataFrame(np.matmul(np.matrix(wntio06_Qinghai),np.diag(np.array(RX06_Qinghai.transpose())[0])))

# regionalisation of Z07
Z07_Guangdong = pd.DataFrame(np.matmul(np.matrix(wntio07_Guangdong),np.diag(np.array(RX07_Guangdong.transpose())[0])))
Z07_Jiangsu = pd.DataFrame(np.matmul(np.matrix(wntio07_Jiangsu),np.diag(np.array(RX07_Jiangsu.transpose())[0])))
Z07_Gansu = pd.DataFrame(np.matmul(np.matrix(wntio07_Gansu),np.diag(np.array(RX07_Gansu.transpose())[0])))
Z07_Qinghai = pd.DataFrame(np.matmul(np.matrix(wntio07_Qinghai),np.diag(np.array(RX07_Qinghai.transpose())[0])))

# regionalisation of Z08
Z08_Guangdong = pd.DataFrame(np.matmul(np.matrix(wntio08_Guangdong),np.diag(np.array(RX08_Guangdong.transpose())[0])))
Z08_Jiangsu = pd.DataFrame(np.matmul(np.matrix(wntio08_Jiangsu),np.diag(np.array(RX08_Jiangsu.transpose())[0])))
Z08_Gansu = pd.DataFrame(np.matmul(np.matrix(wntio08_Gansu),np.diag(np.array(RX08_Gansu.transpose())[0])))
Z08_Qinghai = pd.DataFrame(np.matmul(np.matrix(wntio08_Qinghai),np.diag(np.array(RX08_Qinghai.transpose())[0])))

# regionalisation of Z09
Z09_Guangdong = pd.DataFrame(np.matmul(np.matrix(wntio09_Guangdong),np.diag(np.array(RX09_Guangdong.transpose())[0])))
Z09_Jiangsu = pd.DataFrame(np.matmul(np.matrix(wntio09_Jiangsu),np.diag(np.array(RX09_Jiangsu.transpose())[0])))
Z09_Gansu = pd.DataFrame(np.matmul(np.matrix(wntio09_Gansu),np.diag(np.array(RX09_Gansu.transpose())[0])))
Z09_Qinghai = pd.DataFrame(np.matmul(np.matrix(wntio09_Qinghai),np.diag(np.array(RX09_Qinghai.transpose())[0])))

# regionalisation of Z10
Z10_Guangdong = pd.DataFrame(np.matmul(np.matrix(wntio10_Guangdong),np.diag(np.array(RX10_Guangdong.transpose())[0])))
Z10_Jiangsu = pd.DataFrame(np.matmul(np.matrix(wntio10_Jiangsu),np.diag(np.array(RX10_Jiangsu.transpose())[0])))
Z10_Gansu = pd.DataFrame(np.matmul(np.matrix(wntio10_Gansu),np.diag(np.array(RX10_Gansu.transpose())[0])))
Z10_Qinghai = pd.DataFrame(np.matmul(np.matrix(wntio10_Qinghai),np.diag(np.array(RX10_Qinghai.transpose())[0])))

# regionalisation of Z11
Z11_Guangdong = pd.DataFrame(np.matmul(np.matrix(wntio11_Guangdong),np.diag(np.array(RX11_Guangdong.transpose())[0])))
Z11_Jiangsu = pd.DataFrame(np.matmul(np.matrix(wntio11_Jiangsu),np.diag(np.array(RX11_Jiangsu.transpose())[0])))
Z11_Gansu = pd.DataFrame(np.matmul(np.matrix(wntio11_Gansu),np.diag(np.array(RX11_Gansu.transpose())[0])))
Z11_Qinghai = pd.DataFrame(np.matmul(np.matrix(wntio11_Qinghai),np.diag(np.array(RX11_Qinghai.transpose())[0])))


# export to excel
#Guangdong
writer = pd.ExcelWriter('Regionalized table Guangdong.xlsx') 
Z98_Guangdong.to_excel(writer, sheet_name='reg98', index=False) 
Z99_Guangdong.to_excel(writer, sheet_name='reg99', index=False) 
Z00_Guangdong.to_excel(writer, sheet_name='reg00', index=False) 
Z01_Guangdong.to_excel(writer, sheet_name='reg01', index=False) 
Z02_Guangdong.to_excel(writer, sheet_name='reg02', index=False) 
Z03_Guangdong.to_excel(writer, sheet_name='reg03', index=False) 
Z04_Guangdong.to_excel(writer, sheet_name='reg04', index=False) 
Z05_Guangdong.to_excel(writer, sheet_name='reg05', index=False) 
Z06_Guangdong.to_excel(writer, sheet_name='reg06', index=False) 
Z07_Guangdong.to_excel(writer, sheet_name='reg07', index=False) 
Z08_Guangdong.to_excel(writer, sheet_name='reg08', index=False) 
Z09_Guangdong.to_excel(writer, sheet_name='reg09', index=False) 
Z10_Guangdong.to_excel(writer, sheet_name='reg10', index=False) 
Z11_Guangdong.to_excel(writer, sheet_name='reg11', index=False) 
writer.save()
writer.close()

#Gansu
writer = pd.ExcelWriter('Regionalized table Gansu.xlsx') 
Z98_Gansu.to_excel(writer, sheet_name='reg98', index=False) 
Z99_Gansu.to_excel(writer, sheet_name='reg99', index=False) 
Z00_Gansu.to_excel(writer, sheet_name='reg00', index=False) 
Z01_Gansu.to_excel(writer, sheet_name='reg01', index=False) 
Z02_Gansu.to_excel(writer, sheet_name='reg02', index=False) 
Z03_Gansu.to_excel(writer, sheet_name='reg03', index=False) 
Z04_Gansu.to_excel(writer, sheet_name='reg04', index=False) 
Z05_Gansu.to_excel(writer, sheet_name='reg05', index=False) 
Z06_Gansu.to_excel(writer, sheet_name='reg06', index=False) 
Z07_Gansu.to_excel(writer, sheet_name='reg07', index=False) 
Z08_Gansu.to_excel(writer, sheet_name='reg08', index=False) 
Z09_Gansu.to_excel(writer, sheet_name='reg09', index=False) 
Z10_Gansu.to_excel(writer, sheet_name='reg10', index=False) 
Z11_Gansu.to_excel(writer, sheet_name='reg11', index=False) 
writer.save()
writer.close()

#Jiangsu
writer = pd.ExcelWriter('Regionalized table Jiangsu.xlsx') 
Z98_Jiangsu.to_excel(writer, sheet_name='reg98', index=False) 
Z99_Jiangsu.to_excel(writer, sheet_name='reg99', index=False) 
Z00_Jiangsu.to_excel(writer, sheet_name='reg00', index=False) 
Z01_Jiangsu.to_excel(writer, sheet_name='reg01', index=False) 
Z02_Jiangsu.to_excel(writer, sheet_name='reg02', index=False) 
Z03_Jiangsu.to_excel(writer, sheet_name='reg03', index=False) 
Z04_Jiangsu.to_excel(writer, sheet_name='reg04', index=False) 
Z05_Jiangsu.to_excel(writer, sheet_name='reg05', index=False) 
Z06_Jiangsu.to_excel(writer, sheet_name='reg06', index=False) 
Z07_Jiangsu.to_excel(writer, sheet_name='reg07', index=False) 
Z08_Jiangsu.to_excel(writer, sheet_name='reg08', index=False) 
Z09_Jiangsu.to_excel(writer, sheet_name='reg09', index=False) 
Z10_Jiangsu.to_excel(writer, sheet_name='reg10', index=False) 
Z11_Jiangsu.to_excel(writer, sheet_name='reg11', index=False) 
writer.save()
writer.close()

#Qinghai
writer = pd.ExcelWriter('Regionalized table Qinghai.xlsx') 
Z98_Qinghai.to_excel(writer, sheet_name='reg98', index=False) 
Z99_Qinghai.to_excel(writer, sheet_name='reg99', index=False) 
Z00_Qinghai.to_excel(writer, sheet_name='reg00', index=False) 
Z01_Qinghai.to_excel(writer, sheet_name='reg01', index=False) 
Z02_Qinghai.to_excel(writer, sheet_name='reg02', index=False) 
Z03_Qinghai.to_excel(writer, sheet_name='reg03', index=False) 
Z04_Qinghai.to_excel(writer, sheet_name='reg04', index=False) 
Z05_Qinghai.to_excel(writer, sheet_name='reg05', index=False) 
Z06_Qinghai.to_excel(writer, sheet_name='reg06', index=False) 
Z07_Qinghai.to_excel(writer, sheet_name='reg07', index=False) 
Z08_Qinghai.to_excel(writer, sheet_name='reg08', index=False) 
Z09_Qinghai.to_excel(writer, sheet_name='reg09', index=False) 
Z10_Qinghai.to_excel(writer, sheet_name='reg10', index=False) 
Z11_Qinghai.to_excel(writer, sheet_name='reg11', index=False) 
writer.save()
writer.close()


#Rasmussen and Hirschman backward linkages Output multipliers
I = pd.DataFrame(np.eye(n))
#year 1998
B98 = pd.DataFrame((np.matrix(I,dtype=float)-np.matrix(A98, dtype=float)).I)
Multipliers98 = pd.DataFrame(np.sum(np.matrix(B98),axis=1))

#year 2004
B04 = pd.DataFrame((np.matrix(I,dtype=float)-np.matrix(A04, dtype=float)).I)
Multipliers04 = pd.DataFrame(np.sum(np.matrix(B04),axis=1))

#year 2011
B11 = pd.DataFrame((np.matrix(I,dtype=float)-np.matrix(A11, dtype=float)).I)
Multipliers11 = pd.DataFrame(np.sum(np.matrix(B11),axis=1))

# Importing all Population tables from year 1998 until 2011
pop98 = pd.read_csv('C:/Users/Pavel/Desktop/IES/Bachelor Thesis/data/Population/Population 98-2011 - 98.csv', sep=',', header = 5, skiprows = np.array([7,13,17,25,32,38,44,45,46,47,48,49]))
#pop99 = pd.read_csv('C:/Users/Pavel/Desktop/IES/Bachelor Thesis/data/Population/Population 98-2011 - 99.csv', sep=',', header = 4, skiprows = np.append(np.array([35,41,45,53,60,66],dtype=float),np.arange(5,34,dtype=float)))
#pop00 = pd.read_csv('C:/Users/Pavel/Desktop/IES/Bachelor Thesis/data/Population/Population 98-2011 - 00.csv', sep=',', header = 4, skiprows = np.append(np.array([39,45,49,57,64,70],dtype=float),np.arange(5,38,dtype=float)))
#pop01 = pd.read_csv('C:/Users/Pavel/Desktop/IES/Bachelor Thesis/data/Population/Population 98-2011 - 01.csv', sep=',', header = 4, skiprows = np.append(np.array([30,36,40,48,55,61],dtype=float),np.arange(5,29,dtype=float)))
#pop02 = pd.read_csv('C:/Users/Pavel/Desktop/IES/Bachelor Thesis/data/Population/Population 98-2011 - 02.csv', sep=',', header = 4, skiprows = np.append(np.arange(5,26,dtype=float),np.array([27],dtype=float)))
#pop03 = pd.read_csv('C:/Users/Pavel/Desktop/IES/Bachelor Thesis/data/Population/Population 98-2011 - 03.csv', sep=',', header = 4, skiprows = np.append(np.array([13,19,23,31,38,44],dtype=float),np.arange(5,12,dtype=float)))
pop04 = pd.read_csv('C:/Users/Pavel/Desktop/IES/Bachelor Thesis/data/Population/Population 98-2011 - 04.csv', sep=',', header = 7, skiprows = np.array([9,15,19,27,34,40,46,47,48,49,50,51]))
#pop05 = pd.read_csv('C:/Users/Pavel/Desktop/IES/Bachelor Thesis/data/Population/Population 98-2011 - 05.csv', sep=',', header = 4, skiprows = np.append(np.array([16,22,26,34,41,47],dtype=float),np.arange(5,15,dtype=float)))
#pop06 = pd.read_csv('C:/Users/Pavel/Desktop/IES/Bachelor Thesis/data/Population/Population 98-2011 - 06.csv', sep=',', header = 4, skiprows = np.append(np.array([15,21,25,33,40,46],dtype=float),np.arange(5,14,dtype=float)))
#pop07 = pd.read_csv('C:/Users/Pavel/Desktop/IES/Bachelor Thesis/data/Population/Population 98-2011 - 07.csv', sep=',', header = 4, skiprows = np.append(np.array([17,23,27,35,42,48],dtype=float),np.arange(5,16,dtype=float)))
#pop08 = pd.read_csv('C:/Users/Pavel/Desktop/IES/Bachelor Thesis/data/Population/Population 98-2011 - 08.csv', sep=',', header = 4, skiprows = np.append(np.array([18,24,28,36,43,49],dtype=float),np.arange(5,17,dtype=float)))
#pop09 = pd.read_csv('C:/Users/Pavel/Desktop/IES/Bachelor Thesis/data/Population/Population 98-2011 - 09.csv', sep=',', header = 4, skiprows = np.append(np.array([19,25,29,37,44,50],dtype=float),np.arange(5,18,dtype=float)))
#pop10 = pd.read_csv('C:/Users/Pavel/Desktop/IES/Bachelor Thesis/data/Population/Population 98-2011 - 10.csv', sep=',', header = 4, skiprows = np.append(np.array([20,26,30,38,45,51],dtype=float),np.arange(5,19,dtype=float)))
pop11 = pd.read_csv('C:/Users/Pavel/Desktop/IES/Bachelor Thesis/data/Population/Population 98-2011 - 11.csv', sep=',', header = 10, skiprows = np.array([12,18,22,30,37,43,49,50,51,52,53,54]))
#indexing
pop98 = pd.read_csv('C:/Users/Pavel/Desktop/IES/Bachelor Thesis/data/Population/Population 98-2011 - 98.csv', sep=',', header = 5, skiprows = np.array([7,13,17,25,32,38,44,45,46,47,48,49])).set_index(pop98.columns[0])
#pop99 = pd.read_csv('C:/Users/Pavel/Desktop/IES/Bachelor Thesis/data/Employment/csv/Employment by sectors and regions - 1999.csv', sep=',', header = 4, skiprows = np.append(np.array([35,41,45,53,60,66],dtype=float),np.arange(5,34,dtype=float))).set_index(emp99.columns[0])
#pop00 = pd.read_csv('C:/Users/Pavel/Desktop/IES/Bachelor Thesis/data/Employment/csv/Employment by sectors and regions - 2000.csv', sep=',', header = 4, skiprows = np.append(np.array([39,45,49,57,64,70],dtype=float),np.arange(5,38,dtype=float))).set_index(emp00.columns[0])
#pop01 = pd.read_csv('C:/Users/Pavel/Desktop/IES/Bachelor Thesis/data/Employment/csv/Employment by sectors and regions - 2001.csv', sep=',', header = 4, skiprows = np.append(np.array([30,36,40,48,55,61],dtype=float),np.arange(5,29,dtype=float))).set_index(emp01.columns[0])
#pop02 = pd.read_csv('C:/Users/Pavel/Desktop/IES/Bachelor Thesis/data/Employment/csv/Employment by sectors and regions - 2002.csv', sep=',', header = 4, skiprows = np.append(np.arange(5,26,dtype=float),np.array([27],dtype=float))).set_index(emp02.columns[0])
#pop03 = pd.read_csv('C:/Users/Pavel/Desktop/IES/Bachelor Thesis/data/Employment/csv/Employment by sectors and regions - 2003.csv', sep=',', header = 4, skiprows = np.append(np.array([13,19,23,31,38,44],dtype=float),np.arange(5,12,dtype=float))).set_index(emp03.columns[0])
pop04 = pd.read_csv('C:/Users/Pavel/Desktop/IES/Bachelor Thesis/data/Population/Population 98-2011 - 04.csv', sep=',', header = 7, skiprows = np.array([9,15,19,27,34,40,46,47,48,49,50,51])).set_index(pop04.columns[0])
#pop05 = pd.read_csv('C:/Users/Pavel/Desktop/IES/Bachelor Thesis/data/Employment/csv/Employment by sectors and regions - 2005.csv', sep=',', header = 4, skiprows = np.append(np.array([16,22,26,34,41,47],dtype=float),np.arange(5,15,dtype=float))).set_index(emp05.columns[0])
#pop06 = pd.read_csv('C:/Users/Pavel/Desktop/IES/Bachelor Thesis/data/Employment/csv/Employment by sectors and regions - 2006.csv', sep=',', header = 4, skiprows = np.append(np.array([15,21,25,33,40,46],dtype=float),np.arange(5,14,dtype=float))).set_index(emp06.columns[0])
#pop07 = pd.read_csv('C:/Users/Pavel/Desktop/IES/Bachelor Thesis/data/Employment/csv/Employment by sectors and regions - 2007.csv', sep=',', header = 4, skiprows = np.append(np.array([17,23,27,35,42,48],dtype=float),np.arange(5,16,dtype=float))).set_index(emp07.columns[0])
#pop08 = pd.read_csv('C:/Users/Pavel/Desktop/IES/Bachelor Thesis/data/Employment/csv/Employment by sectors and regions - 2008.csv', sep=',', header = 4, skiprows = np.append(np.array([18,24,28,36,43,49],dtype=float),np.arange(5,17,dtype=float))).set_index(emp08.columns[0])
#pop09 = pd.read_csv('C:/Users/Pavel/Desktop/IES/Bachelor Thesis/data/Employment/csv/Employment by sectors and regions - 2009.csv', sep=',', header = 4, skiprows = np.append(np.array([19,25,29,37,44,50],dtype=float),np.arange(5,18,dtype=float))).set_index(emp09.columns[0])
#pop10 = pd.read_csv('C:/Users/Pavel/Desktop/IES/Bachelor Thesis/data/Employment/csv/Employment by sectors and regions - 2010.csv', sep=',', header = 4, skiprows = np.append(np.array([20,26,30,38,45,51],dtype=float),np.arange(5,19,dtype=float))).set_index(emp10.columns[0])
pop11 = pd.read_csv('C:/Users/Pavel/Desktop/IES/Bachelor Thesis/data/Population/Population 98-2011 - 11.csv', sep=',', header = 10, skiprows = np.array([12,18,22,30,37,43,49,50,51,52,53,54])).set_index(pop11.columns[0])

#multiplication of 10th sector in fcio by (population in the province/total population)
#year 1998
div_pop98_Guangdong = pop98.loc["Guangdong"].div(pop98.loc["National Total"], axis = 0).iloc[0]
cons_possible_Guangdong98 = newfci98.loc[:,'10']*(div_pop98_Guangdong)
div_pop98_Jiangsu = pop98.loc["Jiangsu"].div(pop98.loc["National Total"], axis = 0).iloc[0]
cons_possible_Jiangsu98 = newfci98.loc[:,'10']*(div_pop98_Jiangsu)
div_pop98_Gansu = pop98.loc["Gansu"].div(pop98.loc["National Total"], axis = 0).iloc[0]
cons_possible_Gansu98 = newfci98.loc[:,'10']*(div_pop98_Gansu)
div_pop98_Qinghai = pop98.loc["Qinghai"].div(pop98.loc["National Total"], axis = 0).iloc[0]
cons_possible_Qinghai98 = newfci98.loc[:,'10']*(div_pop98_Qinghai)

#year 2004
div_pop04_Guangdong = pop04.loc["Guangdong"].div(pop04.loc["National Total"], axis = 0).iloc[0]
cons_possible_Guangdong04 = newfci04.loc[:,'10']*(div_pop04_Guangdong)
div_pop04_Jiangsu = pop04.loc["Jiangsu"].div(pop04.loc["National Total"], axis = 0).iloc[0]
cons_possible_Jiangsu04 = newfci04.loc[:,'10']*(div_pop04_Jiangsu)
div_pop04_Gansu = pop04.loc["Gansu"].div(pop04.loc["National Total"], axis = 0).iloc[0]
cons_possible_Gansu04 = newfci04.loc[:,'10']*(div_pop04_Gansu)
div_pop04_Qinghai = pop04.loc["Qinghai"].div(pop04.loc["National Total"], axis = 0).iloc[0]
cons_possible_Qinghai04 = newfci04.loc[:,'10']*(div_pop04_Qinghai)

#year 20011
div_pop11_Guangdong = pop11.loc["Guangdong"].div(pop11.loc["National Total"], axis = 0).iloc[0]
cons_possible_Guangdong11 = newfci11.loc[:,'10']*(div_pop11_Guangdong)
div_pop11_Jiangsu = pop11.loc["Jiangsu"].div(pop11.loc["National Total"], axis = 0).iloc[0]
cons_possible_Jiangsu11 = newfci11.loc[:,'10']*(div_pop11_Jiangsu)
div_pop11_Gansu = pop11.loc["Gansu"].div(pop11.loc["National Total"], axis = 0).iloc[0]
cons_possible_Gansu11 = newfci11.loc[:,'10']*(div_pop11_Gansu)
div_pop11_Qinghai = pop11.loc["Qinghai"].div(pop11.loc["National Total"], axis = 0).iloc[0]
cons_possible_Qinghai11 = newfci11.loc[:,'10']*(div_pop11_Qinghai)

#multiplication of 10,12,13th sectors in fcio by (employment in the sector in province/total employment in the sector)
#year 1998
div_emp98_Guangdong = newemp98.loc["Guangdong","1":"9"].div(newemp98.loc["1998","1":"9"], axis = 0)
possible_Guangdong98 = newfci98.mul(div_emp98_Guangdong, axis = 0)
div_emp98_Jiangsu = newemp98.loc["Jiangsu","1":"9"].div(newemp98.loc["1998","1":"9"], axis = 0)
possible_Jiangsu98 = newfci98.mul(div_emp98_Jiangsu, axis = 0)
div_emp98_Gansu = newemp98.loc["Gansu","1":"9"].div(newemp98.loc["1998","1":"9"], axis = 0)
possible_Gansu98 = newfci98.mul(div_emp98_Gansu, axis = 0)
div_emp98_Qinghai = newemp98.loc["Qinghai","1":"9"].div(newemp98.loc["1998","1":"9"], axis = 0)
possible_Qinghai98 = newfci98.mul(div_emp98_Qinghai, axis = 0)

#year 1999
div_emp99_Guangdong = newemp99.loc["Guangdong","1":"9"].div(newemp99.loc["1999","1":"9"], axis = 0)
possible_Guangdong99 = newfci99.mul(div_emp99_Guangdong, axis = 0)
div_emp99_Jiangsu = newemp99.loc["Jiangsu","1":"9"].div(newemp99.loc["1999","1":"9"], axis = 0)
possible_Jiangsu99 = newfci99.mul(div_emp99_Jiangsu, axis = 0)
div_emp99_Gansu = newemp99.loc["Gansu","1":"9"].div(newemp99.loc["1999","1":"9"], axis = 0)
possible_Gansu99 = newfci99.mul(div_emp99_Gansu, axis = 0)
div_emp99_Qinghai = newemp99.loc["Qinghai","1":"9"].div(newemp99.loc["1999","1":"9"], axis = 0)
possible_Qinghai99 = newfci99.mul(div_emp99_Qinghai, axis = 0)

#year 2000
div_emp00_Guangdong = newemp00.loc["Guangdong","1":"9"].div(newemp00.loc["2000","1":"9"], axis = 0)
possible_Guangdong00 = newfci00.mul(div_emp00_Guangdong, axis = 0)
div_emp00_Jiangsu = newemp00.loc["Jiangsu","1":"9"].div(newemp00.loc["2000","1":"9"], axis = 0)
possible_Jiangsu00 = newfci00.mul(div_emp00_Jiangsu, axis = 0)
div_emp00_Gansu = newemp00.loc["Gansu","1":"9"].div(newemp00.loc["2000","1":"9"], axis = 0)
possible_Gansu00 = newfci00.mul(div_emp00_Gansu, axis = 0)
div_emp00_Qinghai = newemp00.loc["Qinghai","1":"9"].div(newemp00.loc["2000","1":"9"], axis = 0)
possible_Qinghai00 = newfci00.mul(div_emp00_Qinghai, axis = 0)

#year 2001
div_emp01_Guangdong = newemp01.loc["Guangdong","1":"9"].div(newemp01.loc["2001","1":"9"], axis = 0)
possible_Guangdong01 = newfci01.mul(div_emp01_Guangdong, axis = 0)
div_emp01_Jiangsu = newemp01.loc["Jiangsu","1":"9"].div(newemp01.loc["2001","1":"9"], axis = 0)
possible_Jiangsu01 = newfci01.mul(div_emp01_Jiangsu, axis = 0)
div_emp01_Gansu = newemp01.loc["Gansu","1":"9"].div(newemp01.loc["2001","1":"9"], axis = 0)
possible_Gansu01 = newfci01.mul(div_emp01_Gansu, axis = 0)
div_emp01_Qinghai = newemp01.loc["Qinghai","1":"9"].div(newemp01.loc["2001","1":"9"], axis = 0)
possible_Qinghai01 = newfci01.mul(div_emp01_Qinghai, axis = 0)

#year 2002
div_emp02_Guangdong = newemp02.loc["Guangdong","1":"9"].div(newemp02.loc["2002","1":"9"], axis = 0)
possible_Guangdong02 = newfci02.mul(div_emp02_Guangdong, axis = 0)
div_emp02_Jiangsu = newemp02.loc["Jiangsu","1":"9"].div(newemp02.loc["2002","1":"9"], axis = 0)
possible_Jiangsu02 = newfci02.mul(div_emp02_Jiangsu, axis = 0)
div_emp02_Gansu = newemp02.loc["Gansu","1":"9"].div(newemp02.loc["2002","1":"9"], axis = 0)
possible_Gansu02 = newfci02.mul(div_emp02_Gansu, axis = 0)
div_emp02_Qinghai = newemp02.loc["Qinghai","1":"9"].div(newemp02.loc["2002","1":"9"], axis = 0)
possible_Qinghai02 = newfci02.mul(div_emp02_Qinghai, axis = 0)

#year 2003
div_emp03_Guangdong = newemp03.loc["Guangdong","1":"9"].div(newemp03.loc["2003","1":"9"], axis = 0)
possible_Guangdong03 = newfci03.mul(div_emp03_Guangdong, axis = 0)
div_emp03_Jiangsu = newemp03.loc["Jiangsu","1":"9"].div(newemp03.loc["2003","1":"9"], axis = 0)
possible_Jiangsu03 = newfci03.mul(div_emp03_Jiangsu, axis = 0)
div_emp03_Gansu = newemp03.loc["Gansu","1":"9"].div(newemp03.loc["2003","1":"9"], axis = 0)
possible_Gansu03 = newfci03.mul(div_emp03_Gansu, axis = 0)
div_emp03_Qinghai = newemp03.loc["Qinghai","1":"9"].div(newemp03.loc["2003","1":"9"], axis = 0)
possible_Qinghai03 = newfci03.mul(div_emp03_Qinghai, axis = 0)

#year 2004
div_emp04_Guangdong = newemp04.loc["Guangdong","1":"9"].div(newemp04.loc["2004","1":"9"], axis = 0)
possible_Guangdong04 = newfci04.mul(div_emp04_Guangdong, axis = 0)
div_emp04_Jiangsu = newemp04.loc["Jiangsu","1":"9"].div(newemp04.loc["2004","1":"9"], axis = 0)
possible_Jiangsu04 = newfci04.mul(div_emp04_Jiangsu, axis = 0)
div_emp04_Gansu = newemp04.loc["Gansu","1":"9"].div(newemp04.loc["2004","1":"9"], axis = 0)
possible_Gansu04 = newfci04.mul(div_emp04_Gansu, axis = 0)
div_emp04_Qinghai = newemp04.loc["Qinghai","1":"9"].div(newemp04.loc["2004","1":"9"], axis = 0)
possible_Qinghai04 = newfci04.mul(div_emp04_Qinghai, axis = 0)

#year 2005
div_emp05_Guangdong = newemp05.loc["Guangdong","1":"9"].div(newemp05.loc["2005","1":"9"], axis = 0)
possible_Guangdong05 = newfci05.mul(div_emp05_Guangdong, axis = 0)
div_emp05_Jiangsu = newemp05.loc["Jiangsu","1":"9"].div(newemp05.loc["2005","1":"9"], axis = 0)
possible_Jiangsu05 = newfci05.mul(div_emp05_Jiangsu, axis = 0)
div_emp05_Gansu = newemp05.loc["Gansu","1":"9"].div(newemp05.loc["2005","1":"9"], axis = 0)
possible_Gansu05 = newfci05.mul(div_emp05_Gansu, axis = 0)
div_emp05_Qinghai = newemp05.loc["Qinghai","1":"9"].div(newemp05.loc["2005","1":"9"], axis = 0)
possible_Qinghai05 = newfci05.mul(div_emp05_Qinghai, axis = 0)

#year 2006
div_emp06_Guangdong = newemp06.loc["Guangdong","1":"9"].div(newemp06.loc["2006","1":"9"], axis = 0)
possible_Guangdong06 = newfci06.mul(div_emp06_Guangdong, axis = 0)
div_emp06_Jiangsu = newemp06.loc["Jiangsu","1":"9"].div(newemp06.loc["2006","1":"9"], axis = 0)
possible_Jiangsu06 = newfci06.mul(div_emp06_Jiangsu, axis = 0)
div_emp06_Gansu = newemp06.loc["Gansu","1":"9"].div(newemp06.loc["2006","1":"9"], axis = 0)
possible_Gansu06 = newfci06.mul(div_emp06_Gansu, axis = 0)
div_emp06_Qinghai = newemp06.loc["Qinghai","1":"9"].div(newemp06.loc["2006","1":"9"], axis = 0)
possible_Qinghai06 = newfci06.mul(div_emp06_Qinghai, axis = 0)

#year 2007
div_emp07_Guangdong = newemp07.loc["Guangdong","1":"9"].div(newemp07.loc["2007","1":"9"], axis = 0)
possible_Guangdong07 = newfci07.mul(div_emp07_Guangdong, axis = 0)
div_emp07_Jiangsu = newemp07.loc["Jiangsu","1":"9"].div(newemp07.loc["2007","1":"9"], axis = 0)
possible_Jiangsu07 = newfci07.mul(div_emp07_Jiangsu, axis = 0)
div_emp07_Gansu = newemp07.loc["Gansu","1":"9"].div(newemp07.loc["2007","1":"9"], axis = 0)
possible_Gansu07 = newfci07.mul(div_emp07_Gansu, axis = 0)
div_emp07_Qinghai = newemp07.loc["Qinghai","1":"9"].div(newemp07.loc["2007","1":"9"], axis = 0)
possible_Qinghai07 = newfci07.mul(div_emp07_Qinghai, axis = 0)

#year 2008
div_emp08_Guangdong = newemp08.loc["Guangdong","1":"9"].div(newemp08.loc["2008","1":"9"], axis = 0)
possible_Guangdong08 = newfci08.mul(div_emp08_Guangdong, axis = 0)
div_emp08_Jiangsu = newemp08.loc["Jiangsu","1":"9"].div(newemp08.loc["2008","1":"9"], axis = 0)
possible_Jiangsu08 = newfci08.mul(div_emp08_Jiangsu, axis = 0)
div_emp08_Gansu = newemp08.loc["Gansu","1":"9"].div(newemp08.loc["2008","1":"9"], axis = 0)
possible_Gansu08 = newfci08.mul(div_emp08_Gansu, axis = 0)
div_emp08_Qinghai = newemp08.loc["Qinghai","1":"9"].div(newemp08.loc["2008","1":"9"], axis = 0)
possible_Qinghai08 = newfci08.mul(div_emp08_Qinghai, axis = 0)

#year 2009
div_emp09_Guangdong = newemp09.loc["Guangdong","1":"9"].div(newemp09.loc["2009","1":"9"], axis = 0)
possible_Guangdong09 = newfci09.mul(div_emp09_Guangdong, axis = 0)
div_emp09_Jiangsu = newemp09.loc["Jiangsu","1":"9"].div(newemp09.loc["2009","1":"9"], axis = 0)
possible_Jiangsu09 = newfci09.mul(div_emp09_Jiangsu, axis = 0)
div_emp09_Gansu = newemp09.loc["Gansu","1":"9"].div(newemp09.loc["2009","1":"9"], axis = 0)
possible_Gansu09 = newfci09.mul(div_emp09_Gansu, axis = 0)
div_emp09_Qinghai = newemp09.loc["Qinghai","1":"9"].div(newemp09.loc["2009","1":"9"], axis = 0)
possible_Qinghai09 = newfci09.mul(div_emp09_Qinghai, axis = 0)

#year 2010
div_emp10_Guangdong = newemp10.loc["Guangdong","1":"9"].div(newemp10.loc["2010","1":"9"], axis = 0)
possible_Guangdong10 = newfci10.mul(div_emp10_Guangdong, axis = 0)
div_emp10_Jiangsu = newemp10.loc["Jiangsu","1":"9"].div(newemp10.loc["2010","1":"9"], axis = 0)
possible_Jiangsu10 = newfci10.mul(div_emp10_Jiangsu, axis = 0)
div_emp10_Gansu = newemp10.loc["Gansu","1":"9"].div(newemp10.loc["2010","1":"9"], axis = 0)
possible_Gansu10 = newfci10.mul(div_emp10_Gansu, axis = 0)
div_emp10_Qinghai = newemp10.loc["Qinghai","1":"9"].div(newemp10.loc["2010","1":"9"], axis = 0)
possible_Qinghai10 = newfci10.mul(div_emp10_Qinghai, axis = 0)

#year 2011
div_emp11_Guangdong = newemp11.loc["Guangdong","1":"9"].div(newemp11.loc["2011","1":"9"], axis = 0)
possible_Guangdong11 = newfci11.mul(div_emp11_Guangdong, axis = 0)
div_emp11_Jiangsu = newemp11.loc["Jiangsu","1":"9"].div(newemp11.loc["2011","1":"9"], axis = 0)
possible_Jiangsu11 = newfci11.mul(div_emp11_Jiangsu, axis = 0)
div_emp11_Gansu = newemp11.loc["Gansu","1":"9"].div(newemp11.loc["2011","1":"9"], axis = 0)
possible_Gansu11 = newfci11.mul(div_emp11_Gansu, axis = 0)
div_emp11_Qinghai = newemp11.loc["Qinghai","1":"9"].div(newemp11.loc["2011","1":"9"], axis = 0)
possible_Qinghai11 = newfci11.mul(div_emp11_Qinghai, axis = 0)

# export to excel
writer = pd.ExcelWriter('Possible_regional_trade98_dom.xlsx') 
cons_possible_Guangdong98.to_excel(writer, sheet_name='cons_possible_Guangdong98_dom', index=False) 
cons_possible_Jiangsu98.to_excel(writer, sheet_name='cons_possible_Jiangsu98_dom', index=False) 
cons_possible_Gansu98.to_excel(writer, sheet_name='cons_possible_Gansu98_dom', index=False) 
cons_possible_Qinghai98.to_excel(writer, sheet_name='cons_possible_Qinghai98_dom', index=False) 
possible_Guangdong98.to_excel(writer, sheet_name='possible_Guangdong98_dom',index=False)
possible_Jiangsu98.to_excel(writer, sheet_name='possible_Jiangsu98_dom', index=False) 
possible_Gansu98.to_excel(writer, sheet_name='possible_Gansu98_dom', index=False) 
possible_Qinghai98.to_excel(writer, sheet_name='possible_Qinghai98_dom', index=False) 
writer.save()
writer.close()

writer = pd.ExcelWriter('Regionalisation04_dom.xlsx') 
CILQ_emp04_Guangdong.to_excel(writer, sheet_name='CILQ_04_Guangdong', index=False) 
CILQ_emp04_Jiangsu.to_excel(writer, sheet_name='CILQ_04_Jiangsu', index=False) 
CILQ_emp04_Gansu.to_excel(writer, sheet_name='CILQ_04_Gansu', index=False) 
CILQ_emp04_Qinghai.to_excel(writer, sheet_name='CILQ_04_Qinghai', index=False) 
A04.to_excel(writer, sheet_name='A_matrix_04',index=False)
ntio04_Guangdong.to_excel(writer, sheet_name='IO_04_Guangdong', index=False) 
ntio04_Jiangsu.to_excel(writer, sheet_name='IO_04_Jiangsu', index=False) 
ntio04_Gansu.to_excel(writer, sheet_name='IO_04_Gansu', index=False) 
ntio04_Qinghai.to_excel(writer, sheet_name='IO_04_Qinghai', index=False) 
writer.save()
writer.close()

writer = pd.ExcelWriter('Regionalisation11_dom.xlsx') 
CILQ_emp11_Guangdong.to_excel(writer, sheet_name='CILQ_11_Guangdong', index=False) 
CILQ_emp11_Jiangsu.to_excel(writer, sheet_name='CILQ_11_Jiangsu', index=False) 
CILQ_emp11_Gansu.to_excel(writer, sheet_name='CILQ_11_Gansu', index=False) 
CILQ_emp11_Qinghai.to_excel(writer, sheet_name='CILQ_11_Qinghai', index=False) 
A11.to_excel(writer, sheet_name='A_matrix_11',index=False)
ntio11_Guangdong.to_excel(writer, sheet_name='IO_11_Guangdong', index=False) 
ntio11_Jiangsu.to_excel(writer, sheet_name='IO_11_Jiangsu', index=False) 
ntio11_Gansu.to_excel(writer, sheet_name='IO_11_Gansu', index=False) 
ntio11_Qinghai.to_excel(writer, sheet_name='IO_11_Qinghai', index=False) 
writer.save()
writer.close()

# impoting all regional GDP tables
gdp98 = pd.read_csv('C:/Users/Pavel/Desktop/IES/Bachelor Thesis/data/GDP/Regional GDP - 98.csv', sep=',', header = 8, skiprows = np.array([9,15,19,27,34,40,46]))
gdp99 = pd.read_csv('C:/Users/Pavel/Desktop/IES/Bachelor Thesis/data/GDP/Regional GDP - 99.csv', sep=',', header = 14, skiprows = np.array([20,24,32,39,45],dtype=float))
gdp00 = pd.read_csv('C:/Users/Pavel/Desktop/IES/Bachelor Thesis/data/GDP/Regional GDP - 00.csv', sep=',', header = 12, skiprows = np.array([18,22,30,37,43],dtype=float))
gdp01 = pd.read_csv('C:/Users/Pavel/Desktop/IES/Bachelor Thesis/data/GDP/Regional GDP - 01.csv', sep=',', header = 9, skiprows = np.array([15,19,27,34,40],dtype=float))
gdp02 = pd.read_csv('C:/Users/Pavel/Desktop/IES/Bachelor Thesis/data/GDP/Regional GDP - 02.csv', sep=',', header = 12, skiprows = np.array([18,22,30,37,43],dtype=float))
gdp03 = pd.read_csv('C:/Users/Pavel/Desktop/IES/Bachelor Thesis/data/GDP/Regional GDP - 03.csv', sep=',', header = 12, skiprows = np.array([18,22,30,37,43],dtype=float))
gdp04 = pd.read_csv('C:/Users/Pavel/Desktop/IES/Bachelor Thesis/data/GDP/Regional GDP - 04.csv', sep=',', header = 13, skiprows = np.array([19,23,31,38,44],dtype=float))
gdp05 = pd.read_csv('C:/Users/Pavel/Desktop/IES/Bachelor Thesis/data/GDP/Regional GDP - 05.csv', sep=',', header = 14, skiprows = np.array([20,24,32,39,45],dtype=float))
gdp06 = pd.read_csv('C:/Users/Pavel/Desktop/IES/Bachelor Thesis/data/GDP/Regional GDP - 06.csv', sep=',', header = 8, skiprows = np.array([14,18,26,33,39]))
gdp07 = pd.read_csv('C:/Users/Pavel/Desktop/IES/Bachelor Thesis/data/GDP/Regional GDP - 07.csv', sep=',', header = 8, skiprows = np.array([14,18,26,33,39]))
gdp08 = pd.read_csv('C:/Users/Pavel/Desktop/IES/Bachelor Thesis/data/GDP/Regional GDP - 08.csv', sep=',', header = 11, skiprows = np.array([17,21,29,36,42]))
gdp09 = pd.read_csv('C:/Users/Pavel/Desktop/IES/Bachelor Thesis/data/GDP/Regional GDP - 09.csv', sep=',', header = 9, skiprows = np.array([15,19,27,34,40],dtype=float))
gdp10 = pd.read_csv('C:/Users/Pavel/Desktop/IES/Bachelor Thesis/data/GDP/Regional GDP - 10.csv', sep=',', header = 11, skiprows = np.array([17,21,29,36,42]))
gdp11 = pd.read_csv('C:/Users/Pavel/Desktop/IES/Bachelor Thesis/data/GDP/Regional GDP - 11.csv', sep=',', header = 10, skiprows = np.array([16,20,28,35,41],dtype=float))
#indexing
gdp98 = pd.read_csv('C:/Users/Pavel/Desktop/IES/Bachelor Thesis/data/GDP/Regional GDP - 98.csv', sep=',', header = 8, skiprows = np.array([9,15,19,27,34,40,46])).set_index(gdp98.columns[0])
gdp99 = pd.read_csv('C:/Users/Pavel/Desktop/IES/Bachelor Thesis/data/GDP/Regional GDP - 99.csv', sep=',', header = 14, skiprows = np.array([20,24,32,39,45],dtype=float)).set_index(gdp99.columns[0])
gdp00 = pd.read_csv('C:/Users/Pavel/Desktop/IES/Bachelor Thesis/data/GDP/Regional GDP - 00.csv', sep=',', header = 12, skiprows = np.array([18,22,30,37,43],dtype=float)).set_index(gdp00.columns[0])
gdp01 = pd.read_csv('C:/Users/Pavel/Desktop/IES/Bachelor Thesis/data/GDP/Regional GDP - 01.csv', sep=',', header = 9, skiprows = np.array([15,19,27,34,40],dtype=float)).set_index(gdp01.columns[0])
gdp02 = pd.read_csv('C:/Users/Pavel/Desktop/IES/Bachelor Thesis/data/GDP/Regional GDP - 02.csv', sep=',', header = 12, skiprows = np.array([18,22,30,37,43],dtype=float)).set_index(gdp02.columns[0])
gdp03 = pd.read_csv('C:/Users/Pavel/Desktop/IES/Bachelor Thesis/data/GDP/Regional GDP - 03.csv', sep=',', header = 12, skiprows = np.array([18,22,30,37,43],dtype=float)).set_index(gdp03.columns[0])
gdp04 = pd.read_csv('C:/Users/Pavel/Desktop/IES/Bachelor Thesis/data/GDP/Regional GDP - 04.csv', sep=',', header = 13, skiprows = np.array([19,23,31,38,44],dtype=float)).set_index(gdp04.columns[0])
gdp05 = pd.read_csv('C:/Users/Pavel/Desktop/IES/Bachelor Thesis/data/GDP/Regional GDP - 05.csv', sep=',', header = 14, skiprows = np.array([20,24,32,39,45],dtype=float)).set_index(gdp05.columns[0])
gdp06 = pd.read_csv('C:/Users/Pavel/Desktop/IES/Bachelor Thesis/data/GDP/Regional GDP - 06.csv', sep=',', header = 8, skiprows = np.array([14,18,26,33,39])).set_index(gdp06.columns[0])
gdp07 = pd.read_csv('C:/Users/Pavel/Desktop/IES/Bachelor Thesis/data/GDP/Regional GDP - 07.csv', sep=',', header = 8, skiprows = np.array([14,18,26,33,39])).set_index(gdp07.columns[0])
gdp08 = pd.read_csv('C:/Users/Pavel/Desktop/IES/Bachelor Thesis/data/GDP/Regional GDP - 08.csv', sep=',', header = 11, skiprows = np.array([17,21,29,36,42])).set_index(gdp08.columns[0])
gdp09 = pd.read_csv('C:/Users/Pavel/Desktop/IES/Bachelor Thesis/data/GDP/Regional GDP - 09.csv', sep=',', header = 9, skiprows = np.array([15,19,27,34,40],dtype=float)).set_index(gdp09.columns[0])
gdp10 = pd.read_csv('C:/Users/Pavel/Desktop/IES/Bachelor Thesis/data/GDP/Regional GDP - 10.csv', sep=',', header = 11, skiprows = np.array([17,21,29,36,42])).set_index(gdp10.columns[0])
gdp11 = pd.read_csv('C:/Users/Pavel/Desktop/IES/Bachelor Thesis/data/GDP/Regional GDP - 11.csv', sep=',', header = 10, skiprows = np.array([16,20,28,35,41],dtype=float)).set_index(gdp11.columns[0])

#multiplication of 10th sector in fcio by (gdp in the province/total gdp)
#year 1998
div_gdp98_Guangdong = gdp98.loc["Guangdong"].div(gdp98.iloc[:,[0]].sum()).dropna().iloc[0]
gdp_possible_Guangdong98 = newfci98.loc[:,'10']*(div_gdp98_Guangdong)
div_gdp98_Jiangsu = gdp98.loc["Jiangsu"].div(gdp98.iloc[:,[0]].sum()).dropna().iloc[0]
gdp_possible_Jiangsu98 = newfci98.loc[:,'10']*(div_gdp98_Jiangsu)
div_gdp98_Gansu = gdp98.loc["Gansu"].div(gdp98.iloc[:,[0]].sum()).dropna().iloc[0]
gdp_possible_Gansu98 = newfci98.loc[:,'10']*(div_gdp98_Gansu)
div_gdp98_Qinghai = gdp98.loc["Qinghai"].div(gdp98.iloc[:,[0]].sum()).dropna().iloc[0]
gdp_possible_Qinghai98 = newfci98.loc[:,'10']*(div_gdp98_Qinghai)

#year 1999
div_gdp99_Guangdong = gdp99.loc["Guangdong"].div(gdp99.iloc[:,[0]].sum()).dropna().iloc[0]
gdp_possible_Guangdong99 = newfci99.loc[:,'10']*(div_gdp99_Guangdong)
div_gdp99_Jiangsu = gdp99.loc["Jiangsu"].div(gdp99.iloc[:,[0]].sum()).dropna().iloc[0]
gdp_possible_Jiangsu99 = newfci99.loc[:,'10']*(div_gdp99_Jiangsu)
div_gdp99_Gansu = gdp99.loc["Gansu"].div(gdp99.iloc[:,[0]].sum()).dropna().iloc[0]
gdp_possible_Gansu99 = newfci99.loc[:,'10']*(div_gdp99_Gansu)
div_gdp99_Qinghai = gdp99.loc["Qinghai"].div(gdp99.iloc[:,[0]].sum()).dropna().iloc[0]
gdp_possible_Qinghai99 = newfci99.loc[:,'10']*(div_gdp99_Qinghai)

#year 2000
div_gdp00_Guangdong = gdp00.loc["Guangdong"].div(gdp00.iloc[:,[0]].sum()).dropna().iloc[0]
gdp_possible_Guangdong00 = newfci00.loc[:,'10']*(div_gdp00_Guangdong)
div_gdp00_Jiangsu = gdp00.loc["Jiangsu"].div(gdp00.iloc[:,[0]].sum()).dropna().iloc[0]
gdp_possible_Jiangsu00 = newfci00.loc[:,'10']*(div_gdp00_Jiangsu)
div_gdp00_Gansu = gdp00.loc["Gansu"].div(gdp00.iloc[:,[0]].sum()).dropna().iloc[0]
gdp_possible_Gansu00 = newfci00.loc[:,'10']*(div_gdp00_Gansu)
div_gdp00_Qinghai = gdp00.loc["Qinghai"].div(gdp00.iloc[:,[0]].sum()).dropna().iloc[0]
gdp_possible_Qinghai00 = newfci00.loc[:,'10']*(div_gdp00_Qinghai)

#year 2001
div_gdp01_Guangdong = gdp01.loc["Guangdong"].div(gdp01.iloc[:,[0]].sum()).dropna().iloc[0]
gdp_possible_Guangdong01 = newfci01.loc[:,'10']*(div_gdp01_Guangdong)
div_gdp01_Jiangsu = gdp01.loc["Jiangsu"].div(gdp01.iloc[:,[0]].sum()).dropna().iloc[0]
gdp_possible_Jiangsu01 = newfci01.loc[:,'10']*(div_gdp01_Jiangsu)
div_gdp01_Gansu = gdp01.loc["Gansu"].div(gdp01.iloc[:,[0]].sum()).dropna().iloc[0]
gdp_possible_Gansu01 = newfci01.loc[:,'10']*(div_gdp01_Gansu)
div_gdp01_Qinghai = gdp01.loc["Qinghai"].div(gdp01.iloc[:,[0]].sum()).dropna().iloc[0]
gdp_possible_Qinghai01 = newfci01.loc[:,'10']*(div_gdp01_Qinghai)

#year 2002
div_gdp02_Guangdong = gdp02.loc["Guangdong"].div(gdp02.iloc[:,[0]].sum()).dropna().iloc[0]
gdp_possible_Guangdong02 = newfci02.loc[:,'10']*(div_gdp02_Guangdong)
div_gdp02_Jiangsu = gdp02.loc["Jiangsu"].div(gdp02.iloc[:,[0]].sum()).dropna().iloc[0]
gdp_possible_Jiangsu02 = newfci02.loc[:,'10']*(div_gdp02_Jiangsu)
div_gdp02_Gansu = gdp02.loc["Gansu"].div(gdp02.iloc[:,[0]].sum()).dropna().iloc[0]
gdp_possible_Gansu02 = newfci02.loc[:,'10']*(div_gdp02_Gansu)
div_gdp02_Qinghai = gdp02.loc["Qinghai"].div(gdp02.iloc[:,[0]].sum()).dropna().iloc[0]
gdp_possible_Qinghai02 = newfci02.loc[:,'10']*(div_gdp02_Qinghai)

#year 2003
div_gdp03_Guangdong = gdp03.loc["Guangdong"].div(gdp03.iloc[:,[0]].sum()).dropna().iloc[0]
gdp_possible_Guangdong03 = newfci03.loc[:,'10']*(div_gdp03_Guangdong)
div_gdp03_Jiangsu = gdp03.loc["Jiangsu"].div(gdp03.iloc[:,[0]].sum()).dropna().iloc[0]
gdp_possible_Jiangsu03 = newfci03.loc[:,'10']*(div_gdp03_Jiangsu)
div_gdp03_Gansu = gdp03.loc["Gansu"].div(gdp03.iloc[:,[0]].sum()).dropna().iloc[0]
gdp_possible_Gansu03 = newfci03.loc[:,'10']*(div_gdp03_Gansu)
div_gdp03_Qinghai = gdp03.loc["Qinghai"].div(gdp03.iloc[:,[0]].sum()).dropna().iloc[0]
gdp_possible_Qinghai03 = newfci03.loc[:,'10']*(div_gdp03_Qinghai)

#year 2004
div_gdp04_Guangdong = gdp04.loc["Guangdong"].div(gdp04.iloc[:,[0]].sum()).dropna().iloc[0]
gdp_possible_Guangdong04 = newfci04.loc[:,'10']*(div_gdp04_Guangdong)
div_gdp04_Jiangsu = gdp04.loc["Jiangsu"].div(gdp04.iloc[:,[0]].sum()).dropna().iloc[0]
gdp_possible_Jiangsu04 = newfci04.loc[:,'10']*(div_gdp04_Jiangsu)
div_gdp04_Gansu = gdp04.loc["Gansu"].div(gdp04.iloc[:,[0]].sum()).dropna().iloc[0]
gdp_possible_Gansu04 = newfci04.loc[:,'10']*(div_gdp04_Gansu)
div_gdp04_Qinghai = gdp04.loc["Qinghai"].div(gdp04.iloc[:,[0]].sum()).dropna().iloc[0]
gdp_possible_Qinghai04 = newfci04.loc[:,'10']*(div_gdp04_Qinghai)

#year 2005
div_gdp05_Guangdong = gdp05.loc["Guangdong"].div(gdp05.iloc[:,[0]].sum()).dropna().iloc[0]
gdp_possible_Guangdong05 = newfci05.loc[:,'10']*(div_gdp05_Guangdong)
div_gdp05_Jiangsu = gdp05.loc["Jiangsu"].div(gdp05.iloc[:,[0]].sum()).dropna().iloc[0]
gdp_possible_Jiangsu05 = newfci05.loc[:,'10']*(div_gdp05_Jiangsu)
div_gdp05_Gansu = gdp05.loc["Gansu"].div(gdp05.iloc[:,[0]].sum()).dropna().iloc[0]
gdp_possible_Gansu05 = newfci05.loc[:,'10']*(div_gdp05_Gansu)
div_gdp05_Qinghai = gdp05.loc["Qinghai"].div(gdp05.iloc[:,[0]].sum()).dropna().iloc[0]
gdp_possible_Qinghai05 = newfci05.loc[:,'10']*(div_gdp05_Qinghai)

#year 2006
div_gdp06_Guangdong = gdp06.loc["Guangdong"].div(gdp06.iloc[:,[0]].sum()).dropna().iloc[0]
gdp_possible_Guangdong06 = newfci06.loc[:,'10']*(div_gdp06_Guangdong)
div_gdp06_Jiangsu = gdp06.loc["Jiangsu"].div(gdp06.iloc[:,[0]].sum()).dropna().iloc[0]
gdp_possible_Jiangsu06 = newfci06.loc[:,'10']*(div_gdp06_Jiangsu)
div_gdp06_Gansu = gdp06.loc["Gansu"].div(gdp06.iloc[:,[0]].sum()).dropna().iloc[0]
gdp_possible_Gansu06 = newfci06.loc[:,'10']*(div_gdp06_Gansu)
div_gdp06_Qinghai = gdp06.loc["Qinghai"].div(gdp06.iloc[:,[0]].sum()).dropna().iloc[0]
gdp_possible_Qinghai06 = newfci06.loc[:,'10']*(div_gdp06_Qinghai)

#year 2007
div_gdp07_Guangdong = gdp07.loc["Guangdong"].div(gdp07.iloc[:,[0]].sum()).dropna().iloc[0]
gdp_possible_Guangdong07 = newfci07.loc[:,'10']*(div_gdp07_Guangdong)
div_gdp07_Jiangsu = gdp07.loc["Jiangsu"].div(gdp07.iloc[:,[0]].sum()).dropna().iloc[0]
gdp_possible_Jiangsu07 = newfci07.loc[:,'10']*(div_gdp07_Jiangsu)
div_gdp07_Gansu = gdp07.loc["Gansu"].div(gdp07.iloc[:,[0]].sum()).dropna().iloc[0]
gdp_possible_Gansu07 = newfci07.loc[:,'10']*(div_gdp07_Gansu)
div_gdp07_Qinghai = gdp07.loc["Qinghai"].div(gdp07.iloc[:,[0]].sum()).dropna().iloc[0]
gdp_possible_Qinghai07 = newfci07.loc[:,'10']*(div_gdp07_Qinghai)

#year 2008
div_gdp08_Guangdong = gdp08.loc["Guangdong"].div(gdp08.iloc[:,[0]].sum()).dropna().iloc[0]
gdp_possible_Guangdong08 = newfci08.loc[:,'10']*(div_gdp08_Guangdong)
div_gdp08_Jiangsu = gdp08.loc["Jiangsu"].div(gdp08.iloc[:,[0]].sum()).dropna().iloc[0]
gdp_possible_Jiangsu08 = newfci08.loc[:,'10']*(div_gdp08_Jiangsu)
div_gdp08_Gansu = gdp08.loc["Gansu"].div(gdp08.iloc[:,[0]].sum()).dropna().iloc[0]
gdp_possible_Gansu08 = newfci08.loc[:,'10']*(div_gdp08_Gansu)
div_gdp08_Qinghai = gdp08.loc["Qinghai"].div(gdp08.iloc[:,[0]].sum()).dropna().iloc[0]
gdp_possible_Qinghai08 = newfci08.loc[:,'10']*(div_gdp08_Qinghai)

#year 2009
div_gdp09_Guangdong = gdp09.loc["Guangdong"].div(gdp09.iloc[:,[0]].sum()).dropna().iloc[0]
gdp_possible_Guangdong09 = newfci09.loc[:,'10']*(div_gdp09_Guangdong)
div_gdp09_Jiangsu = gdp09.loc["Jiangsu"].div(gdp09.iloc[:,[0]].sum()).dropna().iloc[0]
gdp_possible_Jiangsu09 = newfci09.loc[:,'10']*(div_gdp09_Jiangsu)
div_gdp09_Gansu = gdp09.loc["Gansu"].div(gdp09.iloc[:,[0]].sum()).dropna().iloc[0]
gdp_possible_Gansu09 = newfci09.loc[:,'10']*(div_gdp09_Gansu)
div_gdp09_Qinghai = gdp09.loc["Qinghai"].div(gdp09.iloc[:,[0]].sum()).dropna().iloc[0]
gdp_possible_Qinghai09 = newfci09.loc[:,'10']*(div_gdp09_Qinghai)

#year 2010
div_gdp10_Guangdong = gdp10.loc["Guangdong"].div(gdp10.iloc[:,[0]].sum()).dropna().iloc[0]
gdp_possible_Guangdong10 = newfci10.loc[:,'10']*(div_gdp10_Guangdong)
div_gdp10_Jiangsu = gdp10.loc["Jiangsu"].div(gdp10.iloc[:,[0]].sum()).dropna().iloc[0]
gdp_possible_Jiangsu10 = newfci10.loc[:,'10']*(div_gdp10_Jiangsu)
div_gdp10_Gansu = gdp10.loc["Gansu"].div(gdp10.iloc[:,[0]].sum()).dropna().iloc[0]
gdp_possible_Gansu10 = newfci10.loc[:,'10']*(div_gdp10_Gansu)
div_gdp10_Qinghai = gdp10.loc["Qinghai"].div(gdp10.iloc[:,[0]].sum()).dropna().iloc[0]
gdp_possible_Qinghai10 = newfci10.loc[:,'10']*(div_gdp10_Qinghai)

#year 2011
div_gdp11_Guangdong = gdp11.loc["Guangdong"].div(gdp11.iloc[:,[0]].sum()).dropna().iloc[0]
gdp_possible_Guangdong11 = newfci11.loc[:,'10']*(div_gdp11_Guangdong)
div_gdp11_Jiangsu = gdp11.loc["Jiangsu"].div(gdp11.iloc[:,[0]].sum()).dropna().iloc[0]
gdp_possible_Jiangsu11 = newfci11.loc[:,'10']*(div_gdp11_Jiangsu)
div_gdp11_Gansu = gdp11.loc["Gansu"].div(gdp11.iloc[:,[0]].sum()).dropna().iloc[0]
gdp_possible_Gansu11 = newfci11.loc[:,'10']*(div_gdp11_Gansu)
div_gdp11_Qinghai = gdp11.loc["Qinghai"].div(gdp11.iloc[:,[0]].sum()).dropna().iloc[0]
gdp_possible_Qinghai11 = newfci11.loc[:,'10']*(div_gdp11_Qinghai)


# Building Identity Matrix
I = pd.DataFrame(np.eye(9),columns = dio.keys(), index = dio.keys())

# Regional Outputs
#Guangdong
x_Guangdong98 = pd.DataFrame(np.matmul(np.linalg.inv(np.matrix(I - ntio98_Guangdong)),np.matrix(possible_Guangdong98.loc[:,"10"]).transpose()))
x_Guangdong99 = pd.DataFrame(np.matmul(np.linalg.inv(np.matrix(I - ntio99_Guangdong)),np.matrix(possible_Guangdong99.loc[:,"10"]).transpose()))
x_Guangdong00 = pd.DataFrame(np.matmul(np.linalg.inv(np.matrix(I - ntio00_Guangdong)),np.matrix(possible_Guangdong00.loc[:,"10"]).transpose()))
x_Guangdong01 = pd.DataFrame(np.matmul(np.linalg.inv(np.matrix(I - ntio01_Guangdong)),np.matrix(possible_Guangdong01.loc[:,"10"]).transpose()))
x_Guangdong02 = pd.DataFrame(np.matmul(np.linalg.inv(np.matrix(I - ntio02_Guangdong)),np.matrix(possible_Guangdong02.loc[:,"10"]).transpose()))
x_Guangdong03 = pd.DataFrame(np.matmul(np.linalg.inv(np.matrix(I - ntio03_Guangdong)),np.matrix(possible_Guangdong03.loc[:,"10"]).transpose()))
x_Guangdong04 = pd.DataFrame(np.matmul(np.linalg.inv(np.matrix(I - ntio04_Guangdong)),np.matrix(possible_Guangdong04.loc[:,"10"]).transpose()))
x_Guangdong05 = pd.DataFrame(np.matmul(np.linalg.inv(np.matrix(I - ntio05_Guangdong)),np.matrix(possible_Guangdong05.loc[:,"10"]).transpose()))
x_Guangdong06 = pd.DataFrame(np.matmul(np.linalg.inv(np.matrix(I - ntio06_Guangdong)),np.matrix(possible_Guangdong06.loc[:,"10"]).transpose()))
x_Guangdong07 = pd.DataFrame(np.matmul(np.linalg.inv(np.matrix(I - ntio07_Guangdong)),np.matrix(possible_Guangdong07.loc[:,"10"]).transpose()))
x_Guangdong08 = pd.DataFrame(np.matmul(np.linalg.inv(np.matrix(I - ntio08_Guangdong)),np.matrix(possible_Guangdong08.loc[:,"10"]).transpose()))
x_Guangdong09 = pd.DataFrame(np.matmul(np.linalg.inv(np.matrix(I - ntio09_Guangdong)),np.matrix(possible_Guangdong09.loc[:,"10"]).transpose()))
x_Guangdong10 = pd.DataFrame(np.matmul(np.linalg.inv(np.matrix(I - ntio10_Guangdong)),np.matrix(possible_Guangdong10.loc[:,"10"]).transpose()))
x_Guangdong11 = pd.DataFrame(np.matmul(np.linalg.inv(np.matrix(I - ntio11_Guangdong)),np.matrix(possible_Guangdong11.loc[:,"10"]).transpose()))

#Gansu
x_Gansu98 = pd.DataFrame(np.matmul(np.linalg.inv(np.matrix(I - ntio98_Gansu)),np.matrix(possible_Gansu98.loc[:,"10"]).transpose()))
x_Gansu99 = pd.DataFrame(np.matmul(np.linalg.inv(np.matrix(I - ntio99_Gansu)),np.matrix(possible_Gansu99.loc[:,"10"]).transpose()))
x_Gansu00 = pd.DataFrame(np.matmul(np.linalg.inv(np.matrix(I - ntio00_Gansu)),np.matrix(possible_Gansu00.loc[:,"10"]).transpose()))
x_Gansu01 = pd.DataFrame(np.matmul(np.linalg.inv(np.matrix(I - ntio01_Gansu)),np.matrix(possible_Gansu01.loc[:,"10"]).transpose()))
x_Gansu02 = pd.DataFrame(np.matmul(np.linalg.inv(np.matrix(I - ntio02_Gansu)),np.matrix(possible_Gansu02.loc[:,"10"]).transpose()))
x_Gansu03 = pd.DataFrame(np.matmul(np.linalg.inv(np.matrix(I - ntio03_Gansu)),np.matrix(possible_Gansu03.loc[:,"10"]).transpose()))
x_Gansu04 = pd.DataFrame(np.matmul(np.linalg.inv(np.matrix(I - ntio04_Gansu)),np.matrix(possible_Gansu04.loc[:,"10"]).transpose()))
x_Gansu05 = pd.DataFrame(np.matmul(np.linalg.inv(np.matrix(I - ntio05_Gansu)),np.matrix(possible_Gansu05.loc[:,"10"]).transpose()))
x_Gansu06 = pd.DataFrame(np.matmul(np.linalg.inv(np.matrix(I - ntio06_Gansu)),np.matrix(possible_Gansu06.loc[:,"10"]).transpose()))
x_Gansu07 = pd.DataFrame(np.matmul(np.linalg.inv(np.matrix(I - ntio07_Gansu)),np.matrix(possible_Gansu07.loc[:,"10"]).transpose()))
x_Gansu08 = pd.DataFrame(np.matmul(np.linalg.inv(np.matrix(I - ntio08_Gansu)),np.matrix(possible_Gansu08.loc[:,"10"]).transpose()))
x_Gansu09 = pd.DataFrame(np.matmul(np.linalg.inv(np.matrix(I - ntio09_Gansu)),np.matrix(possible_Gansu09.loc[:,"10"]).transpose()))
x_Gansu10 = pd.DataFrame(np.matmul(np.linalg.inv(np.matrix(I - ntio10_Gansu)),np.matrix(possible_Gansu10.loc[:,"10"]).transpose()))
x_Gansu11 = pd.DataFrame(np.matmul(np.linalg.inv(np.matrix(I - ntio11_Gansu)),np.matrix(possible_Gansu11.loc[:,"10"]).transpose()))

#Jiangsu
x_Jiangsu98 = pd.DataFrame(np.matmul(np.linalg.inv(np.matrix(I - ntio98_Jiangsu)),np.matrix(possible_Jiangsu98.loc[:,"10"]).transpose()))
x_Jiangsu99 = pd.DataFrame(np.matmul(np.linalg.inv(np.matrix(I - ntio99_Jiangsu)),np.matrix(possible_Jiangsu99.loc[:,"10"]).transpose()))
x_Jiangsu00 = pd.DataFrame(np.matmul(np.linalg.inv(np.matrix(I - ntio00_Jiangsu)),np.matrix(possible_Jiangsu00.loc[:,"10"]).transpose()))
x_Jiangsu01 = pd.DataFrame(np.matmul(np.linalg.inv(np.matrix(I - ntio01_Jiangsu)),np.matrix(possible_Jiangsu01.loc[:,"10"]).transpose()))
x_Jiangsu02 = pd.DataFrame(np.matmul(np.linalg.inv(np.matrix(I - ntio02_Jiangsu)),np.matrix(possible_Jiangsu02.loc[:,"10"]).transpose()))
x_Jiangsu03 = pd.DataFrame(np.matmul(np.linalg.inv(np.matrix(I - ntio03_Jiangsu)),np.matrix(possible_Jiangsu03.loc[:,"10"]).transpose()))
x_Jiangsu04 = pd.DataFrame(np.matmul(np.linalg.inv(np.matrix(I - ntio04_Jiangsu)),np.matrix(possible_Jiangsu04.loc[:,"10"]).transpose()))
x_Jiangsu05 = pd.DataFrame(np.matmul(np.linalg.inv(np.matrix(I - ntio05_Jiangsu)),np.matrix(possible_Jiangsu05.loc[:,"10"]).transpose()))
x_Jiangsu06 = pd.DataFrame(np.matmul(np.linalg.inv(np.matrix(I - ntio06_Jiangsu)),np.matrix(possible_Jiangsu06.loc[:,"10"]).transpose()))
x_Jiangsu07 = pd.DataFrame(np.matmul(np.linalg.inv(np.matrix(I - ntio07_Jiangsu)),np.matrix(possible_Jiangsu07.loc[:,"10"]).transpose()))
x_Jiangsu08 = pd.DataFrame(np.matmul(np.linalg.inv(np.matrix(I - ntio08_Jiangsu)),np.matrix(possible_Jiangsu08.loc[:,"10"]).transpose()))
x_Jiangsu09 = pd.DataFrame(np.matmul(np.linalg.inv(np.matrix(I - ntio09_Jiangsu)),np.matrix(possible_Jiangsu09.loc[:,"10"]).transpose()))
x_Jiangsu10 = pd.DataFrame(np.matmul(np.linalg.inv(np.matrix(I - ntio10_Jiangsu)),np.matrix(possible_Jiangsu10.loc[:,"10"]).transpose()))
x_Jiangsu11 = pd.DataFrame(np.matmul(np.linalg.inv(np.matrix(I - ntio11_Jiangsu)),np.matrix(possible_Jiangsu11.loc[:,"10"]).transpose()))

#Qinghai
x_Qinghai98 = pd.DataFrame(np.matmul(np.linalg.inv(np.matrix(I - ntio98_Qinghai)),np.matrix(possible_Qinghai98.loc[:,"10"]).transpose()))
x_Qinghai99 = pd.DataFrame(np.matmul(np.linalg.inv(np.matrix(I - ntio99_Qinghai)),np.matrix(possible_Qinghai99.loc[:,"10"]).transpose()))
x_Qinghai00 = pd.DataFrame(np.matmul(np.linalg.inv(np.matrix(I - ntio00_Qinghai)),np.matrix(possible_Qinghai00.loc[:,"10"]).transpose()))
x_Qinghai01 = pd.DataFrame(np.matmul(np.linalg.inv(np.matrix(I - ntio01_Qinghai)),np.matrix(possible_Qinghai01.loc[:,"10"]).transpose()))
x_Qinghai02 = pd.DataFrame(np.matmul(np.linalg.inv(np.matrix(I - ntio02_Qinghai)),np.matrix(possible_Qinghai02.loc[:,"10"]).transpose()))
x_Qinghai03 = pd.DataFrame(np.matmul(np.linalg.inv(np.matrix(I - ntio03_Qinghai)),np.matrix(possible_Qinghai03.loc[:,"10"]).transpose()))
x_Qinghai04 = pd.DataFrame(np.matmul(np.linalg.inv(np.matrix(I - ntio04_Qinghai)),np.matrix(possible_Qinghai04.loc[:,"10"]).transpose()))
x_Qinghai05 = pd.DataFrame(np.matmul(np.linalg.inv(np.matrix(I - ntio05_Qinghai)),np.matrix(possible_Qinghai05.loc[:,"10"]).transpose()))
x_Qinghai06 = pd.DataFrame(np.matmul(np.linalg.inv(np.matrix(I - ntio06_Qinghai)),np.matrix(possible_Qinghai06.loc[:,"10"]).transpose()))
x_Qinghai07 = pd.DataFrame(np.matmul(np.linalg.inv(np.matrix(I - ntio07_Qinghai)),np.matrix(possible_Qinghai07.loc[:,"10"]).transpose()))
x_Qinghai08 = pd.DataFrame(np.matmul(np.linalg.inv(np.matrix(I - ntio08_Qinghai)),np.matrix(possible_Qinghai08.loc[:,"10"]).transpose()))
x_Qinghai09 = pd.DataFrame(np.matmul(np.linalg.inv(np.matrix(I - ntio09_Qinghai)),np.matrix(possible_Qinghai09.loc[:,"10"]).transpose()))
x_Qinghai10 = pd.DataFrame(np.matmul(np.linalg.inv(np.matrix(I - ntio10_Qinghai)),np.matrix(possible_Qinghai10.loc[:,"10"]).transpose()))
x_Qinghai11 = pd.DataFrame(np.matmul(np.linalg.inv(np.matrix(I - ntio11_Qinghai)),np.matrix(possible_Qinghai11.loc[:,"10"]).transpose()))


#Export into excel regional Outputs (x)
#Guangdong
writer = pd.ExcelWriter('Possible_regional_outputs_Guangdong.xlsx') 
x_Guangdong98.to_excel(writer, sheet_name='x98', index=False) 
x_Guangdong99.to_excel(writer, sheet_name='x99', index=False) 
x_Guangdong00.to_excel(writer, sheet_name='x00', index=False) 
x_Guangdong01.to_excel(writer, sheet_name='x01', index=False) 
x_Guangdong02.to_excel(writer, sheet_name='x02', index=False) 
x_Guangdong03.to_excel(writer, sheet_name='x03', index=False) 
x_Guangdong04.to_excel(writer, sheet_name='x04', index=False) 
x_Guangdong05.to_excel(writer, sheet_name='x05', index=False) 
x_Guangdong06.to_excel(writer, sheet_name='x06', index=False) 
x_Guangdong07.to_excel(writer, sheet_name='x07', index=False) 
x_Guangdong08.to_excel(writer, sheet_name='x08', index=False) 
x_Guangdong09.to_excel(writer, sheet_name='x09', index=False) 
x_Guangdong10.to_excel(writer, sheet_name='x10', index=False) 
x_Guangdong11.to_excel(writer, sheet_name='x11', index=False) 
writer.save()
writer.close()

#Gansu
writer = pd.ExcelWriter('Possible_regional_outputs_Gansu.xlsx') 
x_Gansu98.to_excel(writer, sheet_name='x98', index=False) 
x_Gansu99.to_excel(writer, sheet_name='x99', index=False) 
x_Gansu00.to_excel(writer, sheet_name='x00', index=False) 
x_Gansu01.to_excel(writer, sheet_name='x01', index=False) 
x_Gansu02.to_excel(writer, sheet_name='x02', index=False) 
x_Gansu03.to_excel(writer, sheet_name='x03', index=False) 
x_Gansu04.to_excel(writer, sheet_name='x04', index=False) 
x_Gansu05.to_excel(writer, sheet_name='x05', index=False) 
x_Gansu06.to_excel(writer, sheet_name='x06', index=False) 
x_Gansu07.to_excel(writer, sheet_name='x07', index=False) 
x_Gansu08.to_excel(writer, sheet_name='x08', index=False) 
x_Gansu09.to_excel(writer, sheet_name='x09', index=False) 
x_Gansu10.to_excel(writer, sheet_name='x10', index=False) 
x_Gansu11.to_excel(writer, sheet_name='x11', index=False) 
writer.save()
writer.close()

#Jiangsu
writer = pd.ExcelWriter('Possible_regional_outputs_Jiangsu.xlsx') 
x_Jiangsu98.to_excel(writer, sheet_name='x98', index=False) 
x_Jiangsu99.to_excel(writer, sheet_name='x99', index=False) 
x_Jiangsu00.to_excel(writer, sheet_name='x00', index=False) 
x_Jiangsu01.to_excel(writer, sheet_name='x01', index=False) 
x_Jiangsu02.to_excel(writer, sheet_name='x02', index=False) 
x_Jiangsu03.to_excel(writer, sheet_name='x03', index=False) 
x_Jiangsu04.to_excel(writer, sheet_name='x04', index=False) 
x_Jiangsu05.to_excel(writer, sheet_name='x05', index=False) 
x_Jiangsu06.to_excel(writer, sheet_name='x06', index=False) 
x_Jiangsu07.to_excel(writer, sheet_name='x07', index=False) 
x_Jiangsu08.to_excel(writer, sheet_name='x08', index=False) 
x_Jiangsu09.to_excel(writer, sheet_name='x09', index=False) 
x_Jiangsu10.to_excel(writer, sheet_name='x10', index=False) 
x_Jiangsu11.to_excel(writer, sheet_name='x11', index=False) 
writer.save()
writer.close()

#Qinghai
writer = pd.ExcelWriter('Possible_regional_outputs_Qinghai.xlsx') 
x_Qinghai98.to_excel(writer, sheet_name='x98', index=False) 
x_Qinghai99.to_excel(writer, sheet_name='x99', index=False) 
x_Qinghai00.to_excel(writer, sheet_name='x00', index=False) 
x_Qinghai01.to_excel(writer, sheet_name='x01', index=False) 
x_Qinghai02.to_excel(writer, sheet_name='x02', index=False) 
x_Qinghai03.to_excel(writer, sheet_name='x03', index=False) 
x_Qinghai04.to_excel(writer, sheet_name='x04', index=False) 
x_Qinghai05.to_excel(writer, sheet_name='x05', index=False) 
x_Qinghai06.to_excel(writer, sheet_name='x06', index=False) 
x_Qinghai07.to_excel(writer, sheet_name='x07', index=False) 
x_Qinghai08.to_excel(writer, sheet_name='x08', index=False) 
x_Qinghai09.to_excel(writer, sheet_name='x09', index=False) 
x_Qinghai10.to_excel(writer, sheet_name='x10', index=False) 
x_Qinghai11.to_excel(writer, sheet_name='x11', index=False) 
writer.save()
writer.close()