import pandas as pd
pd.set_option('display.width',100)
#pd.set_option("display.max_rows",100)
pd.set_option('display.max_columns',400)
import numpy as np

import seaborn as sb
from matplotlib import pyplot as plt
from matplotlib.dates import DateFormatter

import datetime
import matplotlib.dates

import nltk
#nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist as fdist
import re
from scipy.stats import ttest_ind

import warnings
warnings.filterwarnings('ignore')

customer_data=pd.read_csv('QVI_purchase_behaviour.csv')
transaction_data=pd.read_csv('QVI_transaction_data.csv')

transaction_data['DATE']=pd.TimedeltaIndex(transaction_data['DATE'],unit='d')+datetime.datetime(1899,12,30)

transaction_data["PACK_SIZE"]=transaction_data['PROD_NAME'].str.extract("(\d+)")
transaction_data["PACK_SIZE"]=transaction_data['PACK_SIZE'].astype(int)

def text_clean(text):
    text=re.sub('[&/]'," ",text)
    text=re.sub("\d\w"," ",text)
    return text
transaction_data["PROD_NAME"]=transaction_data["PROD_NAME"].apply(text_clean)
cleanname=transaction_data["PROD_NAME"]
string="".join(cleanname)
product=word_tokenize(string)

for word in range(len(product)):

    if product[word][0]=='g' and product[word][1].isupper():
        product[word]=product[word].lstrip(product[word][0])

wordfreq=fdist(product)

freq_d=pd.DataFrame(list(wordfreq.items()),columns=['Name','Frequency']).sort_values(by ='Frequency',ascending=False)
print(freq_d.tail(50))
transaction_data["PROD_NAME"]=transaction_data["PROD_NAME"].apply(lambda x:x.lower())
transaction_data=transaction_data[~transaction_data["PROD_NAME"].str.contains('salsa')]
transaction_data["PROD_NAME"]=transaction_data["PROD_NAME"].apply(lambda x:x.title())


#print(transaction_data.describe())


#print(transaction_data['PROD_QTY'].value_counts())
#print(transaction_data.loc[transaction_data["PROD_QTY"]==200])
transaction_data.drop(transaction_data.index[transaction_data["LYLTY_CARD_NBR"]==226000], inplace=True)
#print(list(transaction_data.index[transaction_data["LYLTY_CARD_NBR"]==226000]))# returns the indexes of 226000 purchase
customer_data.drop(transaction_data.index[transaction_data["LYLTY_CARD_NBR"]==226000], inplace = True)
#print(transaction_data.describe()[["PROD_QTY"]]) #single square brc gives data type

transaction_data["DATE"].nunique()
stdate=transaction_data["DATE"].min()
eddate=transaction_data["DATE"].max()
#print(f'Start Date: {stdate}. End Date: {eddate}')
#print(transaction_data["DATE"])
#print(pd.date_range(start='2018-07-01',end='2019-06-30').difference(transaction_data["DATE"]))

a=pd.pivot_table(transaction_data,values='TOT_SALES',index='DATE',aggfunc='sum')

b=pd.DataFrame(index=pd.date_range(start='2018-07-01',end='2019-06-30'))
b['TOT_SALES']=0

c=a+b
c.fillna(0,inplace=True)
c.index.name="Date"
c.rename(columns={'TOT_SALES':'Total Sales'},inplace=True)
print(type(c))
#print(c.loc['2018-12-25'])
#c.to_csv('C:\\Users\\magla\\Desktop\\data4.csv')

timeline = c.index
graph = c['Total Sales']

fig, ax = plt.subplots(figsize = (10, 5))
ax.plot(timeline, graph)

date_form = DateFormatter("%Y-%m")
ax.xaxis.set_major_formatter(date_form)
plt.title('Total Sales from July 2018 to June 2019')
plt.xlabel('Time')
plt.ylabel('Total Sales')
#plt.show()
dec=c[(c.index>'2018-11-30')&(c.index<'2019-01-01')]


#plt.figure(figsize=(15,5))
#ax = sb.lineplot(data=dec,x=dec.index, y='Total Sales')


#dec.reset_index(drop=True, inplace=True)
#print(dec.head)
#dec['Date']=dec.index+1
#print(dec.items)

#sb.barplot(x='Date',y='Total Sales',data=dec)
#plt.show()

#print(transaction_data['PACK_SIZE'].unique())
#transaction_data.groupby('PACK_SIZE').TOT_SALES.count().sort_values(ascending=False)
#plt.figure(figsize=(10,5))
#sb.histplot(transaction_data['PACK_SIZE'],bins=15)
#plt.show()
brand = transaction_data['PROD_NAME'].str.partition()
print(brand)
transaction_data["BRAND"]=brand[0]
print(transaction_data["BRAND"].unique())

transaction_data["BRAND"].replace('Ncc','Natural',inplace=True)
transaction_data["BRAND"].replace('Smith','Smiths',inplace=True)
transaction_data["BRAND"].replace(['Grnwves','Grain'],'Grainwaves',inplace=True)
transaction_data["BRAND"].replace('Rrd','Red',inplace=True)
transaction_data["BRAND"].replace('Dorito','Doritos',inplace=True)
transaction_data["BRAND"].replace('Snbts','Sunbites',inplace=True)
transaction_data["BRAND"].replace('Infzns','Infuzions',inplace=True)
transaction_data["BRAND"].replace('Ww','Woolworths',inplace=True)

#print(transaction_data["BRAND"].unique())
print("=================TOTAL SALES BY BRAND===================")
print(transaction_data.groupby('BRAND').TOT_SALES.sum().sort_values(ascending=False))
print("==========CUSTOMER DATA=================")

#print(customer_data["LYLTY_CARD_NBR"].nunique()) =72636
#print(customer_data[customer_data["LYLTY_CARD_NBR"].duplicated()]) #no dupes
print("=================LIFESTAGE COUNT=====================")
print(customer_data['LIFESTAGE'].value_counts().sort_values(ascending=False))
print("==============PREMIUM CUSTOMER COUNT=================")
print(customer_data["PREMIUM_CUSTOMER"].value_counts().sort_values(ascending=False))
combinedata=pd.merge(transaction_data,customer_data)
#print(combinedata.isnull().sum())
#print(combinedata.head(20))
print("==================TOTAL SALES PER FAMILY PER PREMIUM==================")
print(combinedata.groupby(['PREMIUM_CUSTOMER',"LIFESTAGE"]).TOT_SALES.sum().sort_values(ascending=False))
salesplot=pd.DataFrame(combinedata.groupby(['LIFESTAGE',"PREMIUM_CUSTOMER"]).TOT_SALES.sum())
salesplot.unstack().plot(kind = 'bar', stacked = True, figsize = (15, 7), title = 'Total Sales by Customer Segment')
plt.ylabel('Total Sales')
plt.legend(['Budget', 'Mainstream', 'Premium'], loc = 1)
#plt.show()
print("===========NUMBER OF LOYALTY CARDS PER LIFESTAGES PER PREMIUM=====================")

print(combinedata.groupby(["LIFESTAGE","PREMIUM_CUSTOMER"])['LYLTY_CARD_NBR'].nunique().sort_values(ascending=False))
custplot=pd.DataFrame(combinedata.groupby(["LIFESTAGE","PREMIUM_CUSTOMER"]).LYLTY_CARD_NBR.nunique())
custplot.unstack().plot(kind='bar',stacked=True, figsize=(15,7),title = 'Loyalty Cards per Lifestage')
plt.legend(['BUDGET','MAINSTREAM','PREMIUM'],loc=2)
#plt.show()
avg_units = combinedata.groupby(['PREMIUM_CUSTOMER', 'LIFESTAGE']).PROD_QTY.sum() / combinedata.groupby(['PREMIUM_CUSTOMER', 'LIFESTAGE']).LYLTY_CARD_NBR.nunique()
print("================AVERAGE PURCHASES PER FAMILY PER PREMIUM====================")
df=pd.DataFrame(avg_units).reset_index()
df.columns = ['Premium','Lifestage',"Average Purchase"]


print(df.sort_values(by='Average Purchase',ascending=False))

avg_units.unstack().plot(kind='bar',stacked=True,figsize=(18,8),title='Average sales per Premium')
#plt.show()

avg_price=combinedata.groupby(['PREMIUM_CUSTOMER', 'LIFESTAGE']).TOT_SALES.sum() / combinedata.groupby(['PREMIUM_CUSTOMER', 'LIFESTAGE']).PROD_QTY.sum()
avg_price.unstack().plot(kind='bar',figsize=(15,8),title='Average Profit')
#plt.show()
print("======OLD FAMILIY POPULAR BRAND=======")
oldlove=combinedata[(combinedata["LIFESTAGE"]=='OLDER FAMILIES')]
younglove=combinedata[(combinedata["LIFESTAGE"]=='YOUNG FAMILIES')]
oldlove=oldlove.BRAND.value_counts()
younglove=younglove.BRAND.value_counts()

print(oldlove.sort_values(ascending=False))
print("=======YOUNG FAMILY POPULAR BRAND=============")
print(younglove.sort_values(ascending=False))

priceperunit=combinedata
priceperunit["PRICE"]=priceperunit["TOT_SALES"]/priceperunit["PROD_QTY"]
print("===============PRICE FOR CHIPS==================")
print(priceperunit.head(10))

mainstream = priceperunit[(priceperunit["PREMIUM_CUSTOMER"]=='Mainstream') & ((priceperunit["LIFESTAGE"]=='YOUNG SINGLES/COUPLES')|(priceperunit["LIFESTAGE"]=='MIDAGE SINGLES/COUPLES'))]
nonmainstream = priceperunit[(priceperunit["PREMIUM_CUSTOMER"]!='Mainstream') & ((priceperunit["LIFESTAGE"]=='YOUNG SINGLES/COUPLES')|(priceperunit["LIFESTAGE"]=='MIDAGE SINGLES/COUPLES'))]
print("==========Avg Prices===============")
print("Mainstream Avg is ${:.2f}".format(np.mean(mainstream["PRICE"])))
print("Non-Mainstream Avg is ${:.2f}".format(np.mean(nonmainstream["PRICE"])))
plt.hist(mainstream["PRICE"], label=mainstream)
plt.xlabel("Price")
plt.legend('Mainstream', "Non-Mainstream")
plt.hist(nonmainstream["PRICE"], label=nonmainstream)
#plt.figure(figsize=(10,5))
plt.title=("Mains vs Non Mains")


#plt.show()

print(ttest_ind(mainstream["PRICE"], nonmainstream["PRICE"]))
target = combinedata.loc[(combinedata['LIFESTAGE'] == 'YOUNG SINGLES/COUPLES') & (combinedata['PREMIUM_CUSTOMER'] == 'Mainstream'), :]
nonTarget = combinedata.loc[(combinedata['LIFESTAGE'] != 'YOUNG SINGLES/COUPLES' ) & (combinedata['PREMIUM_CUSTOMER'] != 'Mainstream'), :]
print("=================TARGET================")
print(target.head())
target_brand=target.loc[:,["BRAND","PROD_QTY"]]
nontarget_brand=nonTarget.loc[:,["BRAND","PROD_QTY"]]
print("==============TARGET BRAND================")
print(target_brand)
target_sum=target_brand["PROD_QTY"].sum()
print(target_sum)
nontarget_sum=nontarget_brand["PROD_QTY"].sum()
target_brand["TARGET_BRAND_AFFINITY"]=target_brand["PROD_QTY"]/target_sum
nontarget_brand["TARGET_BRAND_AFFINITY"]=nontarget_brand["PROD_QTY"]/nontarget_sum
target_brand=pd.DataFrame(target_brand.groupby("BRAND").TARGET_BRAND_AFFINITY.sum())
nontarget_brand=pd.DataFrame(nontarget_brand.groupby("BRAND").TARGET_BRAND_AFFINITY.sum())
#print(target_brand.head(20))
target_brand["YOUNG_AFFINITY"]=target_brand["TARGET_BRAND_AFFINITY"]/nontarget_brand["TARGET_BRAND_AFFINITY"]
print(target_brand.head(20).sort_values(by='YOUNG_AFFINITY',ascending=False))




print("===========PACK SIZE AFFINITY================")
pack_size=pd.DataFrame(target.PACK_SIZE.value_counts())
pack_size.sort_values(by="PACK_SIZE" ,ascending=False)
print(pack_size.head(10))
pack_size.unstack().plot(kind='bar',stacked=True, figsize=(15,7),title='Pack Size Affinity')
#plt.show()

targetsize = target.loc[:,["PACK_SIZE","PROD_QTY"]]
targetSum = targetsize['PROD_QTY'].sum()
targetsize['Target Pack Affinity'] = targetsize['PROD_QTY'] / targetSum
targetsize = pd.DataFrame(targetsize.groupby('PACK_SIZE')['Target Pack Affinity'].sum())

# Non-target segment
nonTargetsize = nonTarget.loc[:, ['PACK_SIZE', 'PROD_QTY']]
nonTargetSum = nonTargetsize['PROD_QTY'].sum()
nonTargetsize['Non-Target Pack Affinity'] = nonTargetsize['PROD_QTY'] / nonTargetSum
nonTargetsize = pd.DataFrame(nonTargetsize.groupby('PACK_SIZE')['Non-Target Pack Affinity'].sum())

pack_proportions=pd.merge(targetsize,nonTargetsize, left_index=True,right_index=True)
pack_proportions["Young Affinity"]=pack_proportions["Target Pack Affinity"]/pack_proportions["Non-Target Pack Affinity"]
print(pack_proportions.sort_values(by="Young Affinity",ascending=False).head(10))
print('1')
print(combinedata.loc[combinedata["PACK_SIZE"]==330])

storepop=combinedata.loc[:,["STORE_NBR","PREMIUM_CUSTOMER","LIFESTAGE"]]
storepop=pd.DataFrame(combinedata.STORE_NBR.value_counts())
storepop.columns=['Purchases']
print("============STORE POPULARITY===============")
print(storepop.head(20))
print("============ LOW STORE POPULARITY===============")
print(storepop.tail(20))

best_store=combinedata.loc[combinedata["STORE_NBR"]==227]
print("==========BRAND POPULARITY IN MOST POPULAR STORE======================")
print(best_store.BRAND.value_counts())
#plt.show()
