
# coding: utf-8

# ## TASK -D
# ** Our goal in this Notebook is to cluster our customers to get insights in:
# 
# * Increasing revenue (Knowing customers who present most of our revenue)
# * Increasing customer retention
# * Discovering Trends and patterns
# * Defining customers at risk
# * We will do RFM Analysis ==
# 
# * RFM Analysis answers these questions:
# 
# * Who are our best customers?
# * Who has the potential to be converted in more profitable customers?
# * Which customers we must retain?
# * Which group of customers is most likely to respond to our current campaign? **
# 
# 

# ## IMPORTING PACKAGES

# In[7]:


import numpy as np
import pandas as pd
import os
import datetime as dt


# In[8]:


#loading the dataset
df=pd.read_excel("Online Retail.xlsx")


# In[9]:


#lets see first 5 customer data
df.head()


# In[10]:


#In info you can see total of 541909 enteries
df.info()


# ## PREPARING THE DATA

# In[12]:


#As to clean data i am taking the customer data of United kingdom as it contain most of historical data.Showing 495478 customer after it.
df_uk=df[df['Country']=='United Kingdom']
df_uk.shape


# In[13]:


#removing the canceled orders
df_uk=df_uk[df_uk['Quantity']>0]
df_uk.shape


# In[14]:


#remove customer id with null value
df_uk.dropna(subset=['CustomerID'],how='all',inplace=True)
df_uk.shape


# In[15]:


##restrict the data to one full year because it's better to use a metric per Months or Years in RFM
df_uk=df_uk[df_uk['InvoiceDate']>='2010-12-09']
df_uk.shape


# In[16]:


#lets see unique value of attributes

print("number of transactions=",df_uk['InvoiceNo'].nunique())
print("number of products brought=",df_uk['StockCode'].nunique())
print("number of customers=",df_uk['CustomerID'].nunique())


# # RFM(RECENCY,FREQUENCY,MONETORY) ANALYSIS
# 
# * RECENCY (R): Days since last purchase
# * FREQUENCY (F): Total number of purchases
# * MONETARY VALUE (M): Total money this customer spent.
# 
# 
# 
# 

# In[17]:


#RECENCY (it tells the number of days before which the customer had purchased)


# In[70]:


#last date available in our dataset
df_uk['InvoiceDate'].max()


# In[71]:


df_uk.head()


# In[72]:


#create a new column called date which contains the date of invoice only
df_uk['date']=df_uk['InvoiceDate'].dt.date


# In[73]:


df_uk.head()


# In[74]:



#group by customers and check last date of purshace
recency_df = df_uk.groupby(by='CustomerID',as_index=False)['date'].max()
recency_df.columns=['CustomerID','LastPurchasedate']
recency_df.head()


# In[75]:


now=dt.date(2011,12,9)
print(now)


# In[76]:


#recency calculation


# In[77]:


recency_df['recency']=recency_df['LastPurchasedate'].apply(lambda x: (now - x).days)
recency_df.drop('LastPurchasedate',axis=1,inplace=True)


# In[78]:


recency_df.head()


# In[79]:


#FREQUENCY(it tells how many times customer had purchased from us)


# In[80]:


df_uk.head()


# In[81]:


copy_uk=df_uk
copy_uk.drop_duplicates(subset=['InvoiceNo','CustomerID'],keep='first',inplace=True)
frequency_uk=copy_uk.groupby(by='CustomerID',as_index=False)['InvoiceNo'].count()
frequency_uk.columns=['CustomerID','Frequency']
frequency_uk.head()


# In[82]:


#MONETORY(it tells us that how much money the customer had spend)
#To do that, first, we will create a new column total cost to have the total price per invoice.


# In[83]:


#create column total cost
df_uk['Totalcost']=df_uk['Quantity']*df_uk['UnitPrice']


# In[84]:


monetory_df = df_uk.groupby(by='CustomerID',as_index=False).agg({'Totalcost':'sum'})
monetory_df.columns=['CustomerID','Monetory']
monetory_df.head()


# # RFM TABLE

# In[85]:


#merging recency ,frequency
temp_df=recency_df.merge(frequency_uk,on='CustomerID')
temp_df.head()


# In[86]:


#merging recency ,frequency,monetory
temp_df1=temp_df.merge(monetory_df,on='CustomerID')
temp_df1.head()


# In[87]:


#RFM TABLE CORRECTNESS VERIFICATION


# In[88]:


df_uk[df_uk['CustomerID']== 12346]


# In[89]:


(now-dt.date(2011,1,18)).days==325


# ## CUSTOMER SEGMENT WITH RFM MODEL

# In[90]:


#PARETO PRINCIPLE(80-20 RULE)(it says 20% customer are responsible for 80% revenue)
#and we have to find out those 20 % so that we could focus more on those customer.


# In[91]:


pareto=temp_df1['Monetory'].sum()*0.8
print("the 80% of total revenue is:",round(pareto,2))


# In[92]:


customer_rank=temp_df1
customer_rank['Rank']=customer_rank['Monetory'].rank(ascending=0)
customer_rank.head()


# In[93]:


#top customers


# In[94]:


customer_rank.sort_values('Rank',ascending=True)


# In[95]:


#get top 20% of the customers
top20 = 3863*20/100
top20


# In[96]:


#sum the monetary values over the customer with rank <=773
revenueBy20=customer_rank[customer_rank['Rank']<=772]['Monetory'].sum()
revenueBy20


# In[97]:



#In our case, the 80% of total revenue is not achieved by the 20% of TOP customers but approximately, 
#it does, because they are less than our 20% TOP customers who achieve it. 
#It would be interesting to study this group of customers because they are those who make our most revenue.


# # Applying RFM score formula
# 
# The simplest way to create customers segments from RFM Model is to use Quartiles. We assign a score from 1 to 4 to Recency, Frequency and Monetary. Four is the best/highest value, and one is the lowest/worst value. A final RFM score is calculated simply by combining individual RFM score numbers.

# In[98]:


#QUARTILES


# In[99]:


temp_df1.drop('CustomerID',axis=1,inplace=True)
quantiles = temp_df1.quantile(q=[.25,.50,.75])
quantiles


# In[100]:


quantiles.to_dict()


# In[101]:


#CREATION OF RFM SEGMENTATION TABLE (we will create two segmentation classes because high recency is bad,while high frequency
# and monetory is good)


# In[102]:


#Arguements (x=value,m=recency,frequency,monetory,d=quartiles_dict)
def RScore(x,m,d):
    if x<= d[m][.25]:
        return 4
    elif x<=d[m][.50]:
        return 3
    elif x<=d[m][.75]:
        return 2
    else:
        return 1
    
#Arguements (x=value,m=recency,frequency,monetory,d=quartiles_dict) 

    
def FScore(x,m,d):
    if x<= d[m][.25]:
           return 1
    elif x<=d[m][.50]:
            return 2
    elif x<=d[m][.75]:
            return 3
    else:
            return 4
    


# In[103]:


#creating rfm segmentation table


# In[104]:


temp_df1.head()


# In[105]:


rfm_segment = temp_df1
rfm_segment['R_Quartile']=rfm_segment['recency'].apply(RScore,args=('recency',quantiles,))
rfm_segment['F_Quartile']=rfm_segment['Frequency'].apply(FScore,args=('Frequency',quantiles,))
rfm_segment['M_Quartile']=rfm_segment['Monetory'].apply(FScore,args=('Monetory',quantiles,))


# In[106]:


rfm_segment.head()


# In[107]:


rfm_segment['RFMScore'] = rfm_segment.R_Quartile.map(str)  + rfm_segment.F_Quartile.map(str) + rfm_segment.M_Quartile.map(str)
rfm_segment.head()


# In[108]:


#now
#best recency score=4: most recently purchase
#best frequency score=4: most quantity purchase
#best monetory score=4: most money spend
#that means customer who have RFMScore =444 are the best customer or top customer 


# In[109]:


rfm_segment[rfm_segment['RFMScore']=='444'].sort_values('Monetory',ascending=False).head(10)


# In[110]:


#now we can divide customer into different segments on the basis of RFMScore


# In[111]:



print("top Customers: ",len(rfm_segment[rfm_segment['RFMScore']=='444']))
print('Loyal Customers: ',len(rfm_segment[rfm_segment['F_Quartile']==4]))
print("Big Spenders: ",len(rfm_segment[rfm_segment['M_Quartile']==4]))
print('Almost Lost: ', len(rfm_segment[rfm_segment['RFMScore']=='244']))
print('Lost Customers: ',len(rfm_segment[rfm_segment['RFMScore']=='144']))
print('Lost Cheap Customers: ',len(rfm_segment[rfm_segment['RFMScore']=='111']))


# In[113]:



#Now that we knew our customers segments we can choose how to target or deal with each segment.

#For example:

#Best Customers - Champions: Reward them. They can be early adopters to new products. Suggest them "Refer a friend".

#At Risk: Send them personalized emails to encourage them to shop.

