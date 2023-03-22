#!/usr/bin/env python
# coding: utf-8

# In[1]:

from yahoo_fin import stock_info as si
import requests as regs
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from stocksymbol import StockSymbol
import glob
from pathlib import Path
import re
import os
import shutil
import reticker
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import random
import datetime
import matplotlib.pyplot as plt
from pprint import pprint
tf.get_logger().setLevel('ERROR')
from GoogleNews import GoogleNews
googlenews = GoogleNews()
from gnewsclient import gnewsclient
api_key = '51f442a4fab8401ca0e0bc020ee6ae90'


# In[3]:


domains_string = 'hello'
todays_date = datetime.datetime.today()
todays_date_formatted = todays_date.strftime ('20%y-%m-%d')
todays_date_formatted = str(todays_date_formatted)
url1 = ('https://newsapi.org/v2/everything?'
       'q=stocks&'
       f'from={todays_date_formatted}&'
       'sortBy=relevancy&'
       'domains=marketwatch.com,investing.com,seekingalpha.com,fool.co.uk,ino.com/blog,moneycontrol.com,news.alphastreet.com,equitymaster.com,indiainfoline.com/markets/news,stocksnewsfeed.com,ragingbull.com,scanz.com/blog,wsj.com,nytimes.com'
       'page=1&'
       'apiKey=51f442a4fab8401ca0e0bc020ee6ae90')

url2 = ('https://newsapi.org/v2/everything?'
       'q=stocks&'
       'domains=marketwatch.com,investing.com,seekingalpha.com,fool.co.uk,ino.com/blog,moneycontrol.com,news.alphastreet.com,equitymaster.com,indiainfoline.com/markets/news,stocksnewsfeed.com,ragingbull.com,scanz.com/blog,wsj.com,nytimes.com'
       f'from={todays_date_formatted}&'
       'sortBy=relevancy&'
       'page=2&'
       'apiKey=51f442a4fab8401ca0e0bc020ee6ae90')

url3 = ('https://newsapi.org/v2/everything?'
       'q=stocks&'
       'domains=marketwatch.com,investing.com,seekingalpha.com,fool.co.uk,ino.com/blog,moneycontrol.com,news.alphastreet.com,equitymaster.com,indiainfoline.com/markets/news,stocksnewsfeed.com,ragingbull.com,scanz.com/blog,wsj.com,nytimes.com'
       f'from={todays_date_formatted}&'
       'sortBy=relevancy&'
       'page=3&'
       'apiKey=51f442a4fab8401ca0e0bc020ee6ae90')

url4 = ('https://newsapi.org/v2/everything?'
       'q=stocks&'
       'domains=marketwatch.com,investing.com,seekingalpha.com,fool.co.uk,ino.com/blog,moneycontrol.com,news.alphastreet.com,equitymaster.com,indiainfoline.com/markets/news,stocksnewsfeed.com,ragingbull.com,scanz.com/blog,wsj.com,nytimes.com'
       f'from={todays_date_formatted}&'
       'sortBy=relevancy&'
       'page=4&'
       'apiKey=51f442a4fab8401ca0e0bc020ee6ae90')

url5 = ('https://newsapi.org/v2/everything?'
       'q=stocks&'
       'domains=marketwatch.com,investing.com,seekingalpha.com,fool.co.uk,ino.com/blog,moneycontrol.com,news.alphastreet.com,equitymaster.com,indiainfoline.com/markets/news,stocksnewsfeed.com,ragingbull.com,scanz.com/blog,wsj.com,nytimes.com'
       f'from={todays_date_formatted}&'
       'sortBy=relevancy&'
       'page=5&'
       'apiKey=51f442a4fab8401ca0e0bc020ee6ae90')

response1 = regs.get(url1)
response2 = regs.get(url2)
response3 = regs.get(url3)
response4 = regs.get(url4)
response5 = regs.get(url5)

dict1 = response1.json()
dict2 = response2.json()
dict3 = response3.json()
dict4 = response4.json()
dict5 = response5.json()


# In[4]:


url_list = []
articles = dict1['articles'] + dict2['articles'] + dict3['articles'] + dict4['articles'] + dict5['articles']

for item in articles:
    if item not in url_list:
        url_list.append(item['url'])
print(len(url_list))


# In[5]:


def get_html(my_url):
    article = ''
    response = regs.request('GET', url = my_url, verify = True)
    soup = BeautifulSoup(response.content, 'html.parser')
    for item in soup.find_all('p'):
        article = article + item.text
    return article

text_list = [get_html(item) for item in url_list]


# In[6]:


ss = StockSymbol('abc2b86f-3c33-44fc-b01b-926a904bd13c')
symbol_list_us = ss.get_symbol_list(market="US")
def get_sym(big_list):
    new_list = []
    for item in big_list:
        new_list.append(item['symbol'])
    return new_list

my_sym_list = get_sym(symbol_list_us)


# In[7]:


def get_ticker(list):
    ticker_list = []
    pair = []
    for text in list:
        count = 0
        tick = reticker.TickerExtractor().extract(text)
        for item in tick:
            if item in my_sym_list:
                #print('Found')
                match = re.search(item, text)
                pair = []
                start = match.start() - 500
                end = match.end() + 500
                if start < 0:
                    start = 0

                if end > len(text) - 1:
                    end = -1
                chunk = text[start:end]
                pair.append(text[match.start():match.end()])
                pair.append(chunk)
                ticker_list.append(pair)
    return ticker_list
poss_ticker = get_ticker(text_list)


# In[8]:


def create_new_list(list):
    text_list = []
    match_list = []
    for item in list:
        text_list.append(item[1])
        match_list.append(item[0])
    return text_list, match_list

string_list, match_list = create_new_list(poss_ticker)


# In[9]:


saved_model_path = '/Users/michaelscoleri/Desktop/Coding/Personal/Python/Stock_scraper/Scaper3.0_bert'
reloaded_model = tf.saved_model.load(saved_model_path)


# In[10]:


with tf.device('/cpu:0'):
    reloaded_results1 = tf.sigmoid(reloaded_model(tf.constant(string_list)))


# In[11]:


def return_good_match(match_list_new, seq_list_new, string_list):
    hit = []
    for item, seq, string in zip(match_list_new, seq_list_new, string_list):
        if item > .7:
            pair = []
            pair.append(seq)
            pair.append(f'{item[0]:.6f}')
            pair.append(string)
            hit.append(pair)
    return hit

finished_hits = return_good_match(reloaded_results1, match_list, string_list)


# In[12]:


temp_var = datetime.datetime.today()
temp_df = pd.DataFrame(finished_hits)
temp_df['Occur'] = temp_df.groupby(0)[0].transform('size')
temp_df['Date Added'] =  temp_var.strftime ('%b-%d')
temp_df = temp_df.rename(columns = {1:'Score'})


# In[13]:


def transform_to_num(row):
    number = row['Score']
    number = float(number)
    return number

temp_df['Score'] = temp_df.apply(transform_to_num, axis = 1)


# In[14]:



mean_df = (temp_df.groupby(0))['Score'].mean()
mean_df = mean_df.to_frame('Mean')


# In[15]:


temp_df = temp_df.rename(columns={1:'Score'})
temp_df = temp_df.sort_values('Score', ascending=False)
temp_df = pd.merge(temp_df, mean_df, left_on=0, right_on = 0)


# In[16]:


temp_df = temp_df.drop_duplicates(0)


# In[107]:


current_df = pd.read_excel('/Users/michaelscoleri/Desktop/Coding/Personal/Python/Stock_scraper/finished_hits.xlsx')
def remove_old_date(row):
    ten_days = datetime.datetime.today() - datetime.timedelta(days=10)
    ten_days_formatted = ten_days.strftime ('%b-%d')
    if row['Date Added'] == ten_days_formatted:
        current_df.drop(row, axis = 0)
    return

current_df.apply(remove_old_date, axis = 1)


# In[108]:


current_df = pd.concat([current_df, temp_df], axis = 0)


# In[109]:


sum_df = current_df['Occur'].groupby(by=current_df[0]).sum()
current_df = current_df.drop('Occur', axis = 1)
current_df = pd.merge(current_df, sum_df, left_on=0, right_on=0)


# In[110]:


mean_df = current_df['Mean'].groupby(by=current_df[0]).mean()
current_df = current_df.drop('Mean', axis = 1)


# In[111]:

current_df = pd.merge(current_df, mean_df, left_on=0, right_on=0)



# In[113]:


current_df = current_df.drop_duplicates(0)

# In[114]:

def add_current_price(row):
    name = row[0]
    try:
        price = si.get_live_price(name)
    except:
        price = np.NaN
    return price
current_df["Price"] = current_df.apply(add_current_price, axis = 1)

# In[115]:
current_df = current_df.sort_values(0, ascending=False)

current_df.to_excel('/Users/michaelscoleri/Desktop/Coding/Personal/Python/Stock_scraper/finished_hits.xlsx', index = False)
