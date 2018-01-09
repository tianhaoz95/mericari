
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import collections
import matplotlib.pyplot as plt
import keras.layers as KL
import keras.optimizers as KO
import keras.models as KM
import keras.backend as K

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


train_raw = pd.read_csv('../input/train.tsv', sep='\t')
test_raw = pd.read_csv('../input/test.tsv', sep='\t')


# In[ ]:


print(test_raw.keys())
print(test_raw['name'][0])


# In[ ]:


name_list = []
for n in train_raw['name']:
    name_list.extend(n.split(' '))
for n in test_raw['name']:
    name_list.extend(n.split(' '))


# In[ ]:


name_collection = collections.Counter(name_list)
print('total words: ', len(name_collection.values()))
name_dict = {}
limit = 5
for k in name_collection:
    if name_collection[k] > limit:
        name_dict[k] = np.zeros(202)
print(len(name_dict))


# In[ ]:


for i in range(len(train_raw)):
    price_idx = int(train_raw['price'][i] // 10)
    nalist = train_raw['name'][i].split(' ')
    for na in nalist:
        if na in name_dict:
            name_dict[na][price_idx] = name_dict[na][price_idx] + 1
print(list(name_dict.values())[0])


# In[ ]:


for k in name_dict:
    if sum(name_dict[k]) != 0:
        name_dict[k] = name_dict[k] / sum(name_dict[k])
print(list(name_dict.values())[0])


# In[ ]:


brand_list = []
brand_list.extend(train_raw['brand_name'])
brand_list.extend(test_raw['brand_name'])
brand_list = [i for i in brand_list if str(i) != 'nan']
brand_collection = collections.Counter(brand_list)
max_brand_idx = list(brand_collection.values()).index(max(brand_collection.values()))
max_brand_name = list(brand_collection.keys())[max_brand_idx]
brand_dict = {'unknown': np.zeros(202)}
for i in range(len(train_raw)):
    price_idx = int(train_raw['price'][i] // 10)
    if str(train_raw['brand_name'][i]) == 'nan':
        brand_dict['unknown'][price_idx] = brand_dict['unknown'][price_idx] + 1
    else:
        brand_name = train_raw['brand_name'][i]
        if brand_name not in brand_dict:
            brand_dict[brand_name] = np.zeros(202)
        brand_dict[brand_name][price_idx] = brand_dict[brand_name][price_idx] + 1
print('brand dict size: ', len(brand_dict))
for k in brand_dict:
    brand_dict[k] = brand_dict[k] / sum(brand_dict[k])
print(list(brand_dict.values())[0])


# In[ ]:


total_item_condition = []
total_item_condition.extend(train_raw['item_condition_id'])
total_item_condition.extend(test_raw['item_condition_id'])
print('total item condition id count: ', len(total_item_condition))
print(list(set(total_item_condition)))
condition_list = list(set(total_item_condition))
condition_dict = {}
for i in range(len(condition_list)):
    condition_dict[condition_list[i]] = np.zeros(5)
    condition_dict[condition_list[i]][i] = 1
print('condition dict: ', condition_dict)


# In[ ]:


price_list = train_raw['price']
print('min price: ', min(price_list))
print('max price: ', max(price_list))
print('intervals: ', 2020//10)


# In[ ]:


price_seg = []
for p in train_raw['price']:
    row = np.zeros(202)
    row[int(p//10)] = 1
    price_seg.append(row)
price_seg = np.array(price_seg)
print('price seg shape: ', price_seg.shape)


# In[ ]:


min_category_cnt = 10
max_category_cnt = 0
max_len = 0
category_list = []
category_list.extend(train_raw['category_name'])
category_list.extend(test_raw['category_name'])
category_list = [c for c in category_list if str(c) != 'nan']
print(category_list[100].lower())
label_list = []
char_list = []
for c in category_list:
    clist = c.split('/')
    char_list.extend(list(c))
    label_list.extend(clist)
    max_len = max(max_len, len(c))
    min_category_cnt = min(min_category_cnt, len(clist))
    max_category_cnt = max(max_category_cnt, len(clist))
print('min: ', min_category_cnt,  'max: ', max_category_cnt)
print('total labels: ', len(label_list))
print('max length: ', max_len)


# In[ ]:


category_dict = {'unknown': np.zeros(202)}
for i in range(len(train_raw)):
    price_idx = int(train_raw['price'][i] // 10)
    if str(train_raw['category_name'][i]) == 'nan':
        category_dict['unknown'][price_idx] = category_dict['unknown'][price_idx] + 1
    else:
        clist = train_raw['category_name'][i].split('/')
        cuname = ''
        for cn in clist:
            cuname = cuname + cn
            if cuname not in category_dict:
                category_dict[cuname] = np.zeros(202)
            category_dict[cuname][price_idx] = category_dict[cuname][price_idx] + 1
print('dict size: ', len(category_dict))


# In[ ]:


for k in category_dict:
    category_dict[k] = category_dict[k]/sum(category_dict[k])
print(list(category_dict.values())[0])


# In[ ]:


unknown_cnt = 0
for i in range(len(test_raw)):
    if str(test_raw['category_name'][i]) == 'nan':
        unknown_cnt = unknown_cnt + 1
    else:
        clist = test_raw['category_name'][i].split('/')
        cuname = ''
        found = False
        for cn in clist:
            cuname = cuname + cn
            if cuname in category_dict:
                found = True
        if not found:
            unknown_cnt = unknown_cnt + 1
print('unknown rate: ', unknown_cnt/len(test_raw))


# In[ ]:


y = np.array([[p] for p in train_raw['price']])
print(y.shape)


# In[ ]:


def getX(data, start, end):
    X = []
    for i in range(start, end):
        row = None
        brand_name = 'unknown'
        if str(data['brand_name'][i]) in brand_dict:
            brand_name = data['brand_name'][i]
        row = brand_dict[brand_name]
        category_features = category_dict['unknown']
        if str(data['category_name'][i]) != 'nan':
            clist = data['category_name'][i].split('/')
            cuname = ''
            for cn in clist:
                cuname = cuname + cn
                if cuname in category_dict:
                    category_features = category_dict[cuname]
        row = np.hstack([row, category_features])
        name_features = np.zeros(202)
        for na in train_raw['name'][i].split(' '):
            if na in name_dict:
                name_features = name_features + name_dict[na]
        if sum(name_features) != 0:
            name_features = name_features / sum(name_features)
        row = np.hstack([row, name_features])
        condition_features = condition_dict[data['item_condition_id'][i]]
        row = np.hstack([row, condition_features])
        shipping_features = np.array([data['shipping'][i]])
        row = np.hstack([row, shipping_features])
        X.append(row)
    X = np.array(X)
    print('X shape: ', X.shape)
    return X


# In[ ]:


model = KM.Sequential()
model.add(KL.Dense(1024, activation='relu', input_dim=612))
model.add(KL.Dense(512, activation='relu'))
model.add(KL.Dense(256, activation='relu'))
model.add(KL.Dropout(0.2))
model.add(KL.Dense(1, activation='relu'))
model.compile(optimizer='rmsprop', loss='mean_squared_error')


# In[ ]:


trunk = 20000
for i in range(int(len(train_raw)//trunk)+1):
    print(i*trunk/len(train_raw)*100)
    start = i*trunk
    end = min((i+1)*trunk, len(train_raw))
    X = getX(train_raw, start, end)
    H = model.fit(x=X, y=y[start:end], batch_size=32, verbose=1, epochs=5)


# In[ ]:


predictions = []
for i in range(int(len(test_raw)//trunk)+1):
    print(i*trunk/len(test_raw)*100)
    start = i*trunk
    end = min((i+1)*trunk, len(test_raw))
    X_test = getX(test_raw, start, end)
    pred = model.predict(x=X_test, verbose=1)
    for j in range(pred.shape[0]):
        predictions.append(pred[j,0])
output_dict = {'test_id':test_raw['test_id'],'price':predictions}
output = pd.DataFrame(output_dict)
print('output length: ', len(output))
print('should be: ', len(test_raw))
output.to_csv('submission.csv', index=False)
print(check_output(["ls"]).decode("utf8"))

