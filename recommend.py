import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
import math
import scipy.spatial.distance
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine
'''
tx_data    = pd.read_csv('D:\data set\Release\data\Data_original.csv')
tx_data.TRANSACTION_NO = tx_data.TRANSACTION_NO.str[10:]

tx_dataframe = pd.DataFrame(tx_data)
print(tx_data)
tx_data.to_csv('D:\data set\Release\data\Data_modify.csv',index = False)

'''

from IPython.display import display
tx_data     = pd.read_csv('D:/data set/Release/data/Data_modify.csv')
#brand_data  = pd.read_csv('D:/data set/Release/data/brand_table.csv', engine='python')
brand_data  = pd.read_csv('D:/data set/Release/data/brand_table_modify.csv', engine='python')
tx_frame    = pd.DataFrame(tx_data)
brand_frame = pd.DataFrame(brand_data)

vote_frame      = pd.DataFrame(brand_frame['POS번호'])
store_name_df   = pd.DataFrame(brand_frame['BRAND_NAME'])
vote_numpy      = np.array(vote_frame)
sum_vote_by_pos = np.zeros(len(vote_frame))

#거래내역에서 유니크ID 준비
txid_np = np.array(tx_frame['TRANSACTION_NO'],ndmin=2)
print(txid_np.shape)
print(txid_np[0])

txid_uniques = np.unique(txid_np)
'''
voter_id = txid_uniques.reshape((1,-1))
zero_2d = np.zeros((len(vote_frame)+1,txid_uniques.shape[0]))
vote_panel = np.concatenate((voter_id,zero_2d),axis=0)
'''
print(len(vote_frame))
vote_panel = np.zeros((len(vote_frame),txid_uniques.shape[0]))

print(txid_uniques.shape)



#print(uniques)
for row in tx_frame.itertuples(index=True):
    row_idx = np.where(vote_numpy == getattr(row, "STORE"))
    col_idx = np.where(txid_uniques == getattr(row, "TRANSACTION_NO"))
    vote_panel[row_idx[0], col_idx[0]] = np.array([1])
    sum_vote_by_pos[row_idx[0]]        += np.array([1])

vote_frame['vote'] = sum_vote_by_pos
vote_panel_df      = pd.DataFrame(vote_panel,columns=txid_uniques)

#vote_panel_df = vote_panel_df.drop(0,0)
#vote_panel_df = vote_panel_df.reset_index(drop=True)
#merged_df = pd.merge(vote_frame,vote_panel_df, how='left', left_index=True, right_index=True)

#cosine similarity

#print(sum_vote_by_pos.shape)
#print(vote_panel.shape)
#cosine_result = cosine_similarity(vote_panel_df,vote_panel_df)



#cosine_result = 1-pairwise_distances(vote_panel, metric="cosine")

cosine_result = cosine_similarity(vote_panel)
cosine_test =  np.argsort(cosine_result[84])[::-1]

for i in range(15):
    print(store_name_df.iloc[cosine_test[i]])




#print(cosine_result.shape)
#cosine_result_df = pd.DataFrame(cosine_result,columns=store_name_df)
#merged_df = pd.merge(vote_frame,cosine_result_df, how='left', left_index=True, right_index=True)
#merged_df.to_csv('D:\data set\Release\data\pos.csv',index = False,encoding='ms949')

#print(len(cosine_result))


'''
for base_idx, row in enumerate(vote_panel_df.itertuples(),0):
    if base_idx == len(vote_frame):
        break
    else:
        x = np.array(vote_panel[base_idx + 1])
        #x = x.reshape(1,-1)
        for reference_idx, re_row in enumerate(vote_panel_df.itertuples(),0):
            if reference_idx == len(vote_frame):
                break
            else:
                y = np.array(vote_panel[reference_idx+1])
                #y = y.reshape(1,-1)
                #print(base_idx, reference_idx)
                #cosine_result = cosine_similarity(x,y)
                dot = np.dot(x, y)
                norma = np.linalg.norm(x)
                normb = np.linalg.norm(y)
                cos = dot / (norma * normb)
                #print(cos)

'''
'''
data = []
data.insert(0,{'POS번호':0,'vote':0})
vote_frame = pd.concat([pd.DataFrame(data),vote_frame],ignore_index=True)
print_df=pd.DataFrame(txid_vote,columns=uniques)
print_df = pd.merge(vote_frame,print_df, how='left', left_index=True, right_index=True)
print(print_df)
print_df = print_df.drop(0,0)
print_df = print_df.reset_index(drop=True)
print(print_df)
#print_df.to_csv('D:\data set\Release\data\pos.csv',index = False,encoding='ms949')
'''


