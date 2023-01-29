import pandas as pd
import numpy as np
user_num={
    'amazon-book_new':44004,
    'amazon-book': 70679,
    'last-fm': 23566,
    'movie-lens': 37385,
    'yelp2018': 45919,
    'amazon-book_homo': 70679,
    'last-fm_homo': 23566,
    'movie-lens_homo': 37385,
    'yelp2018_homo': 45919,
}

def enti_max_user(x,user_n):
    return x+user_n

def save_tfidf_dic(data_name):
    path = 'Analyze/'+data_name+'_u_e_tf_idf.txt'
    t = pd.read_table(path, sep=' ', header=0)
    u_n = user_num[data_name]
    t['enti'] = t['enti'].apply(enti_max_user, args=(u_n,))

    tf_idf_dic={}
    all_len=len(t)
    flag=0
    for indexs in t.index:
        flag+=1
        temp=t.loc[indexs]
        print('\r'+str(flag)+' of '+str(all_len)+' with '+str(temp['enti'])+'+'+str(temp['user'])+'+'+str(temp['idf']),end='')
        tf_idf_dic[(int(temp['enti']) ,int(temp['user']) )]=temp['idf']
    np.save(data_name+'_idf_dic.npy',tf_idf_dic)
def load_tfidf_dic(data_name):
    path=data_name+'_idf_dic.npy'
    aim=np.load(path,allow_pickle=True).item()
    return aim
if __name__ == '__main__':
    data_name = 'amazon-book_homo'
    save_tfidf_dic(data_name)
    load_dic=load_tfidf_dic(data_name)


    # print(load_dic[((46113+user_num[data_name]),(11300))])