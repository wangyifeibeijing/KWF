import numpy as np
import pandas as pd
user_num_dic={
    'amazon-book':70679,
    'last-fm':23566,
    'movie-lens':37385,
    'yelp2018':45919,

}
def compare_ndcg(model,data):
    path_head = 'Analyze/Compare_folder/'
    mf_model='mf'
    mf_file = path_head + mf_model + '_'+data+'_user_ndcg100.txt'
    md_file = path_head + model + '_' + data + '_user_ndcg100.txt'
    mf_pd = pd.read_table(mf_file, sep=' ', header=0)
    mf_pd.columns = ['user', 'mf_ndcg']
    md_pd = pd.read_table(md_file, sep=' ', header=0)
    md_pd.columns = ['user', 'x_ndcg']
    compare = pd.merge(mf_pd, md_pd)
    compare['x-mf']=compare['x_ndcg'] - compare['mf_ndcg']
    compare = compare.sort_values(by='x-mf')
    compare_path = path_head + model + '_' + data + '_compare.txt'
    compare.to_csv(compare_path, index=False, sep=" ")
    return compare
def add_mask(input):
    return input+45919
def ret_attn(model,data,uid):
    path_head = 'Analyze/Compare_folder/'
    md_file = path_head + model + '_' + data + '_attn.txt'
    md_pd = pd.read_table(md_file, sep=' ', header=0)
    md_aim = md_pd[md_pd['tail']==uid]
    md_aim = md_aim.sort_values(by='attetion')
    tf_file = path_head + data + '_u_e_tf_idf.txt'
    tf_pd = pd.read_table(tf_file, sep=' ', header=0)
    tf_pd=tf_pd[tf_pd['user']==uid]
    tf_pd["enti"]=tf_pd["enti"].apply(add_mask, args=())
    md_aim.columns=['enti','user','attn']
    print(md_aim)
    # print(tf_pd)
    re= pd.merge(md_aim, tf_pd)
    re = re.sort_values(by='tfidf')
    return re

def con_graph(data):
    tr = 'Data/' + data + '/train.txt'

    f_tr = open(tr, 'r')
    tr_u=[]
    tr_i=[]
    for line in f_tr.readlines():
        line_l = line.split(' ')
        user= int(line_l[0])
        line_i=line_l[1:]
        for i in line_i:
            i=i.strip()
            if(i!=''):
                item=int(i)
                tr_u.append(user)
                tr_i.append(item)
    temp = pd.DataFrame(
        {'user': tr_u,'poi': tr_i})
    path_head = 'Analyze/Compare_folder/'
    compare_path = path_head + data + '_tr_u_i.txt'
    temp.to_csv(compare_path, index=False, sep=" ")
    f_tr.close()
    te = 'Data/' + data + '/test.txt'
    f_tr = open(te, 'r')
    tr_u = []
    tr_i = []
    for line in f_tr.readlines():
        line_l = line.split(' ')
        user = int(line_l[0])
        line_i = line_l[1:]
        for i in line_i:
            i = i.strip()
            if (i != ''):
                item = int(i)
                tr_u.append(user)
                tr_i.append(item)
    temp = pd.DataFrame(
        {'user': tr_u, 'poi': tr_i})
    path_head = 'Analyze/Compare_folder/'
    compare_path = path_head + data + '_te_u_i.txt'
    temp.to_csv(compare_path, index=False, sep=" ")
    f_tr.close()
def check_enti(model,data,uid):
    pass

if __name__ == '__main__':
    model='x0823'
    data='yelp2018_homo'
    # print(compare_ndcg(model, data))
    uid=490
    aim_attn=ret_attn(model, data, uid)
    print(aim_attn)
    print(len(aim_attn))


    pass