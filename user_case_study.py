# coding=utf-8
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
def enti_min_user(x,user_n):
    return x-user_n
def get_user_tfidf(uid,data):
    Analyze = 'Analyze/'
    tfidf_path = Analyze + data+'_u_e_tf_idf.txt'
    tfidf=pd.read_table(tfidf_path, sep=' ', header=0)
    # print(tfidf)
    user_tfidf=tfidf[tfidf['user']==uid]
    # print(user_tfidf)
    user_tfidf=user_tfidf.sort_values(by='tfidf')
    return user_tfidf

def get_user_attn(uid, data,model):
    Analyze = 'Analyze/'
    attn_path = Analyze+'saved_attn/' + model + '_' + data + '_attn.txt'
    attn = pd.read_table(attn_path, sep=' ', header=0)
    if(model!='HGB'):
        attn.columns = ['enti', 'user', 'attn']
        attn = attn[attn['user'] == uid]  # attn['enti']!=uid
        # list_sum=list(attn['attn'])
        # print(np.sum(list_sum)) # 加和为1才是正确的
        attn = attn[attn['enti'] != uid]
        attn['enti'] = attn['enti'].apply(enti_min_user, args=(user_num[data],))
        # attn = attn.groupby('enti').sum()
        # attn = attn.reset_index()
    else:
        attn.columns = ['head', 'tail', 'attn']
        attn1 = attn[attn['tail'] == uid]  # attn['enti']!=uid
        neigh1=attn1['head']
        attn2 = attn[attn['tail'].isin(neigh1)]  # attn['enti']!=uid
        attn1.columns = ['neigh1', 'user', 'attn1']
        attn2.columns = ['neigh2', 'neigh1', 'attn2']
        attn3 = pd.merge(attn1,attn2)
        attn3["attn"] = attn3["attn1"].mul(attn3["attn2"])
        attn4=pd.DataFrame({'user':attn3['user'],'enti':attn3['neigh2'],'attn':attn3['attn']})
        attn4 = attn4.groupby('enti').sum()
        attn4 = attn4.reset_index()
        attn4['enti'] = attn4['enti'].apply(enti_min_user, args=(user_num[data],))
        # print(attn4)  # 加和为1才是正确的
        # list_sum = list(attn4['attn'])
        # print(np.sum(list_sum))  # 加和为1才是正确的
        Analyze = 'Analyze/'
        tfidf_path = Analyze  + data + '_u_e_tf_idf.txt'
        tfidf = pd.read_table(tfidf_path, sep=' ', header=0)
        needpart=tfidf[tfidf['user']==uid]
        entilist=needpart['enti']
        attn=attn4[attn4['enti'].isin(entilist)]
    return attn
def compare_one_user(uid, data,model):
    tfidf = get_user_tfidf(uid,data)
    attn = get_user_attn(uid,data,model)
    attn = attn[attn['enti'].isin(tfidf['enti'])]
    result = pd.merge(tfidf, attn, on='enti', how='outer')
    result = result.sort_values(by='tfidf')
    result = result.fillna(0)
    return result
def one_user_relation(uid, data,model):
    onetfidf=compare_one_user(uid,data,model)

    onetfidf=onetfidf.fillna(0)
    tfidf=list(onetfidf['tfidf'])
    attn = list(onetfidf['attn'])

    pccs = 0.5*(np.corrcoef(tfidf, attn)[0, 1]+np.corrcoef(tfidf, attn)[1, 0])
    if(np.isnan(pccs)):
        print('--------' * 10)
        print(uid)
        print(tfidf)
        print(attn)
        print(pccs)
    else:
        print(np.sum(attn))
    return pccs
from multiprocessing.dummy import Pool
def get_1k_rela(data,model,check_un=100):
    user_list=range(check_un)
    # user_list=range(390,390+100)
    print(user_list)
    user_start_id=[]
    user_id=[]
    user_rela=[]
    def run(uid):
        user_start_id.append(uid)
        out_str='\r' + str(len(user_start_id)) + ' started ' + str(len(user_id)) + ' finished '
        print(out_str,end="")
        result = one_user_relation(uid,data,model)
        user_id.append(uid)
        user_rela.append(result)
        out_str = '\r' + str(len(user_start_id)) + ' started ' + str(len(user_id)) + ' finished '
        print(out_str,end="")
    def use_pool(id_list):
        print ('concurrent:')  # 创建多个进程，并行执行
        pool = Pool(10)  # 创建拥有10个进程数量的进程池
        # id_list:要处理的数据列表，run：处理 id_list 列表中数据的函数
        pool.map(run, id_list)
        pool.close()  # 关闭进程池，不再接受新的进程
        pool.join()  # 主进程阻塞等待子进程的退出
        print ("done")
    use_pool(user_list)
    return np.mean(user_rela)
if __name__ == '__main__':
#     print('start')
#     model_name = 'x0823'
#     data_l=[
#         'amazon-book_homo',
# #         'yelp2018_homo',
# #         'last-fm_homo',
# #         'movie-lens_homo',
#
#     ]
#     save_txt=''
#     for aim_data in data_l:
#         print('-'*10)
#         save_txt+='-'*10
#         save_txt+='\n'
#         print(aim_data)
#         save_txt+=aim_data
#         save_txt+='\n'
#         all_user_num = user_num[aim_data]
#         check_un = all_user_num
#         pcc=get_1k_rela(aim_data, model_name,check_un)
#         print('pccs is '+str(pcc))
#         save_txt+='pccs is '+str(pcc)
#         save_txt+='\n'
# #         print('-' * 10)
#     f=open('Analyze/data_pccs.txt','w')
#     f.write(save_txt)
#     print('finished')
    '''q'''
    path='Analyze/saved_attn/x0823_amazon-book_homo_attn.txt'
    attn = pd.read_table(path, sep=' ', header=0)
    # aim_0=70662
    # for i in range(18):
    #     aim=aim_0+i
    #     print('-'*20)
    #     print(aim)
    #
    #     print(aim_attn)
    path = 'Analyze/amazon-book_homo_u_e_tf_idf.txt'
    t = pd.read_table(path, sep=' ', header=0)
    aim_0 = 12395
    for i in range(1):
        aim = aim_0 + i
        print('-' * 20)
        print(aim)
        tfidf = (t[t['user'] == aim])

        aim_attn = (attn[attn['tail'] == aim])
        aim_attn.columns = ['enti1', 'user', 'attetion']
        print(aim_attn)
        aim_attn=aim_attn[aim_attn['enti1']!=aim]
        aim_attn['enti']=aim_attn['enti1'].apply(enti_min_user, args=(70679,))
        # print(aim_attn)
        # print(tfidf)
        aim_pd=pd.merge(tfidf, aim_attn, on='enti', how='outer')
        aim_pd=aim_pd.fillna(0)
        print(aim_pd.sort_values(by='tfidf'))
        t1=list(aim_pd['tfidf'])
        a1 = list(aim_pd['attetion'])
        pccs = 0.5 * (np.corrcoef(t1, a1)[0, 1] + np.corrcoef(t1, a1)[1, 0])
        print(pccs)
#path='Analyze/saved_attn/x0823_amazon-book_homo_attn.txt'
