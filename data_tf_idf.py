# coding=utf-8
import pandas as pd
import numpy as np
from multiprocessing.dummy import Pool
user_num_dic={
    'amazon-book':70679,
    'last-fm':23566,
    'movie-lens':37385,
    'yelp2018':45919,

}
aim_data_name='amazon-book'#'amazon-book'#'amazon-book_homo'#'yelp2018'#'movie-lens'#
aim_user_num=user_num_dic[aim_data_name]
# 保存 train 为 pd 格式
def save_uig():
    path='Analyze/'+aim_data_name+'_u_i_g.txt'
    user_l=[]
    item_l=[]
    f=open('Data/'+aim_data_name+'/train.txt')
    for lines in f.readlines():
        line=lines.split(' ')
        user=int(line[0])
        print('\r' +'moving to '+ str(user),end='')
        aim=line[1:]
        for item in aim:
            poi=int(item)
            user_l.append(user)
            item_l.append(poi)
    df=pd.DataFrame(
        {0: user_l,1: item_l})
    df.to_csv(path, index=False, sep=" ", header=None)
    return len(df)
# tf-idf 计算
def count_enti_for_all_u(enti_id):
    u_i=pd.read_table('Analyze/'+aim_data_name+'_u_i_g.txt', sep=' ', header=None)
    u_i.columns = ['user', 'poi']
    # data_name = 'amazon-book_new'
    path = 'Data/' + aim_data_name + '/'
    kg_path = path + 'kg_final.txt'
    kg = pd.read_table(kg_path, sep=' ', header=None)
    kg.columns = ['poi', 're', 'enti']
    u_e=pd.merge(u_i,kg)
    u_e_cout=len(u_e[u_e['enti']==enti_id]['user'].unique())
    return u_e_cout

def u_e():
    u_i = pd.read_table('Analyze/'+aim_data_name+'_u_i_g.txt', sep=' ', header=None)
    u_i.columns = ['user', 'poi']
    # data_name = 'amazon-book_new'
    path = 'Data/' + aim_data_name + '/'
    kg_path = path + 'kg_final.txt'
    kg = pd.read_table(kg_path, sep=' ', header=None)
    kg.columns = ['poi', 're', 'enti']
    u_e = pd.merge(u_i, kg)
    u_e_result=pd.DataFrame(
        {'user': u_e['user'], 'enti': u_e['enti']})
    return u_e_result

def user_all_num(u_e):
    u_e_cout = len(u_e['user'].unique())
    return u_e_cout

def enti_used(enti_id,u_e):
    u_e_cout = len(u_e[u_e['enti'] == enti_id]['user'].unique())
    return u_e_cout

def enti_idf_one(enti_id,u_e,u_num):
    idf = np.log(u_num / (enti_used(enti_id,u_e) + 1))
    return idf

def enti_idf(u_e,u_num):
    enti_l=u_e['enti'].unique()
    temp = pd.DataFrame(
        {'enti': enti_l})
    idf_l=temp['enti'].apply(enti_idf_one,args=(u_e,u_num))
    result = pd.DataFrame(
        {'enti': temp['enti'],'idf': idf_l})
    return result


def user_tf(u_e, uid, idf):
    u_e_aim=u_e[u_e['user']==uid]
    enti_num=len(u_e_aim)
    result=u_e_aim['enti'].value_counts()/enti_num
    result = pd.DataFrame({'enti':result.index,'tf':result.values})
    idf_use=idf[idf['enti'].isin(result['enti'])]
    tf_idf = pd.merge(result, idf_use)
    tf_idf["tfidf"] = tf_idf["tf"].mul(tf_idf["idf"])
    tf_idf=tf_idf.sort_values(by='tfidf')
    user_=[uid]*len(tf_idf)
    tf_idf["user"]=user_
    return tf_idf

def save_tfidf():
    u_e_r = u_e()
    id_list = range(aim_user_num)
    u_num = user_all_num(u_e_r)
    idf = enti_idf(u_e_r, u_num)
    pool_list = []
    user_start_list = []
    user_end_list = []
    def run(uid):
        user_start_list.append(uid)
        monitor_str = '\r' + str(len(user_start_list)) + ' started, ' + str(len(user_end_list)) + ' finished. '
        print(monitor_str, end='')
        result = user_tf(u_e_r, uid, idf)
        pool_list.append(result)
        user_end_list.append(uid)
        monitor_str = '\r' + str(len(user_start_list)) + ' started, ' + str(len(user_end_list)) + ' finished. '
        print(monitor_str, end='')

    def use_pool(id_list):
        print('concurrent in :'+str(len(id_list)))  # 创建多个进程，并行执行
        pool = Pool(10)  # 创建拥有10个进程数量的进程池
        # id_list:要处理的数据列表，run：处理 id_list 列表中数据的函数
        pool.map(run, id_list)
        pool.close()  # 关闭进程池，不再接受新的进程
        pool.join()  # 主进程阻塞等待子进程的退出
        print("done")

    use_pool(id_list)
    aim = pd.concat(pool_list)
    aim = aim.sort_values(by='user')
    order = ['user', 'enti', 'tfidf', 'tf', 'idf']
    aim = aim[order]
    # print(aim)
    aim_path = 'Analyze/'+aim_data_name+'_u_e_tf_idf.txt'
    aim.to_csv(aim_path, index=False, sep=" ")


#用于创建累加型tfidf列表的相关函数，为创造黑名单准备
def oneuser_tfidf_accu(uid,all_tf_idf):
    aimpart=all_tf_idf[all_tf_idf['user']==uid]
    # templist = list(aimpart['tfidf'])
    aimpart['tfidf+']=aimpart['tfidf']
    aimpart.loc[aimpart['tfidf+'] < 0, 'tfidf+'] = 0
    aimpart = aimpart.sort_values(by='tfidf',ascending=False)
    list_temp=list(aimpart['tfidf+'])
    total=np.sum(list_temp)
    list_accu=[]
    flag=0
    for i in list_temp:
        flag+=i
        list_accu.append(flag/total)
    aimpart['account']=list_accu
    return aimpart
def tf_idf_range_accu():
    aim_path = 'Analyze/'+aim_data_name+'_u_e_tf_idf.txt'
    all_tf_idf=pd.read_table(aim_path, sep=' ', header=0)
    user_list = range(aim_user_num)
    tfidf_list=[]
    user_start_list=[]
    user_end_list = []
    def deal_one(uid):
        user_start_list.append(uid)
        monitor_str='\r'+str(len(user_start_list))+' started, '+str(len(user_end_list))+' finished. '
        print(monitor_str,end='')
        result = oneuser_tfidf_accu(uid,all_tf_idf)
        tfidf_list.append(result)
        user_end_list.append(uid)
        monitor_str = '\r' + str(len(user_start_list)) + ' started, ' + str(len(user_end_list)) + ' finished. '
        print(monitor_str, end='')
    def use_pool_tfidf_accu(user_list):
        print('concurrent in :'+str(len(user_list)))  # 创建多个进程，并行执行
        pool = Pool(10)  # 创建拥有10个进程数量的进程池
        # id_list:要处理的数据列表，run：处理 id_list 列表中数据的函数
        pool.map(deal_one, user_list)
        pool.close()  # 关闭进程池，不再接受新的进程
        pool.join()  # 主进程阻塞等待子进程的退出
        print("done")

    use_pool_tfidf_accu(user_list)
    aim = pd.concat(tfidf_list)
    path='Analyze/'+aim_data_name+'_u_e_tf_idf_accu.txt'
    aim = aim.sort_values(by=['user','tfidf'], ascending=(True,False))
    aim.to_csv(path, index=False, sep=" ")
    return aim
#用于创建黑名单的函数
def add_mask(input,mask):
    return input+mask
def tfidf_based_black(rate,tfidf_accu):
    # tfidf_accu=tfidf_accu[tfidf_accu['account']>rate]
    tfidf_accu = tfidf_accu[tfidf_accu['tfidf'] < 0]
    list_edge=pd.DataFrame({'user':tfidf_accu['user'],'enti':tfidf_accu['enti']})
    u_num=aim_user_num

    list_edge["enti"] = list_edge["enti"].apply(add_mask,args=(u_num,))
    list_edge = list_edge.sort_values(by=['user', 'enti'], ascending=(True, True))
    path='Analyze/'+aim_data_name+'_neg_enti_'+str(int(rate*100))+'%.txt'
    list_edge.to_csv(path, index=False, sep="\t", header=None)
    droped_edges=len(list_edge)
    print('droped '+str(droped_edges)+' edges')
    return droped_edges
if __name__ == '__main__':
    import warnings

    warnings.filterwarnings('ignore')
    print(str(aim_data_name)+' start')
    ui_edges=save_uig()#用于生成u-i图，当前版本只包含train
    kgpath='Data/'+aim_data_name+'/kg_final.txt'
    ie_edges=pd.read_table(kgpath, sep=' ', header=0)
    ie_num=len(ie_edges)
    print('\nu-i graph generated, process tf-idf')
    save_tfidf()#用于保存tfidf，当前版本只包含train
    print('\ntf-idf generated, process accumulate')
    all_tf_idf=tf_idf_range_accu()
    all_edge=len(all_tf_idf)+ie_num
    # print(all_tf_idf)
    print('edge number: '+str(all_edge))

    path = 'Analyze/' + aim_data_name + '_u_e_tf_idf_accu.txt'
    use_tf_idf = pd.read_table(path, sep=' ', header=0)

    print('\naccumulate generated, process black list')
    rate=1.0
    droped_edges = tfidf_based_black(rate, use_tf_idf)
    print('left edge number: ' + str(all_edge-droped_edges))
    # print('\n85% drop generated, process 90% drop')
    # rate = 0.90
    # droped_edges = tfidf_based_black(rate, use_tf_idf)
    # print('left edge number: ' + str(all_edge - droped_edges))
    # print('\n90% drop generated, process 99% drop')
    # rate = 0.99
    # droped_edges = tfidf_based_black(rate, use_tf_idf)
    # print('left edge number: ' + str(all_edge - droped_edges))
    print(str(aim_data_name)+' finished')
    # import math
    # print((2**(-1/math.log(2))))