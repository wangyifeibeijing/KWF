import pandas as pd
import numpy as np
# 保存 train/test 为 pd 格式
def save_uig(aim_data_name,mode='train',save=False):#mode='test'

    user_l=[]
    item_l=[]
    f=open('Data/'+aim_data_name+'/'+mode+'.txt')
    for lines in f.readlines():
        line=lines.split(' ')
        user=int(line[0])
        print('\r' +'moving to '+ str(user),end='')
        aim=line[1:]
        for item in aim:
            temp_i=item.strip()
            if(temp_i!=''):
                poi=int(temp_i)
                user_l.append(user)
                item_l.append(poi)
    df=pd.DataFrame(
        {0: user_l,1: item_l})
    df.columns = ['user', 'poi']
    if(save):
        path = 'Analyze/' + aim_data_name + '_u_i_g' + mode + '.txt'
        df.to_csv(path, index=False, sep=" ", header=None)
    print()
    return df
# 保存 train/test u-e 为 pd 格式
def save_ueg(aim_data_name,mode='train'):#mode='test'
    path = 'Data/' + aim_data_name + '/'
    kg_path = path + 'kg_final.txt'
    kg = pd.read_table(kg_path, sep=' ', header=None)
    kg.columns = ['poi', 're', 'enti']
    u_i= save_uig(aim_data_name,mode)# call function save_uig()
    u_e = pd.merge(u_i, kg)
    return u_e
def one_user(uid,u_e_use,mode='train'):
    if(mode=='train'):
        mark='tr'
    else:
        mark = 'te'
    u_e_temp=u_e_use[u_e_use['user']==uid]
    all_enti=len(u_e_temp)
    enti_list=(u_e_temp.groupby(['enti']).count()).reset_index()
    enti_list.columns = ['enti', mark+'_num']
    # print('\n'+'--enti_list--'*10)
    # print(enti_list)
    user_list=[uid]*len(enti_list)
    all_list=[1/(enti_list[mark+'_num'].sum())]*len(enti_list)
    enti_list['user']=user_list
    enti_list[mark+'_all_enti'] = all_list

    enti_list[mark+"_rate"] = enti_list[mark+'_num'].mul(enti_list[mark+'_all_enti'])
    # print('\n' + '--enti_list--' * 10)
    # print(enti_list)
    return enti_list
# 准备分布表：
from multiprocessing.dummy import Pool
def distri_u_e(aim_data_name,mode='test',save=True):
    print('deal with '+aim_data_name+' '+mode)
    if (mode == 'train'):
        mark = 'tr'
    else:
        mark = 'te'
    u_e=save_ueg(aim_data_name,mode)# call function save_ueg()
    u_e_use=pd.DataFrame({'user':u_e['user'],'enti':u_e['enti']})
    # uid_list = range(10)#
    uid_list = u_e_use['user'].unique()
    pd_land=[]
    user_start_list = []
    user_end_list = []

    def run(uid):
        user_start_list.append(uid)
        monitor_str = '\r' + str(len(user_start_list)) + ' started, ' + str(len(user_end_list)) + ' finished. '
        print(monitor_str, end='')
        u_pd=one_user(uid, u_e_use,mode)
        pd_land.append(u_pd)
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

    use_pool(uid_list)
    aim = pd.concat(pd_land)
    aim = aim.sort_values(by=['user','enti'])
    order = ['user', 'enti', mark+"_num", mark+'_all_enti', mark+"_rate"]
    aim = aim[order]
    if(save):
        path = 'Analyze/' + aim_data_name + '_user_enti_rate_' + mode + '.txt'
        aim.to_csv(path, index=False, sep=" ")
    print('\n' + '--all_list--' * 10)
    # print(aim)
    return aim
# 准备 apply log2
def app_log(input):
    if(input==0.0):
        input =1e-100
    return np.log2(input)
def clean(input):
    # if(abs(input)<=1e-50):
    #     return input
    # else:
    return input
# 合并表并在无 entity 处填充 0
def conc_tr_te(aim_data_name,save=True):
    u_e_r_tr = distri_u_e(aim_data_name, mode='train', save=True)
    u_e_r_te = distri_u_e(aim_data_name, mode='test', save=True)
    final = u_e_r_tr.merge(u_e_r_te,how='outer', on=['user','enti'])
    final=final.fillna(0.0)
    final["tr_log2"] = final["tr_rate"].apply(app_log, args=())
    final["te_log2"] = final["te_rate"].apply(app_log, args=())
    final["KL"] = final["te_rate"].mul(final["te_log2"]-final["tr_log2"])
    # final["KL"] =final["KL"].apply(clean, args=())
    if (save):
        path = 'Analyze/' + aim_data_name + '_user_enti_KL.txt'
        final.to_csv(path, index=False, sep=" ")
    # print(final)
    return final
def KL_cal(aim_data_name,save=True,read_it=True):
    if(read_it):
        try:
            path = 'Analyze/' + aim_data_name + '_user_enti_KL.txt'
            final=pd.read_table(path, sep=' ', header=0)
        except:
            print('generate new table for '+aim_data_name)
            final = conc_tr_te(aim_data_name,save=True)
    else:
        print('construct new table for ' + aim_data_name)
        final = conc_tr_te(aim_data_name, save=True)

    result=(final.groupby(['user'])["KL"].sum()).reset_index()
    if (save):
        path = 'Analyze/' + aim_data_name + '_user_KL.txt'
        result.to_csv(path, index=False, sep=" ")
    print()
    # print(result)
def av(aim_data_name):
    path = 'Analyze/' + aim_data_name + '_user_KL.txt'
    final = pd.read_table(path, sep=' ', header=0)
    return final['KL'].sum()/len(final)
if __name__ == '__main__':
    print(app_log(4))
    aim_data_name='last-fm'
    data_list=[
        'amazon-book',
        'last-fm',
        'movie-lens',
        'yelp2018'
    ]

    for data in data_list:
        conc_tr_te(data)
        KL_cal(data, read_it=True)
        print(data)
        print(av(data))