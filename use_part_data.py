import numpy as np
import random
import pandas as pd
import os
from shutil import copyfile

data_name = 'amazon-book'
import torch
from sklearn.cluster import KMeans, MiniBatchKMeans
import matplotlib.pyplot as plt
import matplotlib
import time

'''
获得图计数部分，调用时
        new_d_name=【数据名称】

        num_new = check_num(new_d_name)
就会返回所有边的数目
'''


def get_u_i_count(data_name):
    ftr = open('new_1k_datas/Data/' + data_name + '/train.txt', 'r')

    user_list = []
    item_list = []
    for line in ftr.readlines():
        line_l = line.split(' ')
        user = int(line_l[0].strip())
        for item in line_l[1:]:
            poi = int(item.strip())
            user_list.append(user)
            item_list.append(poi)
    '''
    fte=open('new_1k_datas/Data/'+data_name+'/test.txt','r')
    for line in fte.readlines():
        line_l = line.split(' ')
        user = int(line_l[0].strip())
        for item in line_l[1:]:
            poi = int(item.strip())
            user_list.append(user)
            item_list.append(poi)
    '''
    df_ui = pd.DataFrame({'user': user_list, 'poi': item_list})
    df_ui = df_ui.sort_values(by='user', ascending=True)
    list_mid = ['+'] * len(df_ui["user"])
    df_ui['id'] = df_ui["user"].map(str) + list_mid + df_ui["poi"].map(str)  # +list_mid+ df_ue["relation"].map(str)
    # print(len((df_ui['id'])))
    # print(len(pd.unique(df_ui['id'])))
    user_num = (np.max(pd.unique(df_ui['user']))) + 1
    print('user_num' + '-' * 20)
    print(user_num)

    return df_ui


def get_kg(data_name):
    path = 'new_1k_datas/Data/' + data_name + '/kg_final.txt'
    df_kg = pd.read_table(path, sep=' ', header=None)
    poi_list = df_kg[0]
    rela_list = df_kg[1]
    enti_list = df_kg[2]

    df_ie = pd.DataFrame({'poi': poi_list, 'relation': rela_list, 'entity': enti_list})
    df_ie = df_ie.sort_values(by='poi', ascending=True)
    poi_min = (np.min(pd.unique(df_ie['poi'])))
    # print('poi_min'+'-'*20)
    # print(poi_min)

    return df_ie


def get_all(data_name):
    df_ie = get_kg(data_name)
    df_ui = get_u_i_count(data_name)
    # print(df_ie)
    # print(df_ui)
    result = pd.merge(df_ie, df_ui, on=['poi'])
    result = result.sort_values(by='user', ascending=True)
    # print(result)
    return result


def get_ue(data_name):
    uie = get_all(data_name)
    user_list = uie['user']
    rela_list = uie['relation']
    enti_list = uie['entity']
    df_ue = pd.DataFrame({'user': user_list, 'relation': rela_list, 'entity': enti_list})
    list_mid = ['+'] * len(df_ue["user"])
    df_ue["entity"] = df_ue["entity"].map(add_ui)
    df_ue['id'] = df_ue["user"].map(str) + list_mid + df_ue["entity"].map(str)  # +list_mid+ df_ue["relation"].map(str)
    # print(df_ue)

    df_result = pd.unique(df_ue['id'])
    df_ue_use = pd.DataFrame({'id': df_ue["id"], 'user': df_ue["user"], 'entity': df_ue["entity"]})

    i_e = get_kg(data_name)
    list_mid = ['+'] * len(i_e["poi"])
    i_e['poi'] = i_e["poi"].map(add_u)
    i_e['entity'] = i_e["entity"].map(add_ui)
    i_e['id'] = i_e["poi"].map(str) + list_mid + i_e["entity"].map(str)
    df_result_ie = pd.unique(i_e['id'])

    df_ie_use = pd.DataFrame({'id': i_e["id"], 'poi': i_e["poi"], 'entity': i_e["entity"]})
    g_size = len(df_result) * 2 + len(df_result_ie) * 2
    # print(g_size)

    return df_ue_use, df_ie_use


def add_u(a):
    if data_name == 'amazon-book':
        user_total = 70680
    elif data_name == 'last-fm':
        user_total = 23566
    elif data_name == 'movie-lens':
        user_total = 37384
    elif data_name == 'yelp2018':
        user_total = 45919
    else:
        user_total = 0
    return a + user_total  # 37385


def add_ui(a):
    if data_name == 'amazon-book':
        user_total = 70680
    elif data_name == 'last-fm':
        user_total = 23566
    elif data_name == 'movie-lens':
        user_total = 37384
    elif data_name == 'yelp2018':
        user_total = 45919
    else:
        user_total = 0
    return a + user_total  # 37385


def give_our(df_ue, i_e):
    df_ue_1 = pd.DataFrame({'head': df_ue["user"], 'tail': df_ue["entity"]})
    df_ue_2 = pd.DataFrame({'head': df_ue["entity"], 'tail': df_ue["user"]})
    df_ie_1 = pd.DataFrame({'head': i_e["poi"], 'tail': i_e["entity"]})
    df_ie_2 = pd.DataFrame({'head': i_e["entity"], 'tail': i_e["poi"]})
    '''
    res1= pd.concat([df_ue_1,df_ue_2], ignore_index=True)#res2= pd.concat([df_ue_1,df_ue_2,df_ie_1,df_ie_2], ignore_index=True)
    res1.rename(columns={0:'head',1:'tail'})
    list_mid = ['+']*len(res1["head"])
    res1['id'] = res1["head"].map(str) +list_mid+ res1["tail"].map(str)
    print('新生成边数：')
    print(len(pd.unique(res1['id'])))
    '''
    res2 = pd.concat([df_ue_1, df_ue_2, df_ie_1, df_ie_2],
                     ignore_index=True)  # res2= pd.concat([df_ue_1,df_ue_2,df_ie_1,df_ie_2], ignore_index=True)
    res2.rename(columns={0: 'head', 1: 'tail'})
    list_mid = ['+'] * len(res2["head"])
    res2['id'] = res2["head"].map(str) + list_mid + res2["tail"].map(str)

    print('总边数：')
    print(len(pd.unique(res2['id'])))
    return res2


def check_num(data_name):
    df_ue, i_e = get_ue(data_name)
    res = give_our(df_ue, i_e)
    return len(pd.unique(res['id']))


'''
抽取子图部分，调用时
        save_all(data_name,use_num)
就会存名字为旧名字+_new的新数据集
'''


def get_pretrain(data_name='amazon-book'):
    cat_data = np.load('pretrain/' + data_name + '/mf.npz')
    user_embed = cat_data['user_embed']
    item_embed = cat_data['item_embed']
    return user_embed, item_embed


def str2intli(strlist):
    aim_list = []
    for aa in strlist:
        try:
            aim_list.append(int(aa.strip()))
        except:
            pass  # print(aa)
    return aim_list


def get_u_i(data_name, use_num):
    if data_name == 'amazon-book':
        user_total = 70680
    elif data_name == 'last-fm':
        user_total = 23566
    elif data_name == 'movie-lens':
        user_total = 37384
    elif data_name == 'yelp2018':
        user_total = 45919
    else:
        user_total = 0
    aim = range(user_total)
    rs = random.sample(aim, use_num)
    ftr = open('Data/' + data_name + '/train.txt', 'r')
    fte = open('Data/' + data_name + '/test.txt', 'r')
    trset = []
    teset = []
    flag = 0
    user_list = []
    item_list = []
    for line in ftr.readlines():
        if (flag in rs):
            line_l = line.split(' ')
            user = int(line_l[0])
            pois = str2intli(line_l[1:])
            trset.append([user, pois])
            user_list.append(user)
            item_list.extend(pois)
        flag += 1
    flag = 0
    for line in fte.readlines():
        if (flag in rs):
            line_l = line.split(' ')
            user = int(line_l[0])
            pois = str2intli(line_l[1:])
            teset.append([user, pois])
            user_list.append(user)
            item_list.extend(pois)
        flag += 1

    user_list = list(set(user_list))
    item_list = list(set(item_list))

    user_dic = {}
    item_dic = {}
    for i in range(len(user_list)):
        user_dic[user_list[i]] = i
    for i in range(len(item_list)):
        item_dic[item_list[i]] = i
    return user_list, item_list, trset, teset, user_dic, item_dic


def get_dic_li(dic_aim, list_aim):
    list_n = []
    for i in list_aim:
        list_n.append(dic_aim[i])
    return list_n


def get_enti(item_list, item_dic, data_name='amazon-book'):
    path = 'Data/' + data_name + '/kg_final.txt'
    df = pd.read_table(path, sep=' ', header=None)
    df = df[df[0].isin(item_list)]
    head = get_dic_li(item_dic, df[0])
    re = df[1]
    entiy = list(set(df[2]))
    enti_list = range(len(entiy))
    enti_list = [i + (len(item_list) + 1) for i in enti_list]

    enti_dic = {}
    for i in range(len(entiy)):
        enti_dic[entiy[i]] = enti_list[i]
    tail = get_dic_li(enti_dic, df[2])
    df_item = pd.DataFrame({'0': head, '1': re, '2': tail})
    df_item.to_csv('new_1k_datas/Data/' + data_name + '_new/kg_final.txt', index=False, sep=" ", header=None)  # 存文件
    return len(df_item)


def save_pretrain(user_list, item_list, user_dic, item_dic, data_name='amazon-book'):
    u_l = get_dic_li(user_dic, user_list)
    i_l = get_dic_li(item_dic, item_list)
    user_embed, item_embed = get_pretrain(data_name)
    u_cho = np.matrix(user_embed[user_list])
    i_cho = np.matrix(item_embed[item_list])
    [n, m] = u_cho.shape
    [k, l] = i_cho.shape
    u_e = np.zeros((n, m))
    i_e = np.zeros((k, l))
    u_e[u_l] = user_embed[user_list]
    i_e[i_l] = item_embed[item_list]
    np.savez('new_1k_datas/pretrain/' + data_name + '_new/mf.npz', user_embed=u_e, item_embed=i_e)


def save_tr_te(trset, teset, user_dic, item_dic, data_name='amazon-book'):
    '''
    tr_new=[]
    te_new=[]
    for aim_u_i in trset:
        user=user_dic[aim_u_i[0]]
        pois=get_dic_li(item_dic,aim_u_i[1])
        tr_new.append([user,pois])
    for aim_u_i in teset:
        user=user_dic[aim_u_i[0]]
        pois=get_dic_li(item_dic,aim_u_i[1])
        te_new.append([user,pois])
    '''
    text_tr = ''
    text_te = ''
    for aim_u_i in trset:
        user = aim_u_i[0]
        pois = aim_u_i[1]
        tr_temp = str(user_dic[user])
        for poi in pois:
            tr_temp += (' ' + str(item_dic[poi]))
        text_tr += (tr_temp + '\n')
        # print(text_tr)
    for aim_u_i in teset:
        user = aim_u_i[0]
        pois = aim_u_i[1]
        te_temp = str(user_dic[user])
        for poi in pois:
            te_temp += (' ' + str(item_dic[poi]))
        text_te += (te_temp + '\n')
        # print(text_te)
    ftr = open('new_1k_datas/Data/' + data_name + '_new/train.txt', 'w')
    fte = open('new_1k_datas/Data/' + data_name + '_new/test.txt', 'w')
    ftr.write(text_tr)
    fte.write(text_te)
    ftr.close()
    fte.close()


def save_all(data_name='amazon-book', use_num=100000):
    user_list, item_list, trset, teset, user_dic, item_dic = get_u_i(data_name, use_num)
    np.save('new_1k_datas/Data/' + data_name + '_new/user_dic.npy', user_dic)
    np.save('new_1k_datas/Data/' + data_name + '_new/item_dic.npy', item_dic)
    # print(len(user_list))
    # print(len(item_list))
    # print(len(trset))
    # print(len(teset))
    # print()
    get_enti(item_list, item_dic, data_name)
    save_pretrain(user_list, item_list, user_dic, item_dic, data_name)
    save_tr_te(trset, teset, user_dic, item_dic, data_name)


'''
二叉搜索最适合aim边数的user数目

data_namelist =[
        'amazon-book',
        'last-fm',
        'movie-lens',
        'yelp2018',
    ]
    aim=3500000
    for data_name in data_namelist:
        tr_best(data_name,aim)
'''


def tr_best(data_name, aim):
    if data_name == 'amazon-book':
        user_total = 70680
    elif data_name == 'last-fm':
        user_total = 23566
    elif data_name == 'movie-lens':
        user_total = 37384
    elif data_name == 'yelp2018':
        user_total = 45919
    else:
        user_total = 0
    start = 1000
    end = user_total
    while start < (end - 1000):
        use_num = int((start + end) * 0.5)
        print(data_name + ' trying ' + str(use_num) + ' with start ' + str(start) + ' end ' + str(end))
        if not os.path.exists('new_1k_datas/Data/' + data_name + '_new/'):
            os.mkdir('new_1k_datas/Data/' + data_name + '_new/')
        if not os.path.exists('new_1k_datas/pretrain/' + data_name + '_new/'):
            os.mkdir('new_1k_datas/pretrain/' + data_name + '_new/')

        save_all(data_name, use_num)

        new_d_name = data_name + '_new'
        # new_d_name='amazon_book_1w'
        num_new = check_num(new_d_name)
        print(data_name + ' see ' + str(use_num) + ' edge ' + str(num_new))
        if num_new < aim:
            start = use_num
        else:
            end = use_num
    print(data_name + ' set ' + str(use_num) + ' edge ' + str(num_new))
    f = open('new_1k_datas/mark_num.txt', 'a')
    f.write(data_name + ' set ' + str(use_num) + ' edge ' + str(num_new) + '\n')


'''
KG转同质图
just_homo('amazon-book_new')
'''


def save_same_homo(original_name, aim_par='_homo'):
    data_pos = 'new_1k_datas/Data/'
    data_pre = 'new_1k_datas/pretrain/'
    data_train = data_pos + original_name + '/train.txt'
    data_test = data_pos + original_name + '/test.txt'
    data_pretrain = data_pre + original_name + '/mf.npz'
    new_data_pa = data_pos + original_name + aim_par
    new_pret_pa = data_pre + original_name + aim_par
    if not os.path.exists(new_data_pa):
        os.mkdir(new_data_pa)
    if not os.path.exists(new_pret_pa):
        os.mkdir(new_pret_pa)
    new_data_train = new_data_pa + '/train.txt'
    new_data_test = new_data_pa + '/test.txt'
    new_data_pretrain = new_data_pa + '/mf.npz'
    copyfile(data_train, new_data_train)
    copyfile(data_test, new_data_test)
    copyfile(data_pretrain, new_data_pretrain)


def read_as_homo(original_name):
    data_pos = 'new_1k_datas/Data/'
    data_kg = data_pos + original_name + '/kg_final.txt'
    df_kg = pd.read_table(data_kg, sep=' ', header=None)
    poi_list = df_kg[0]
    rela_list = [0] * len(df_kg[1])
    enti_list = df_kg[2]
    df_kg_aim = pd.DataFrame({0: poi_list, 1: rela_list, 2: enti_list})
    return df_kg_aim


def just_homo(original_name):
    aim_par = '_homo'
    data_pos = 'new_1k_datas/Data/'
    save_same_homo(original_name, aim_par=aim_par)
    df_kg = read_as_homo(original_name)
    new_data_pa = data_pos + original_name + aim_par
    aim_path = new_data_pa + '/kg_final.txt'
    df_kg.to_csv(aim_path, index=False, sep=" ", header=None)  # 存文件


def pca_part(adj):
    n, m = adj.shape
    drop_rate = 1.0
    result = torch.pca_lowrank(torch.from_numpy(adj), q=int(m * drop_rate))
    k = int(m * drop_rate)
    # print(result)
    # (type(result))
    U = result[0]
    S = result[1]
    V = result[2]

    ret = torch.matmul(torch.from_numpy(adj), V[:, :k]).numpy()
    rate = S.numpy()
    rate = rate / (np.sum(rate))
    return ret, rate


def do_pca(original_name):
    aim_par = '_pca'
    data_pos = 'new_1k_datas/Data/'
    '''
    save_same_homo(original_name, aim_par= aim_par)
    '''
    df_kg = read_as_homo(original_name)
    head_list = df_kg[0]
    tail_list = df_kg[2]
    # print(head_list)
    # print(df_kg)
    # print(head_list[~head_list.isin(tail_list)])
    head_max = np.max(list(head_list[~head_list.isin(tail_list)]))
    tail_max = np.max(list(tail_list))
    poi_num = head_max + 1
    enti_nmu = tail_max - head_max
    # print(poi_num)
    # print(enti_nmu)
    adj = np.zeros((int(poi_num), int(enti_nmu)))
    for i in range(len(head_list)):
        head = head_list[i]
        tail = tail_list[i] - poi_num
        if (tail >= 0 and tail < enti_nmu and head >= 0 and head < poi_num):
            adj[head, tail] = 1
        else:
            # print(head,tail)
            pass
    print(adj.shape)
    # print(adj)
    result, rate = pca_part(adj)
    print(result.shape)
    np.save("new_1k_datas/notes/" + original_name + "_pca_result.npy", result)
    print(rate.shape)
    np.save("new_1k_datas/notes/" + original_name + "_pca_rate.npy", rate)
    print(rate)
    '''
    new_data_pa = data_pos + original_name + aim_par
    aim_path = new_data_pa + '/kg_final.txt'
    df_kg.to_csv(aim_path, index=False, sep=" ", header=None)  # 存文件
    '''


def load_pca(original_name):
    pca_file = "new_1k_datas/notes/" + original_name + "_pca_result.npy"
    features = np.load(pca_file)
    return features


def do_kmeans(features, clu_num):
    start=time.process_time()
    cluster = KMeans(n_clusters=clu_num,init = 'random', max_iter = 1).fit(features)
    end = time.process_time()
    print('kmeans time: '+str(end-start))
    # cluster = MiniBatchKMeans(n_clusters=clu_num, batch_size=128).fit(features)
    centers = cluster.cluster_centers_
    labels = cluster.labels_
    return centers


def read_rate(original_name):
    rate_file = "new_1k_datas/notes/" + original_name + "_pca_rate.npy"
    list_rate = np.load(rate_file)
    data = list_rate  # [0:200]
    print(data)
    plt.bar(range(len(data)), data)
    # plt.show()
    plt.savefig("new_1k_datas/notes/" + original_name + "_pca_rate_all.png")
    plt.cla()

    data = list_rate[0:200]
    print(data)
    plt.bar(range(len(data)), data)
    # plt.show()
    plt.savefig("new_1k_datas/notes/" + original_name + "_pca_rate_top_200.png")
    plt.cla()

    all_count = []
    p = []
    flag = 0
    count_flag = 0
    for rate in list_rate:
        count_flag += 1
        flag += rate

        all_count.append(flag)
        p.append(round(flag, 4))
    plt.plot(all_count, label=p)
    print(p)
    for i in range(len(p)):
        print(str(i) + ' : ' + str(p[i]))
    # plt.show()
    plt.savefig("new_1k_datas/notes/" + original_name + "_pca_rate_count.png")
    plt.cla()


if __name__ == '__main__':
    print('start')

    original_name = 'amazon-book_new'
    # do_pca(original_name)
    # read_rate(original_name)
    clu_num = 5000
    fea = load_pca(original_name)[:, :4000]
    print(fea.shape)
    cen = do_kmeans(fea, clu_num)
    print(cen)
    print(cen.shape)
    print('end')





