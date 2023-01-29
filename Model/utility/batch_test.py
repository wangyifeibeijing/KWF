
import utility.metrics as metrics
from utility.parser import parse_args
import multiprocessing
import heapq
import numpy as np
import torch

from utility.loader_kgat import KGAT_loader
import os
import gc

cores = multiprocessing.cpu_count() // 2

args = parse_args()
Ks = eval(args.Ks)

data_generator = KGAT_loader(args=args, path=args.data_path + args.dataset)

batch_test_flag = False


USR_NUM, ITEM_NUM = data_generator.n_users, data_generator.n_items
N_TRAIN, N_TEST = data_generator.n_train, data_generator.n_test
BATCH_SIZE = args.batch_size
def get_hard_neg_item_dict_parallel(item):
    neg_item_list = set()
    kg_dict = data_generator.kg_dict
    entity_dict = data_generator.entity_dict
    black_entity_set = data_generator.black_entity_set
    if item in kg_dict:
        kg_list = kg_dict[item]
        for entity_info in kg_list:
            eid = entity_info[0]
            if eid in black_entity_set:
                continue
            if eid in entity_dict:
                n_items_list = entity_dict[eid]
                for n_item in n_items_list:
                    neg_item_list.add(n_item[0])
    # print("neg_item_list:", len(neg_item_list))
    return (item, neg_item_list)

def set_hard_neg_item_dict():
    print('\tset hard neg item data start.')

    u_hard_negitem_file = data_generator.path + '/u_hard_negitem_dict.txt'
    if os.path.exists(u_hard_negitem_file):
        lines = open(u_hard_negitem_file, 'r').readlines()
        if lines:
            for l in lines:
                tmps = l.strip()
                userid, item_list = tmps.split('\t')
                neg_item_list = [int(x) for x in item_list.split(',')]
                data_generator.u_hard_negitem_dict[int(userid)] = neg_item_list
    else:
        item_hard_negitem_file = data_generator.path + '/item_hard_negitem_dict.txt'
        u_batch_size = 300
        n_item_batchs = ITEM_NUM // u_batch_size + 1
        all_item_list = list(range(ITEM_NUM))
        count = 0
        pool = multiprocessing.Pool(10)
        item_item_dict = {}
        with open(item_hard_negitem_file, 'w') as f:
            for u_batch_id in range(n_item_batchs):
                gc.collect()
                start = u_batch_id * u_batch_size
                end = (u_batch_id + 1) * u_batch_size
                item_list_batch = all_item_list[start: end]
                result_list = pool.map(get_hard_neg_item_dict_parallel, item_list_batch)
                # count += len(result_list)
                for x, y in result_list:
                    # item_item_dict[x] = y
                    count += 1
                    f.write("%s\t%s\n" % (x, ",".join([str(t) for t in y])))
                    print('\rcount:', count, end='', flush=True)
                del result_list, item_list_batch
            f.close()
        assert count == data_generator.n_items
        pool.close()
        gc.collect()
        lines = open(item_hard_negitem_file, 'r').readlines()
        if lines:
            for l in lines:
                tmps = l.strip()
                item, item_list = tmps.split('\t')
                neg_item_list = [int(x) for x in item_list.split(',')]
                item_item_dict[int(item)] = neg_item_list

        for user, item_list in data_generator.train_user_dict.items():
            neg_item_list = set()
            for item in item_list:
                if item in item_item_dict:
                    neg_item_list.union(item_item_dict[item])
            data_generator.u_hard_negitem_dict[user] = list(neg_item_list)
        del item_item_dict
        with open(u_hard_negitem_file, 'w') as f:
            for x, y in data_generator.u_hard_negitem_dict.items():
                f.write("%s\t%s\n" % (x, ",".join([str(t) for t in y])))
            f.close()
    # data_generator.u_hard_negitem_dict = {x: y for x, y in result_list}
    print('\tset hard neg item data done.')

def ranklist_by_heapq(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = 0.
    return r, auc

def get_auc(item_score, user_pos_test):
    item_score = sorted(item_score.items(), key=lambda kv: kv[1])
    item_score.reverse()
    item_sort = [x[0] for x in item_score]
    posterior = [x[1] for x in item_score]

    r = []
    for i in item_sort:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = metrics.auc(ground_truth=r, prediction=posterior)
    return auc

def ranklist_by_sorted(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = get_auc(item_score, user_pos_test)
    return r, auc


def get_performance(user_pos_test, r, auc, Ks):
    precision, recall, ndcg, hit_ratio = [], [], [], []

    for K in Ks:
        precision.append(metrics.precision_at_k(r, K))
        recall.append(metrics.recall_at_k(r, K, len(user_pos_test)))
        ndcg.append(metrics.ndcg_at_k(r, K, user_pos_test))
        hit_ratio.append(metrics.hit_at_k(r, K))

    return {'recall': np.array(recall), 'precision': np.array(precision),
            'ndcg': np.array(ndcg), 'hit_ratio': np.array(hit_ratio), 'auc': auc}


def test_one_user(x):
    # user u's ratings for user u
    rating = x[0]
    #uid
    u = x[1]
    #user u's items in the training set
    try:
        training_items = data_generator.train_user_dict_o[u]
    except Exception:
        training_items = []
    #user u's items in the test set
    user_pos_test = data_generator.test_user_dict[u]

    all_items = set(range(ITEM_NUM))

    test_items = list(all_items - set(training_items))

    if args.test_flag == 'part':
        r, auc = ranklist_by_heapq(user_pos_test, test_items, rating, Ks)
    else:
        r, auc = ranklist_by_sorted(user_pos_test, test_items, rating, Ks)

    # # .......checking.......
    # try:
    #     assert len(user_pos_test) != 0
    # except Exception:
    #     print(u)
    #     print(training_items)
    #     print(user_pos_test)
    #     exit()
    # # .......checking.......

    return get_performance(user_pos_test, r, auc, Ks)

def top_popular_test(users_to_test):
    result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks)),
              'hit_ratio': np.zeros(len(Ks)), 'auc': 0.}

    pool = multiprocessing.Pool(cores)

    if args.model_type in ['ripple']:

        u_batch_size = BATCH_SIZE
        i_batch_size = BATCH_SIZE // 20
    elif args.model_type in ['fm', 'nfm']:
        u_batch_size = BATCH_SIZE
        i_batch_size = BATCH_SIZE
    else:
        u_batch_size = BATCH_SIZE * 2
        i_batch_size = BATCH_SIZE

    test_users = users_to_test
    n_test_users = len(test_users)
    n_user_batchs = n_test_users // u_batch_size + 1
    item_dict = data_generator.train_item_dict
    count = 0
    item_batch = range(ITEM_NUM)
    item_ratings = [0.] * ITEM_NUM
    for i in range(ITEM_NUM):
        if i in item_dict:
            item_ratings[i] = float(len(item_dict[i]))
    for u_batch_id in range(n_user_batchs):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size

        user_batch = test_users[start: end]

        rate_batch = np.expand_dims(np.array(item_ratings), axis=0).repeat(len(user_batch), axis=0)

        user_batch_rating_uid = zip(rate_batch, user_batch)
        batch_result = pool.map(test_one_user, user_batch_rating_uid)
        count += len(batch_result)

        for re in batch_result:
            result['precision'] += re['precision'] / n_test_users
            result['recall'] += re['recall'] / n_test_users
            result['ndcg'] += re['ndcg'] / n_test_users
            result['hit_ratio'] += re['hit_ratio'] / n_test_users
            result['auc'] += re['auc'] / n_test_users

    assert count == n_test_users
    pool.close()
    return result

def test(g, model, users_to_test, isMF=False, pretrain=[]):
    if(isMF):
        result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks)),
              'hit_ratio': np.zeros(len(Ks)), 'auc': 0.}

        pool = multiprocessing.Pool(cores)

        u_batch_size = BATCH_SIZE
        i_batch_size = BATCH_SIZE // 20
        test_users = users_to_test
        n_test_users = len(test_users)
        n_user_batchs = n_test_users // u_batch_size + 1

        count = 0

        for u_batch_id in range(n_user_batchs):
            start = u_batch_id * u_batch_size
            end = (u_batch_id + 1) * u_batch_size

            user_batch = test_users[start: end]

            item_batch = range(ITEM_NUM)
            with torch.no_grad():
                # embedding, _, _,_ = model(g)
                user_embed = pretrain['user_embed']
                item_embed = pretrain['item_embed']

                embedding = torch.FloatTensor(np.concatenate([user_embed, item_embed], axis=0)).cuda()

                user = embedding[user_batch]
                item = embedding[item_batch+data_generator.n_users]
                rate_batch = torch.mm(user, torch.transpose(item, 0, 1)).cpu().numpy()

            user_batch_rating_uid = zip(rate_batch, user_batch)
            batch_result = pool.map(test_one_user, user_batch_rating_uid)
            count += len(batch_result)

            for re in batch_result:
                result['precision'] += re['precision']/n_test_users
                result['recall'] += re['recall']/n_test_users
                result['ndcg'] += re['ndcg']/n_test_users
                result['hit_ratio'] += re['hit_ratio']/n_test_users
                result['auc'] += re['auc']/n_test_users


        assert count == n_test_users
        pool.close()
        return result
    model.eval()
    result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks)),
              'hit_ratio': np.zeros(len(Ks)), 'auc': 0.}

    pool = multiprocessing.Pool(cores)

    if args.model_type in ['ripple']:

        u_batch_size = BATCH_SIZE
        i_batch_size = BATCH_SIZE // 20
    elif args.model_type in ['fm', 'nfm']:
        u_batch_size = BATCH_SIZE
        i_batch_size = BATCH_SIZE
    else:
        u_batch_size = BATCH_SIZE * 2
        i_batch_size = BATCH_SIZE

    test_users = users_to_test
    n_test_users = len(test_users)
    n_user_batchs = n_test_users // u_batch_size + 1

    count = 0

    for u_batch_id in range(n_user_batchs):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size

        user_batch = test_users[start: end]

        item_batch = range(ITEM_NUM)
        with torch.no_grad():
            embedding, _, _, _, _ = model(g)
            user = embedding[user_batch]
            item = embedding[item_batch+data_generator.n_users]
            rate_batch = torch.mm(user, torch.transpose(item, 0, 1)).cpu().numpy()

        user_batch_rating_uid = zip(rate_batch, user_batch)
        batch_result = pool.map(test_one_user, user_batch_rating_uid)
        count += len(batch_result)

        for re in batch_result:
            result['precision'] += re['precision']/n_test_users
            result['recall'] += re['recall']/n_test_users
            result['ndcg'] += re['ndcg']/n_test_users
            result['hit_ratio'] += re['hit_ratio']/n_test_users
            result['auc'] += re['auc']/n_test_users


    assert count == n_test_users
    pool.close()
    return result


def save_file(g, model, users_to_test):
    model.eval()

    if args.model_type in ['ripple']:

        u_batch_size = BATCH_SIZE
        i_batch_size = BATCH_SIZE // 20
    elif args.model_type in ['fm', 'nfm']:
        u_batch_size = BATCH_SIZE
        i_batch_size = BATCH_SIZE
    else:
        u_batch_size = BATCH_SIZE * 2
        i_batch_size = BATCH_SIZE

    test_users = users_to_test
    n_test_users = len(test_users)
    n_user_batchs = n_test_users // u_batch_size + 1

    res = []

    for u_batch_id in range(n_user_batchs):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size

        user_batch = test_users[start: end]

        item_batch = range(ITEM_NUM)
        with torch.no_grad():
            embedding = model(g)
            user = embedding[user_batch]
            item = embedding[item_batch+data_generator.n_users]
            rate_batch = torch.mm(user, torch.transpose(item, 0, 1)).cpu().numpy()
            res.append(rate_batch)

    res = np.concatenate(res, axis=0)
    np.savetxt('{}_test_rate.txt'.format(args.dataset), res, fmt="%.06f")



def test_saved_file(users_to_test):
    res = np.loadtxt('{}_test_rate.txt'.format(args.dataset))
    result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks)),
              'hit_ratio': np.zeros(len(Ks)), 'auc': 0.}

    pool = multiprocessing.Pool(cores)

    if args.model_type in ['ripple']:

        u_batch_size = BATCH_SIZE
        i_batch_size = BATCH_SIZE // 20
    elif args.model_type in ['fm', 'nfm']:
        u_batch_size = BATCH_SIZE
        i_batch_size = BATCH_SIZE
    else:
        u_batch_size = BATCH_SIZE * 2
        i_batch_size = BATCH_SIZE

    test_users = users_to_test
    n_test_users = len(test_users)
    n_user_batchs = n_test_users // u_batch_size + 1

    count = 0

    for u_batch_id in range(n_user_batchs):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size

        user_batch = test_users[start: end]

        item_batch = range(ITEM_NUM)
        rate_batch = res[start:end]

        user_batch_rating_uid = zip(rate_batch, user_batch)
        batch_result = pool.map(test_one_user, user_batch_rating_uid)
        count += len(batch_result)

        for re in batch_result:
            result['precision'] += re['precision']/n_test_users
            result['recall'] += re['recall']/n_test_users
            result['ndcg'] += re['ndcg']/n_test_users
            result['hit_ratio'] += re['hit_ratio']/n_test_users
            result['auc'] += re['auc']/n_test_users


    assert count == n_test_users
    pool.close()
    return result

