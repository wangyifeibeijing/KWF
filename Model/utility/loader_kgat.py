
import numpy as np
from utility.load_data import Data
from time import time
import scipy.sparse as sp
import random as rd
import collections
import numba
from numba.typed import Dict, List
from numba import types
import multiprocessing

# numba.config.NUMBA_DEFAULT_NUM_THREADS=multiprocessing.cpu_count() // 2
# @numba.njit(parallel=True)
# def _get_hard_neg_item_dict_parallel(train_user_list, train_items_list, kg_dict, ue_black_dict, entity_dict, n_users):
#     # int_list_list = types.ListType(types.ListType(types.int32))
#     # int_list = types.ListType(types.int32)
#     # user_list = List(lsttype=int_list)
#     # result_list = List(lsttype=int_list_list)
#     # user_list = [numba.int64(0)]*len(train_user_list)
#     result_list = [[0]]*len(train_user_list)
#     count = numba.int64(0)
#     for i in numba.prange(len(train_user_list)):
#         # print('\rcount:', count, end='', flush=True)
#         if count % 1000 == 0:
#             print('count:', count)
#         user = train_user_list[i]
#         item_list = train_items_list[i]
#         # neg_item_list = List(lsttype=int_list)
#         neg_item_list = []
#         # for j in numba.prange(len(item_list)):
#         #     item = item_list[j]
#         for item in item_list:
#             if item in kg_dict:
#                 kg_list = kg_dict[item]
#                 # for k in numba.prange(len(kg_list)):
#                 #     eid = kg_list[k]
#                 for eid in kg_list:
#                     # if user in ue_black_dict and np.argwhere(ue_black_dict[user]==eid + n_users).shape[0]!=0:
#                     if user in ue_black_dict and eid + n_users in ue_black_dict[user]:
#                         continue
#                     if eid in entity_dict:
#                         n_items_list = entity_dict[eid]
#                         # for m in numba.prange(len(n_items_list)):
#                         #     n_item = n_items_list[m]
#                         for n_item in entity_dict[eid]:
#                             neg_item_list.append(n_item)
#                             # print('\rcount:', count, m, end='', flush=True)
#                         # neg_item_list.append(entity_dict[eid])
#
#         # user_list.append(user)
#         result_list[i] = neg_item_list
#         count += 1
#     return result_list

class KGAT_loader(Data):
    def __init__(self, args, path):
        super().__init__(args, path)
        self.sprate = args.sprate
        self.idf_sampling = args.idf_sampling

        # ue_black_file = path + '/amazon_book_neg_enti.txt'
        # ue_black_file = path + '/amazon_book_neg_enti_90%.txt'
        print(args.dataset)
        if(args.entirate<=100):
            ue_black_file = path + '/'+args.dataset+'_neg_enti_'+str(args.entirate)+'%.txt'

            self.ue_black_dict, self.black_entity_set = self._load_ue_black_dict(ue_black_file)
        else:
            self.ue_black_dict ={}
            self.black_entity_set = set()
        if args.idf_sampling!=0:
            idf_file = path + '/' + args.dataset + '_idf_dic.npy'
            self.ue_dict, self.ue_idf_dict = self._load_eu_idf(idf_file)

        # generate the sparse adjacency matrices for user-item interaction & relational kg data.
        self.adj_list, self.adj_r_list = self._get_relational_adj_list()

        # generate the sparse laplacian matrices.
        self.lap_list = self._get_relational_lap_list()

        # generate the triples dictionary, key is 'head', value is '(tail, relation)'.
        self.all_kg_dict = self._get_all_kg_dict()

        self.all_h_list, self.all_r_list, self.all_t_list, self.all_v_list = self._get_all_kg_data()

    def _load_eu_idf(self, file_name):
        aim = np.load(file_name, allow_pickle=True).item()
        ue_dict = {}
        ue_idf_dict = {}
        ue_idf_sum = {}
        for pair, value in aim.items():
            entity, user = pair
            if user in self.ue_black_dict and entity in self.ue_black_dict[user]:
                continue
            ue_dict.setdefault(user, [])
            ue_idf_dict.setdefault(user, [])
            ue_idf_sum.setdefault(user, 0)
            ue_dict[user].append(entity-self.n_users)
            if value <= 0:
                raise ValueError("idf <=0")
            prop = 1 / (value + 1e-8)
            ue_idf_sum[user] += prop
            ue_idf_dict[user].append(prop)
        for u, s in ue_idf_sum.items():
            ue_idf_dict[u] /= s
        return ue_dict, ue_idf_dict

    def _load_ue_black_dict(self, file_name):
        user_dict = collections.defaultdict(set)
        black_entity_set = set()
        lines = open(file_name, 'r').readlines()
        if lines:
            for l in lines:
                tmps = l.strip()
                userid, eid = tmps.split('\t')
                user_dict[int(userid)].add(int(eid))
                black_entity_set.add(int(eid))
        return user_dict, black_entity_set

    # def _get_hard_neg_item_dict(self):
    #     print('\tget hard neg item data start.')
    #     # int_list_list = types.ListType(types.ListType(types.int32))
    #     int_list = types.ListType(types.int64)
    #     # train_user_list = list(self.train_user_dict.keys())
    #     # train_items_list = list(self.train_user_dict.values())
    #     train_user_list = List.empty_list(types.int64)
    #     train_items_list = List.empty_list(types.ListType(types.int64))
    #     for x,y in self.train_user_dict.items():
    #         tmp = List.empty_list(types.int64)
    #         for t in y:
    #             tmp.append(numba.int64(t))
    #         train_user_list.append(numba.int64(x))
    #         train_items_list.append(tmp)
    #     # int_array = types.int32[:]
    #     kg_dict = Dict.empty(
    #         key_type=types.int64,
    #         value_type=int_list,
    #     )
    #     ue_black_dict = Dict.empty(
    #         key_type=types.int64,
    #         value_type=int_list,
    #     )
    #     entity_dict = Dict.empty(
    #         key_type=types.int64,
    #         value_type=int_list,
    #     )
    #     for x,y in self.kg_dict.items():
    #         in_lst = List(lsttype=int_list)
    #         for t in y:
    #             in_lst.append(numba.int64(t[0]))
    #         kg_dict[x] = in_lst
    #         # kg_dict[x] = np.asarray([t[0] for t in y], dtype='int32')
    #     print('\tconv kg_dict done.')
    #     for x,y in self.ue_black_dict.items():
    #         in_lst = List(lsttype=int_list)
    #         for t in y:
    #             in_lst.append(numba.int64(t))
    #         ue_black_dict[x] = in_lst
    #         # ue_black_dict[x] = np.asarray(list(y), dtype='int32')
    #     print('\tconv ue_black_dict done.')
    #     for x,y in self.entity_dict.items():
    #         in_lst = List(lsttype=int_list)
    #         for t in y:
    #             in_lst.append(numba.int64(t[0]))
    #         entity_dict[x] = in_lst
    #         # entity_dict[x] = np.asarray([t[0] for t in y], dtype='int32')
    #     # entity_dict = {x:[t[0] for t in y] for x,y in self.entity_dict.items()}
    #     print('\tconv entity_dict done.')
    #     hard_neg_item_list = _get_hard_neg_item_dict_parallel(train_user_list, train_items_list, kg_dict, ue_black_dict, entity_dict, numba.int64(self.n_users))
    #     print('\tget hard neg item data done.')
    #     return {int(x): y for x, y in zip(train_user_list, hard_neg_item_list)}

    def _get_relational_adj_list(self):
        t1 = time()
        adj_mat_list = []
        adj_r_list = []

        def _np_mat2sp_adj(np_mat, row_pre, col_pre):
            n_all = self.n_users + self.n_entities
            # single-direction
            a_rows = np_mat[:, 0] + row_pre
            a_cols = np_mat[:, 1] + col_pre
            a_vals = [1.] * len(a_rows)

            b_rows = a_cols
            b_cols = a_rows
            b_vals = [1.] * len(b_rows)

            a_adj = sp.coo_matrix((a_vals, (a_rows, a_cols)), shape=(n_all, n_all))
            b_adj = sp.coo_matrix((b_vals, (b_rows, b_cols)), shape=(n_all, n_all))

            return a_adj, b_adj

        def _user2entity_adj_new(row_pre, col_pre):
            R_list = []
            adj_r_list = []
            n_all = self.n_users + self.n_entities
            a_rows_map = collections.defaultdict(list)
            a_cols_map = collections.defaultdict(list)
            for user in self.train_user_dict.keys():
                for item in self.train_user_dict[user]:
                    if item in self.kg_dict:
                        for entity_info in self.kg_dict[item]:
                            uid = user+row_pre
                            eid = entity_info[0]+col_pre
                            if uid in self.ue_black_dict and eid in self.ue_black_dict[uid]:
                                continue
                            a_rows_map[entity_info[1]].append(uid)
                            a_cols_map[entity_info[1]].append(eid)

            for r_id in a_rows_map.keys():
                a_rows = a_rows_map[r_id]
                a_cols = a_cols_map[r_id]
                a_vals = [1.] * len(a_rows)
                b_rows = a_cols
                b_cols = a_rows
                b_vals = [1.] * len(b_rows)
                a_adj = sp.coo_matrix((a_vals, (a_rows, a_cols)), shape=(n_all, n_all))
                b_adj = sp.coo_matrix((b_vals, (b_rows, b_cols)), shape=(n_all, n_all))
                R_list.append(a_adj)
                adj_r_list.append(r_id)
                R_list.append(b_adj)
                adj_r_list.append(r_id + self.n_relations)
            return R_list, adj_r_list

        def _user2entity_adj(row_pre, col_pre):
            n_all = self.n_users + self.n_entities
            # single-direction
            a_rows = []
            a_cols = []
            for user in self.train_user_dict.keys():
                for item in self.train_user_dict[user]:
                    if item in self.kg_dict:
                        for entity_info in self.kg_dict[item]:
                            a_rows.append(user+row_pre)
                            a_cols.append(entity_info[0]+col_pre)

            a_vals = [1.] * len(a_rows)

            b_rows = a_cols
            b_cols = a_rows
            b_vals = [1.] * len(b_rows)
            a_adj = sp.coo_matrix((a_vals, (a_rows, a_cols)), shape=(n_all, n_all))
            b_adj = sp.coo_matrix((b_vals, (b_rows, b_cols)), shape=(n_all, n_all))

            return a_adj, b_adj

        R_list, r_list = _user2entity_adj_new(row_pre=0, col_pre=self.n_users)
        adj_mat_list.extend(R_list)
        adj_r_list.extend(r_list)

#         R, R_inv = _np_mat2sp_adj(self.train_data, row_pre=0, col_pre=self.n_users)
#         adj_mat_list.append(R)
#         # adj_r_list.append(0)
#         adj_r_list.append(2 * self.n_relations)
#         adj_mat_list.append(R_inv)
#         # adj_r_list.append(self.n_relations + 1)
#         adj_r_list.append(2 * self.n_relations + 1)

        # R, R_inv = _user2entity_adj(row_pre=0, col_pre=self.n_users)
        # adj_mat_list.append(R)
        # adj_r_list.append(0)
        # adj_mat_list.append(R_inv)
        # adj_r_list.append(self.n_relations + 1)
        print('\tconvert ratings into adj mat done.')

        # key = int(129954-self.n_users)
        # print("entity_dict:", self.entity_dict[key])
        for r_id in self.relation_dict.keys():
            K, K_inv = _np_mat2sp_adj(np.array(self.relation_dict[r_id]), row_pre=self.n_users, col_pre=self.n_users)
            adj_mat_list.append(K)
            # adj_r_list.append(r_id + 1)
            adj_r_list.append(r_id + 2 * self.n_relations)
#             adj_r_list.append(r_id + 2 * self.n_relations + 2)

            adj_mat_list.append(K_inv)
            # adj_r_list.append(r_id + 2 + self.n_relations)
            adj_r_list.append(r_id + 3 * self.n_relations)
#             adj_r_list.append(r_id + 3 * self.n_relations + 2)
        print('\tconvert %d relational triples into adj mat done. @%.4fs' %(len(adj_mat_list), time()-t1))

        self.n_relations = len(adj_r_list)
        # print('\tadj relation list is', adj_r_list)

        return adj_mat_list, adj_r_list

    def _get_relational_lap_list(self):
        def _bi_norm_lap(adj):
            rowsum = np.array(adj.sum(1))

            d_inv_sqrt = np.power(rowsum, -0.5).flatten()
            d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
            d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

            bi_lap = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
            return bi_lap.tocoo()

        def _si_norm_lap(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            return norm_adj.tocoo()

        if self.args.adj_type == 'bi':
            lap_list = [_bi_norm_lap(adj) for adj in self.adj_list]
            print('\tgenerate bi-normalized adjacency matrix.')
        else:
            lap_list = [_si_norm_lap(adj) for adj in self.adj_list]
            print('\tgenerate si-normalized adjacency matrix.')
        return lap_list

    def _get_all_kg_dict(self):
        all_kg_dict = collections.defaultdict(list)
        for l_id, lap in enumerate(self.lap_list):

            rows = lap.row
            cols = lap.col

            for i_id in range(len(rows)):
                head = rows[i_id]
                tail = cols[i_id]
                relation = self.adj_r_list[l_id]

                all_kg_dict[head].append((tail, relation))
        return all_kg_dict

    def _get_all_kg_data(self):
        def _reorder_list(org_list, order):
            new_list = np.array(org_list)
            new_list = new_list[order]
            return new_list

        all_h_list, all_t_list, all_r_list = [], [], []
        all_v_list = []

        for l_id, lap in enumerate(self.lap_list):
            all_h_list += list(lap.row)
            all_t_list += list(lap.col)
            all_v_list += list(lap.data)
            all_r_list += [self.adj_r_list[l_id]] * len(lap.row)

        assert len(all_h_list) == sum([len(lap.data) for lap in self.lap_list])

        # resort the all_h/t/r/v_list,
        # ... since tensorflow.sparse.softmax requires indices sorted in the canonical lexicographic order
        print('\treordering indices...')
        org_h_dict = dict()

        for idx, h in enumerate(all_h_list):
            if h not in org_h_dict.keys():
                org_h_dict[h] = [[],[],[]]

            org_h_dict[h][0].append(all_t_list[idx])
            org_h_dict[h][1].append(all_r_list[idx])
            org_h_dict[h][2].append(all_v_list[idx])
        print('\treorganize all kg data done.')

        sorted_h_dict = dict()
        for h in org_h_dict.keys():
            org_t_list, org_r_list, org_v_list = org_h_dict[h]
            sort_t_list = np.array(org_t_list)
            sort_order = np.argsort(sort_t_list)

            sort_t_list = _reorder_list(org_t_list, sort_order)
            sort_r_list = _reorder_list(org_r_list, sort_order)
            sort_v_list = _reorder_list(org_v_list, sort_order)

            sorted_h_dict[h] = [sort_t_list, sort_r_list, sort_v_list]
        print('\tsort meta-data done.')

        od = collections.OrderedDict(sorted(sorted_h_dict.items()))
        new_h_list, new_t_list, new_r_list, new_v_list = [], [], [], []

        for h, vals in od.items():
            new_h_list += [h] * len(vals[0])
            new_t_list += list(vals[0])
            new_r_list += list(vals[1])
            new_v_list += list(vals[2])


        assert sum(new_h_list) == sum(all_h_list)
        assert sum(new_t_list) == sum(all_t_list)
        assert sum(new_r_list) == sum(all_r_list)
        # try:
        #     assert sum(new_v_list) == sum(all_v_list)
        # except Exception:
        #     print(sum(new_v_list), '\n')
        #     print(sum(all_v_list), '\n')
        print('\tsort all data done.')


        return new_h_list, new_r_list, new_t_list, new_v_list

    def _generate_train_A_batch(self):
        exist_heads = self.all_kg_dict.keys()

        if self.batch_size_kg <= len(exist_heads):
            heads = rd.sample(exist_heads, self.batch_size_kg)
        else:
            heads = [rd.choice(exist_heads) for _ in range(self.batch_size_kg)]

        def sample_pos_triples_for_h(h, num):
            pos_triples = self.all_kg_dict[h]
            n_pos_triples = len(pos_triples)

            pos_rs, pos_ts = [], []
            while True:
                if len(pos_rs) == num: break
                pos_id = np.random.randint(low=0, high=n_pos_triples, size=1)[0]

                t = pos_triples[pos_id][0]
                r = pos_triples[pos_id][1]

                if r not in pos_rs and t not in pos_ts:
                    pos_rs.append(r)
                    pos_ts.append(t)
            return pos_rs, pos_ts

        def sample_neg_triples_for_h(h, r, num):
            neg_ts = []
            while True:
                if len(neg_ts) == num: break

                t = np.random.randint(low=0, high=self.n_users + self.n_entities, size=1)[0]
                if (t, r) not in self.all_kg_dict[h] and t not in neg_ts:
                    neg_ts.append(t)
            return neg_ts

        pos_r_batch, pos_t_batch, neg_t_batch = [], [], []

        for h in heads:
            pos_rs, pos_ts = sample_pos_triples_for_h(h, 1)
            pos_r_batch += pos_rs
            pos_t_batch += pos_ts

            neg_ts = sample_neg_triples_for_h(h, pos_rs[0], 1)
            neg_t_batch += neg_ts

        return heads, pos_r_batch, pos_t_batch, neg_t_batch

    def generate_train_batch(self):
        users, pos_items, neg_items, pos_entities, neg_entities = self._generate_train_cf_batch()

        batch_data = {}
        batch_data['users'] = users
        batch_data['pos_items'] = pos_items
        batch_data['neg_items'] = neg_items
        batch_data['pos_entities'] = pos_entities
        batch_data['neg_entities'] = neg_entities

        return batch_data

    def _generate_train_cf_batch(self):
        if self.batch_size <= self.n_users:
            users = rd.sample(self.exist_users, self.batch_size)
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]

        # users = list(self.exist_users)[:self.batch_size]

        def sample_pos_items_for_u(u, num):
            pos_items = self.train_user_dict[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num: break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            neg_items = []
            while True:
                if len(neg_items) == num: break
                neg_i_id = np.random.randint(low=0, high=self.n_items, size=1)[0]

                if neg_i_id not in self.train_user_dict[u] and neg_i_id not in neg_items:
                    neg_items.append(neg_i_id)
            return neg_items

        # def sample_neg_e_items_for_u(u, num):
        #     neg_items = []
        #     while True:
        #         if len(neg_items) == num: break
        #         if u in self.u_hard_negitem_dict:
        #             hard_negitem_list = self.u_hard_negitem_dict[u]
        #             index = np.random.randint(low=0, high=len(hard_negitem_list), size=1)[0]
        #             neg_i_id = hard_negitem_list[index]
        #             if neg_i_id not in neg_items:
        #                 neg_items.append(neg_i_id)
        #     return neg_items


        def sample_entity_for_i(item, num):
            entity_list = []
            while True:
                if len(entity_list) == num: break
                if item in self.kg_dict:
                    entityinfo_list = self.kg_dict[item]
                    index = np.random.randint(low=0, high=len(entityinfo_list), size=1)[0]
                    eid = entityinfo_list[index][0] - self.n_items
                else:
                    eid = np.random.randint(low=0, high=self.n_entities - self.n_items, size=1)[0]
                entity_list.append(eid)
            return entity_list

            # if item in self.kg_dict:
            #     for entity_info in self.kg_dict[item]:
            #         eid = entity_info[0] + self.n_users
            #         if u in self.ue_black_dict and eid in self.ue_black_dict[u]:
            #             continue
            #         entity_set.add(eid)
            # entity_list = list(entity_set)
            # u_list = [u] * len(entity_list)
            # return u_list,  entity_list

        def sample_entity_for_i_new(user, num):
            entity_list = []
            while True:
                if len(entity_list) == num: break
                if user in self.ue_idf_dict and user in self.ue_dict:
                    entityinfo_list = self.ue_dict[user]
                    prop_list = self.ue_idf_dict[user]
                    eid = np.random.choice(a=entityinfo_list, size=1, replace=True, p=prop_list)[0]
                    eid = eid - self.n_items
                # else:
                #     eid = np.random.randint(low=0, high=self.n_entities - self.n_items, size=1)[0]
                else:
                    eid = np.random.randint(low=0, high=self.n_entities - self.n_items, size=1)[0]
                entity_list.append(eid)
            return entity_list

        pos_items, neg_items = [], []
        pos_entities, neg_entities = [], []
        for u in users:
            pos_item = sample_pos_items_for_u(u, 1)
            neg_item = sample_neg_items_for_u(u, 1)
            pos_items += pos_item
            neg_items += neg_item

            # neg_e_item = sample_neg_e_items_for_u(u, 1)
            pos_entity = sample_entity_for_i(pos_item[0], self.sprate)
            if self.idf_sampling == 1:
                neg_entity = sample_entity_for_i_new(u, self.sprate)
            else:
                neg_entity = sample_entity_for_i(neg_item[0], self.sprate)
            # neg_entity = sample_entity_for_i(neg_e_item[0], self.sprate)
            # neg_entity = sample_entity_for_i(neg_item[0], self.sprate)
            pos_entities += pos_entity
            neg_entities += neg_entity

        return users, pos_items, neg_items, pos_entities, neg_entities

    def generate_train_feed_dict(self, model, batch_data):
        feed_dict = {
            model.users: batch_data['users'],
            model.pos_items: batch_data['pos_items'],
            model.neg_items: batch_data['neg_items'],

            model.mess_dropout: eval(self.args.mess_dropout),
            model.node_dropout: eval(self.args.node_dropout),
        }

        return feed_dict

    def generate_train_A_batch(self):
        heads, relations, pos_tails, neg_tails = self._generate_train_A_batch()

        batch_data = {}

        batch_data['heads'] = heads
        batch_data['relations'] = relations
        batch_data['pos_tails'] = pos_tails
        batch_data['neg_tails'] = neg_tails
        return batch_data

    def generate_train_A_feed_dict(self, model, batch_data):
        feed_dict = {
            model.h: batch_data['heads'],
            model.r: batch_data['relations'],
            model.pos_t: batch_data['pos_tails'],
            model.neg_t: batch_data['neg_tails'],

        }

        return feed_dict


    def generate_test_feed_dict(self, model, user_batch, item_batch, drop_flag=True):

        feed_dict ={
            model.users: user_batch,
            model.pos_items: item_batch,
            model.mess_dropout: [0.] * len(eval(self.args.layer_size)),
            model.node_dropout: [0.] * len(eval(self.args.layer_size)),

        }

        return feed_dict

