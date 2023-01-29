
from utility.helper import *
from utility.batch_test import *
from time import time
from time import  strftime, localtime
from GNN import myGAT
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import logging

import os
import sys
def repeat_it(in_li,times,rep_f=False):
    if(rep_f):
        aim_user=np.mat(in_li).T
        aim_user_1=np.tile(aim_user,times)
        aim_user_2=aim_user_1.flatten()
        return(aim_user_2[0])
    else:
        u_l=[]
        for u in in_li:
            for i in range(times):
                u_l.append(u)
        return u_l


def load_pretrained_data(args):
    pre_model = 'mf'
    if args.pretrain == -2:
        pre_model = 'kgat'
    # pretrain_path = '%s../pretrain/%s/%s.npz' % (args.proj_path, args.dataset, pre_model)
    pretrain_path = '../pretrain/'+args.dataset+'/'+pre_model+'.npz'
    try:
        pretrain_data = np.load(pretrain_path)
        print('load the pretrained bprmf model parameters.')
    except Exception:
        pretrain_data = None
    return pretrain_data


if __name__ == '__main__':
    # set_hard_neg_item_dict()
    # 格式化成2016-03-20 11:45:39形式
    time_use = strftime("%Y%m%d%H%M", localtime())
    logger=logging.getLogger()
    logger.setLevel(level=logging.INFO)
    log_save_path = 'logs/'+args.dataset+'/'+'sample_rate_'+str(args.sprate)+'_time_'+time_use+'_log.txt'
    ensureDir(log_save_path)
    filehandle01=logging.FileHandler(log_save_path)
    filehandle01.setLevel(level=logging.INFO)
    #formatter01=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    formatter01=logging.Formatter('%(asctime)s \n %(message)s')
    filehandle01.setFormatter(formatter01)
    logger.addHandler(filehandle01)

    # get argument settings.
    # torch.manual_seed(2021)
    # np.random.seed(2019)
    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    """
    *********************************************************
    Load Data from data_generator function.
    """
    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items
    config['n_relations'] = data_generator.n_relations
    config['n_entities'] = data_generator.n_entities

    if args.model_type in ['kgat', 'cfkg']:
        "Load the laplacian matrix."
        config['A_in'] = sum(data_generator.lap_list)

        "Load the KG triplets."
        config['all_h_list'] = data_generator.all_h_list
        config['all_r_list'] = data_generator.all_r_list
        config['all_t_list'] = data_generator.all_t_list
        config['all_v_list'] = data_generator.all_v_list

    t0 = time()

    """
    *********************************************************
    Use the pretrained data to initialize the embeddings.
    """
    if args.pretrain in [-1, -2]:
        pretrain_data = load_pretrained_data(args)
    else:
        pretrain_data = None

    """
    *********************************************************
    Select one of the models.
    """
    weight_size = eval(args.layer_size)
    num_layers = len(weight_size) - 2
    heads = [args.heads] * num_layers + [1]
    model = myGAT(config['n_users']+config['n_entities'], args.kge_size, config['n_relations']+1, args.embed_size, weight_size[-2], weight_size[-1], num_layers, heads, F.elu, 0.1, 0., 0.01, False, pretrain=pretrain_data, alpha=args.alpha).cuda()
    if args.model_path:
        model = torch.load(args.model_path).cuda()
    edge2type = {}
    for i,mat in enumerate(data_generator.lap_list):
        for u,v in zip(*mat.nonzero()):
            edge2type[(u,v)] = i
    for i in range(data_generator.n_users+data_generator.n_entities):
        edge2type[(i,i)] = len(data_generator.lap_list)

    adjM = sum(data_generator.lap_list)
    log_edge_num = str(len(adjM.nonzero()[0]))
    logger.info("Total edge number: "+log_edge_num)
    print(len(adjM.nonzero()[0]))
    # g = dgl.DGLGraph(adjM)
    g = dgl.from_scipy(adjM)
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)
#     e_feat = []
#     edge2id = {}
#     for u, v in zip(*g.edges()):
#         u = u.item()
#         v = v.item()
#         if u == v:
#             break
#         e_feat.append(edge2type[(u,v)])
#         edge2id[(u,v)] = len(edge2id)
#     for i in range(data_generator.n_users+data_generator.n_entities):
#         e_feat.append(edge2type[(i,i)])
#         edge2id[(i,i)] = len(edge2id)
#     e_feat = torch.tensor(e_feat, dtype=torch.long)

    """
    *********************************************************
    Save the model parameters.
    """

    if args.save_flag == 1:
        weights_save_path = 'weights/'+args.dataset+'/'+'sample_rate_'+str(args.sprate)+'_time_'+time_use+'.pt'

        ensureDir(weights_save_path)
        torch.save(model, weights_save_path)

    cur_best_pre_0 = 0.

    """
    *********************************************************
    Train.
    """
    loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
    stopping_step = 0
    should_stop = False
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    g = g.to('cuda')
#     e_feat = e_feat.cuda()
    for epoch in range(args.epoch):
        log_aim='epoch '+str(epoch)+'-'*100+'\n'
        logger.info(log_aim)
        t1 = time()
        loss, base_loss, ue_loss, ei_loss, reg_loss = 0., 0., 0., 0., 0.
        n_batch = data_generator.n_train // args.batch_size + 1

        """
        *********************************************************
        Alternative Training for KGAT:
        ... phase 1: to train the recommender.
        """
        for idx in range(n_batch):
            model.train()
            btime= time()

            batch_data = data_generator.generate_train_batch()

            embedding, embedding_trans, e_embedings,_,_ = model(g)

            u_emb = embedding[batch_data['users']]
            p_emb = embedding[batch_data['pos_items']+data_generator.n_users]
            n_emb = embedding[batch_data['neg_items']+data_generator.n_users]
            pos_scores = (u_emb * p_emb).sum(dim=1)
            neg_scores = (u_emb * n_emb).sum(dim=1)
            base_loss = F.softplus(-pos_scores+neg_scores).mean()
            if args.ue_lambda > 0:
                sprate = args.sprate
                user_use=batch_data['users']
                ue_emb = embedding_trans[user_use]
                n1,n2=ue_emb.size()
                ue_emb = ue_emb.unsqueeze(1).repeat(1, sprate, 1).reshape(n1*sprate,n2)
                pe_emb = e_embedings[batch_data['pos_entities']]
                ne_emb = e_embedings[batch_data['neg_entities']]
                pos_e_scores = (ue_emb * pe_emb).sum(dim=1)
                neg_e_scores = (ue_emb * ne_emb).sum(dim=1)
                ue_loss = F.softplus(-pos_e_scores + neg_e_scores).mean()
                pos_i_use=[]
                neg_i_use=[]
                pos_i_emb = embedding_trans[batch_data['pos_items']]
                n1,n2=pos_i_emb.size()
                pos_i_emb = pos_i_emb.unsqueeze(1).repeat(1, sprate, 1).reshape(n1*sprate,n2)
                # neg_i_emb = embedding_trans[batch_data['neg_items']]
                # neg_i_emb = neg_i_emb.unsqueeze(1).repeat(1, sprate, 1).reshape(n1*sprate,n2)

                pos_i_scores = (pos_i_emb * pe_emb).sum(dim=1)
                neg_i_scores = (pos_i_emb * ne_emb).sum(dim=1)
                ei_loss = F.softplus(-pos_i_scores + neg_i_scores).mean()
                # reg_loss = args.weight_decay * ((u_emb*u_emb).sum()/2 + (p_emb*p_emb).sum()/2 + (n_emb*n_emb).sum()/2) / args.batch_size
                # loss = base_loss + reg_loss # Since we don't do accum, the printed loss may seem smaller
                reg_loss = args.weight_decay * ((u_emb * u_emb).sum() / 2 + (p_emb * p_emb).sum() / 2 + (n_emb * n_emb).sum() / 2 + (pe_emb * pe_emb).sum() / 2 + (ne_emb * ne_emb).sum() / 2) / args.batch_size
                # loss = base_loss + 0.05 * (ue_loss+ei_loss) + reg_loss
                # loss = base_loss + 0.1 * (0.8 * ue_loss + 0.2 * ei_loss) + reg_loss
                loss = base_loss + args.ue_lambda * ue_loss + reg_loss
            else:
                reg_loss = args.weight_decay * ((u_emb * u_emb).sum() / 2 + (p_emb * p_emb).sum() / 2 + (
                            n_emb * n_emb).sum() / 2) / args.batch_size
                loss = base_loss + reg_loss  # Since we don't do accum, the printed loss may seem smaller
            if idx % 100 == 0:
                print(idx, loss)
                log_aim='idx '+str(epoch)+' loss '+str(loss)+'\n'
                logger.info(log_aim)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            

        show_step = 10
        if ((epoch + 1) % show_step != 0 and epoch != 0):
            if args.verbose > 0 and epoch % args.verbose == 0:
                perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f + %.5f + %.5f]' % (
                    epoch, time() - t1, loss, base_loss, ue_loss, ei_loss, reg_loss)
                print(perf_str)
                log_aim=perf_str
                logger.info(log_aim)
            continue

        """
        *********************************************************
        Test.
        """
        t2 = time()
        users_to_test = list(data_generator.test_user_dict.keys())

        ret = test(g, model, users_to_test)
        # ret = test(g, model, users_to_test, isMF=True, pretrain=pretrain_data)
        # ret = top_popular_test(users_to_test)

        """
        *********************************************************
        Performance logging.
        """
        t3 = time()

        loss_loger.append(loss)
        rec_loger.append(ret['recall'])
        pre_loger.append(ret['precision'])
        ndcg_loger.append(ret['ndcg'])
        hit_loger.append(ret['hit_ratio'])

        if args.verbose > 0:
            perf_str = 'Epoch %d [%.1fs + %.1fs]: train==[%.5f=%.5f + %.5f + %.5f + %.5f], recall=[%.5f, %.5f], ' \
                       'precision=[%.5f, %.5f], hit=[%.5f, %.5f], ndcg=[%.5f, %.5f]' % \
                       (epoch, t2 - t1, t3 - t2, loss, base_loss, ue_loss, ei_loss, reg_loss, ret['recall'][0], ret['recall'][-1],
                        ret['precision'][0], ret['precision'][-1], ret['hit_ratio'][0], ret['hit_ratio'][-1],
                        ret['ndcg'][0], ret['ndcg'][-1])
            print(perf_str)
            log_aim=perf_str
            logger.info(log_aim)
        cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best_pre_0,
                                                                    stopping_step, expected_order='acc', flag_step=50)

        # *********************************************************
        # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
        if should_stop == True:
            break

        # *********************************************************
        # save the user & item embeddings for pretraining.
        if ret['recall'][0] == cur_best_pre_0 and args.save_flag == 1:
            # save_saver.save(sess, weights_save_path + '/weights', global_step=epoch)
            torch.save(model, weights_save_path)
            print('save the weights in path: ', weights_save_path)
            log_aim='save the weights in path: '+ weights_save_path
            logger.info(log_aim)
            # print('saving prediction')
            # save_file(g, e_feat, model, users_to_test)
            # print('saved')
            # print(test_saved_file(users_to_test))

    recs = np.array(rec_loger)
    pres = np.array(pre_loger)
    ndcgs = np.array(ndcg_loger)
    hit = np.array(hit_loger)

    best_rec_0 = max(recs[:, 0])
    idx = list(recs[:, 0]).index(best_rec_0)

    final_perf = "Best Iter=[%d]@[%.1f]\trecall=[%s], precision=[%s], hit=[%s], ndcg=[%s]" % \
                 (idx, time() - t0, '\t'.join(['%.5f' % r for r in recs[idx]]),
                  '\t'.join(['%.5f' % r for r in pres[idx]]),
                  '\t'.join(['%.5f' % r for r in hit[idx]]),
                  '\t'.join(['%.5f' % r for r in ndcgs[idx]]))
    print(final_perf)
    log_aim=final_perf
    logger.info(log_aim)
    save_path = '%soutput/%s/%s.result' % (args.proj_path, args.dataset, args.model_type)
    ensureDir(save_path)
    f = open(save_path, 'a')

    f.write('embed_size=%d, lr=%.4f, layer_size=%s, node_dropout=%s, mess_dropout=%s, regs=%s, adj_type=%s, use_att=%s, use_kge=%s, pretrain=%d\n\t%s\n'
            % (args.embed_size, args.lr, args.layer_size, args.node_dropout, args.mess_dropout, args.regs, args.adj_type, args.use_att, args.use_kge, args.pretrain, final_perf))
    f.close()
