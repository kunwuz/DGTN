# -*- coding: utf-8 -*-
"""
Created on 5/4/2019
@author: RuihongQiu
"""


import numpy as np
import logging
import time



def forward(model, loader, device, writer, epoch, top_k=20, optimizer=None, train_flag=True):
    start = time.time()
    if train_flag:
        model.train()
    else:
        model.eval()
        hit10, mrr10 = [], []
        hit5, mrr5 = [], []
        hit20, mrr20 = [], []

    mean_loss = 0.0
    updates_per_epoch = len(loader)
    test_dict = {}
    for i, batch in enumerate(loader):
        if train_flag:
            optimizer.zero_grad()
        scores = model(batch.to(device))
        targets = batch.y - 1
        loss = model.loss_function(scores, targets)

        if train_flag:
            loss.backward()
            optimizer.step()
            writer.add_scalar('loss/train_batch_loss', loss.item(), epoch * updates_per_epoch + i)
        else:
            sub_scores = scores.topk(20)[1]    # batch * top_k
            for score, target in zip(sub_scores.detach().cpu().numpy(), targets.detach().cpu().numpy()):
                hit20.append(np.isin(target, score))
                if len(np.where(score == target)[0]) == 0:
                    mrr20.append(0)
                else:
                    mrr20.append(1 / (np.where(score == target)[0][0] + 1))

            sub_scores = scores.topk(top_k)[1]    # batch * top_k
            for score, target in zip(sub_scores.detach().cpu().numpy(), targets.detach().cpu().numpy()):
                hit10.append(np.isin(target, score))
                if len(np.where(score == target)[0]) == 0:
                    mrr10.append(0)
                else:
                    mrr10.append(1 / (np.where(score == target)[0][0] + 1))

            sub_scores = scores.topk(5)[1]    # batch * top_k
            for score, target in zip(sub_scores.detach().cpu().numpy(), targets.detach().cpu().numpy()):
                hit5.append(np.isin(target, score))
                if len(np.where(score == target)[0]) == 0:
                    mrr5.append(0)
                else:
                    mrr5.append(1 / (np.where(score == target)[0][0] + 1))


        mean_loss += loss / batch.num_graphs
        end = time.time()
        print("\rProcess: [%d/%d]   %.2f   usetime: %fs" % (i, updates_per_epoch, i/updates_per_epoch * 100, end - start),
              end='', flush=True)
    print('\n')

    if train_flag:
        writer.add_scalar('loss/train_loss', mean_loss.item(), epoch)
        print("Train_loss: ", mean_loss.item())
    else:
        writer.add_scalar('loss/test_loss', mean_loss.item(), epoch)
        hit20 = np.mean(hit20) * 100
        mrr20 = np.mean(mrr20) * 100

        hit10 = np.mean(hit10) * 100
        mrr10 = np.mean(mrr10) * 100

        hit5 = np.mean(hit5) * 100
        mrr5 = np.mean(mrr5) * 100
        # writer.add_scalar('index/hit', hit, epoch)
        # writer.add_scalar('index/mrr', mrr, epoch)
        print("Result:")
        print("\tMrr@", 20, ": ", mrr20)
        print("\tRecall@", 20, ": ", hit20)

        print("\tMrr@", top_k, ": ", mrr10)
        print("\tRecall@", top_k, ": ", hit10)

        print("\tMrr@", 5, ": ", mrr5)
        print("\tRecall@", 5, ": ", hit5)
        # for seq_len in range(1, 31):
        #     sub_hit = test_dict[seq_len][0]
        #     sub_mrr = test_dict[seq_len][1]
        #     print("Len ", seq_len, ": Recall@", top_k, ": ", np.mean(sub_hit) * 100, "Mrr@", top_k, ": ", np.mean(sub_mrr) * 100)

        return mrr20, hit20, mrr10, hit10, mrr5, hit5


def case_study(model, loader, device, n_node):
    model.eval()
    for i, batch in enumerate(loader):
        sc, ss, sg, mg, alpha_s, alpha_g = model(batch.to(device))
        targets = batch.y - 1
        scs = sc.topk(n_node)[1].detach().cpu().numpy()
        sss = ss.topk(n_node)[1].detach().cpu().numpy()
        sgs = sg.topk(n_node)[1].detach().cpu().numpy()
        mgs = mg.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()

        # batch * top_k
        for sc, ss, sg, ms, a_s, a_g, target in zip(scs, sss, sgs, mgs, alpha_s, alpha_g, targets):
            rc = np.where(sc == target)[0][0] + 1
            rs = np.where(ss == target)[0][0] + 1
            rg = np.where(sg == target)[0][0] + 1
            print("rank c:", rc, "rank s:", rs, "rank g:", rg, "gate:", ms)
            print("att s:", a_s, "att g:", a_g)


