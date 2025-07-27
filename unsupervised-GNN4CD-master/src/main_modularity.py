import numpy as np
import os
from data_generator import Generator
from load import get_gnn_inputs
from models import GNN_multiclass
import time
import argparse
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
import time
import numpy as np
import torch
import pandas as pd
from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from losses import compute_loss_multiclass, compute_accuracy_multiclass,compute_modularity_loss_multiclass


parser = argparse.ArgumentParser()

###############################################################################
#                             General Settings                                #
###############################################################################

parser.add_argument('--num_examples_train', nargs='?', const=1, type=int,
                    default=int(6000))
parser.add_argument('--num_examples_test', nargs='?', const=1, type=int,
                    default=int(1000))
parser.add_argument('--edge_density', nargs='?', const=1, type=float,
                    default=0.2)
parser.add_argument('--p_SBM', nargs='?', const=1, type=float,
                    default=0.0394)
parser.add_argument('--q_SBM', nargs='?', const=1, type=float,
                    default=0.0297)
parser.add_argument('--random_noise', action='store_true')
parser.add_argument('--noise', nargs='?', const=1, type=float, default=0.03)
parser.add_argument('--noise_model', nargs='?', const=1, type=int, default=2)
#########################
#parser.add_argument('--generative_model', nargs='?', const=1, type=str,
#                    default='ErdosRenyi')
parser.add_argument('--generative_model', nargs='?', const=1, type=str,
                    default='SBM_multiclass')
parser.add_argument('--batch_size', nargs='?', const=1, type=int, default=1)
parser.add_argument('--mode', nargs='?', const=1, type=str, default='train')

default_path = os.path.join(os.path.dirname(os.path.abspath(__file__)))
parser.add_argument('--path_gnn', nargs='?', const=1, type=str, default=default_path)
parser.add_argument('--filename_existing_gnn', nargs='?', const=1, type=str, default='')
parser.add_argument('--print_freq', nargs='?', const=1, type=int, default=100)
parser.add_argument('--test_freq', nargs='?', const=1, type=int, default=500)
parser.add_argument('--save_freq', nargs='?', const=1, type=int, default=2000)
parser.add_argument('--clip_grad_norm', nargs='?', const=1, type=float,
                    default=40.0)
parser.add_argument('--freeze_bn', dest='eval_vs_train', action='store_true')
parser.set_defaults(eval_vs_train=False)

###############################################################################
#                                 GNN Settings                                #
###############################################################################

parser.add_argument('--num_features', nargs='?', const=1, type=int,
                    default=20)
parser.add_argument('--num_layers', nargs='?', const=1, type=int,
                    default=100)
parser.add_argument('--n_classes', nargs='?', const=1, type=int,
                    default=2)
parser.add_argument('--J', nargs='?', const=1, type=int, default=4)
parser.add_argument('--N_train', nargs='?', const=1, type=int, default=1000)
parser.add_argument('--N_test', nargs='?', const=1, type=int, default=1000)
parser.add_argument('--lr', nargs='?', const=1, type=float, default=1e-3)

args = parser.parse_args()

if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
    dtype_l = torch.cuda.LongTensor
    # torch.cuda.manual_seed(0)
else:
    dtype = torch.FloatTensor
    dtype_l = torch.LongTensor
    # torch.manual_seed(1)

batch_size = args.batch_size
criterion = nn.CrossEntropyLoss()
template1 = '{:<10} {:<10} {:<10} {:<15} {:<10} {:<10} {:<10} '
template2 = '{:<10} {:<10.5f} {:<10.5f} {:<15} {:<10} {:<10} {:<10.3f} \n'
template3 = '{:<10} {:<10} {:<10} '
template4 = '{:<10} {:<10.5f} {:<10.5f} \n'

# def train_single(gnn, optimizer, gen, n_classes, it):
#     start = time.time()
#     W, labels = gen.sample_otf_single(is_training=True, cuda=torch.cuda.is_available())
#     labels = labels.type(dtype_l)
#
#     if (args.generative_model == 'SBM_multiclass') and (args.n_classes == 2):
#         labels = (labels + 1)/2
#
#     WW, x = get_gnn_inputs(W, args.J)
#
#     if (torch.cuda.is_available()):
#         WW.cuda()
#         x.cuda()
#
#     pred = gnn(WW.type(dtype), x.type(dtype))
#     penultimate_features = gnn.get_penultimate_output()
#     print('penultimate_features', penultimate_features)
#
#     loss = compute_loss_multiclass(pred, labels, n_classes)
#     gnn.zero_grad()
#     loss.backward()
#     nn.utils.clip_grad_norm_(gnn.parameters(), args.clip_grad_norm)
#     optimizer.step()
#
#     acc = compute_accuracy_multiclass(pred, labels, n_classes)
#
#     elapsed = time.time() - start
#
#     if(torch.cuda.is_available()):
#         loss_value = float(loss.data.cpu().numpy())
#     else:
#         loss_value = float(loss.data.numpy())
#
#     info = ['iter', 'avg loss', 'avg acc', 'edge_density',
#             'noise', 'model', 'elapsed']
#     out = [it, loss_value, acc, args.edge_density,
#            args.noise, 'GNN', elapsed]
#     print(template1.format(*info))
#     print(template2.format(*out))
#
#     del WW
#     del x
#
#     return loss_value, acc
def train_single(gnn, optimizer, gen, n_classes, it):
    start = time.time()
    W, labels = gen.sample_otf_single(is_training=True, cuda=torch.cuda.is_available())

    # 确保 labels 是 Tensor
    labels = labels.type(dtype_l)

    # if (args.generative_model == 'SBM_multiclass') and (args.n_classes == 2):
    #     labels = (labels + 1) / 2

    WW, x = get_gnn_inputs(W, args.J)

    if torch.cuda.is_available():
        WW = WW.cuda()
        x = x.cuda()
    print(labels.shape)
    gnn.zero_grad(set_to_none=True)
    # --- 2. GNN 前向传播 ---
    pred = gnn(WW.type(dtype), x.type(dtype))
    # --- 4. 计算损失并反向传播 ---
    loss = compute_modularity_loss_multiclass(pred,W)
    # loss = compute_loss_multiclass(pred, labels, n_classes)
    loss.backward()
    nn.utils.clip_grad_norm_(gnn.parameters(), args.clip_grad_norm)
    optimizer.step()

    acc, best_matched_pred = compute_accuracy_multiclass(pred, labels, n_classes)
    elapsed = time.time() - start

    loss_value = float(loss.data.cpu().numpy()) if torch.cuda.is_available() else float(loss.data.numpy())


    info = ['iter', 'avg loss', 'avg acc', 'edge_density', 'noise', 'model', 'elapsed']
    out = [it, loss_value, acc, args.edge_density, args.noise, 'GNN', elapsed]

    # print(out)
    print(template1.format(*info))
    print(template2.format(*out))

    del WW, x
    return loss_value, acc

def train(gnn, gen, n_classes=args.n_classes, iters=args.num_examples_train):
    gnn.train()
    optimizer = torch.optim.Adamax(gnn.parameters(), lr=args.lr)
    loss_lst = np.zeros([iters])
    acc_lst = np.zeros([iters])
    for it in range(iters):
        loss_single, acc_single = train_single(gnn, optimizer, gen, n_classes, it)
        loss_lst[it] = loss_single
        acc_lst[it] = acc_single
        torch.cuda.empty_cache()
    print ('Avg train loss', np.mean(loss_lst))
    print ('Avg train acc', np.mean(acc_lst))
    print ('Std train acc', np.std(acc_lst))

def test_single(gnn, gen, n_classes, it):
    start = time.time()

    # --- 1. 生成数据 ---
    W, labels = gen.sample_otf_single(is_training=False, cuda=torch.cuda.is_available())

    # --- 2. 处理 batch 维度 ---
    # W: (1, 1000, 1000) -> (1000, 1000)（用于特征计算）
    W_np = W.squeeze(0).cpu().numpy() if isinstance(W, Tensor) else W.squeeze(0)

    # labels: 保持 (1, 1000)（因为 compute_loss_multiclass 可能需要 batch 维度）
    labels = labels.type(torch.long)  # (1, 1000)

    # --- 3. 计算邻接矩阵特征向量 ---
    W_for_eig = (W_np + W_np.T) / 2  # 确保对称
    eigvals_W, eigvecs_W = np.linalg.eigh(W_for_eig)
    eigvals_W, eigvecs_W = np.real(eigvals_W), np.real(eigvecs_W)
    adjacency_eigvecs = eigvecs_W[:, np.argsort(eigvals_W)[-2:]]  # (1000, 2)
    adjacency_eigvecs /= np.linalg.norm(adjacency_eigvecs, axis=0, keepdims=True)

    # --- 4. 计算拉普拉斯矩阵特征向量 ---
    D_np = np.diag(np.sum(W_for_eig, axis=1))
    L_np = D_np - W_for_eig
    eigvals_L, eigvecs_L = np.linalg.eigh(L_np)
    eigvals_L, eigvecs_L = np.real(eigvals_L), np.real(eigvecs_L)
    laplacian_eigvecs = eigvecs_L[:, np.argsort(eigvals_L)[:2]]  # (1000, 2)
    laplacian_eigvecs /= np.linalg.norm(laplacian_eigvecs, axis=0, keepdims=True)

    # --- 5. GNN 前向传播 ---
    WW, x = get_gnn_inputs(W, args.J)  # W 保持 (1, 1000, 1000)
    if torch.cuda.is_available():
        WW, x = WW.cuda(), x.cuda()

    pred = gnn(WW.float(), x.float())  # pred: (1, 1000, n_classes)
    penultimate_features = gnn.get_penultimate_output().detach().cpu().numpy()  # (1, 1000, d)
    penultimate_features = penultimate_features.squeeze(0)  # (1000, d)
    penultimate_features /= np.linalg.norm(penultimate_features, axis=0, keepdims=True)
    output_filename = "all_test_results_p=0.1_q=0.05.xlsx"

    # --- 6. 保存结果 ---
    df = pd.DataFrame({
        'GNN_Feature1': penultimate_features[:, 0],
        'GNN_Feature2': penultimate_features[:, 1],
        'Adj_EigVec1': adjacency_eigvecs[:, 0],
        'Adj_EigVec2': adjacency_eigvecs[:, 1],
        'Lap_EigVec1': laplacian_eigvecs[:, 0],
        'Lap_EigVec2': laplacian_eigvecs[:, 1],
        'Labels': labels.squeeze(0).cpu().numpy()  # (1000,)
    })

    if it == 0:
        df.to_excel(output_filename, sheet_name=f'Iteration_{it}', index=False)
    else:
        with pd.ExcelWriter(output_filename, mode='a', engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name=f'Iteration_{it}', index=False)

    # --- 7. 计算损失和准确率 ---
    loss = compute_loss_multiclass(pred, labels, n_classes)  # labels: (1, 1000)
    acc = compute_accuracy_multiclass(pred, labels, n_classes)

    # --- 8. 打印日志 ---
    print(f"iter {it}: loss={loss.item():.4f}, acc={acc:.2f}%, time={time.time() - start:.2f}s")
    return loss.item(), acc
def test(gnn, gen, n_classes, iters=args.num_examples_test):
    gnn.train()
    loss_lst = np.zeros([iters])
    acc_lst = np.zeros([iters])
    for it in range(iters):
        loss_single, acc_single = test_single(gnn, gen, n_classes, it)
        loss_lst[it] = loss_single
        acc_lst[it] = acc_single
        torch.cuda.empty_cache()
    print ('Avg test loss', np.mean(loss_lst))
    print ('Avg test acc', np.mean(acc_lst))
    print ('Std test acc', np.std(acc_lst))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



if __name__ == '__main__':

    gen = Generator()
    gen.N_train = args.N_train
    gen.N_test = args.N_test
    gen.edge_density = args.edge_density
    gen.p_SBM = args.p_SBM
    gen.q_SBM = args.q_SBM
    gen.random_noise = args.random_noise
    gen.noise = args.noise
    gen.noise_model = args.noise_model
    gen.generative_model = args.generative_model
    gen.n_classes = args.n_classes


    torch.backends.cudnn.enabled=False

    if (args.mode == 'test'):
        print ('In testing mode')
        filename = "gnn_J4_lyr20_Ntr1000_num2000"
        path_plus_name = os.path.join(args.path_gnn, filename)
        if ((filename != '') and (os.path.exists(path_plus_name))):
            print ('Loading gnn ' + filename)
            gnn = torch.load(path_plus_name)
            if torch.cuda.is_available():
                gnn.cuda()
        else:
            print ('No such a gnn exists; creating a brand new one')
            if (args.generative_model == 'SBM_multiclass'):
                gnn = GNN_multiclass(args.num_features, args.num_layers, args.J + 2, n_classes=args.n_classes)
            filename = 'gnn_J' + str(args.J) + '_lyr' + str(args.num_layers) + '_Ntr' + str(args.N_train) + '_num' + str(args.num_examples_train)
            path_plus_name = os.path.join(args.path_gnn, filename)
            if torch.cuda.is_available():
                gnn.cuda()
            print ('Training begins')


    elif (args.mode == 'train'):
        filename = args.filename_existing_gnn
        path_plus_name = os.path.join(args.path_gnn, filename)
        if ((filename != '') and (os.path.exists(path_plus_name))):
            print ('Loading gnn ' + filename)
            gnn = torch.load(path_plus_name)
            filename = filename + '_Ntr' + str(args.N_train) + '_num' + str(args.num_examples_train) + '_p= ' + str(args.p_SBM)  + '_q= ' + str(args.q_SBM)
            path_plus_name = os.path.join(args.path_gnn, filename)
        else:
            print ('No such a gnn exists; creating a brand new one')
            filename = 'gnn_J' + str(args.J) + '_lyr' + str(args.num_layers) + '_Ntr' + str(args.N_train) + '_num' + str(args.num_examples_train)+ '_p= ' + str(args.p_SBM)  + '_q= ' + str(args.q_SBM)
            path_plus_name = os.path.join(args.path_gnn, filename)
            if (args.generative_model == 'SBM_multiclass'):
                gnn = GNN_multiclass(args.num_features, args.num_layers, args.J + 3, n_classes=args.n_classes)

        print ('total num of params:', count_parameters(gnn))

        if torch.cuda.is_available():
            gnn.cuda()
        print ('Training begins')
        if (args.generative_model == 'SBM_multiclass'):
            train(gnn, gen, args.n_classes)
        print ('Saving gnn ' + filename)
        if torch.cuda.is_available():
            torch.save(gnn.cpu(), path_plus_name)
            gnn.cuda()
        else:
            torch.save(gnn, path_plus_name)


    print ('Testing the GNN:')
    if args.eval_vs_train:
        print ('model status: eval')
        gnn.eval()
    else:
        print ('model status: train')
        gnn.train()

    test(gnn, gen, args.n_classes)

    print ('total num of params:', count_parameters(gnn))


