import setGPU
import torch
from torch_geometric.data import Data, DataLoader
from graph_data import GraphDataset
from models import EdgeNet
import os
import os.path as osp
import math
import numpy as np
import tqdm
import argparse
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('using device %s'%device)


def get_model_fname(model):
    model_name = type(model).__name__
    model_params = sum(p.numel() for p in model.parameters())
    import hashlib
    model_cfghash = hashlib.blake2b(repr(model).encode()).hexdigest()[:10]
    model_user = os.environ['USER']
    
    model_fname = '%s_%d_%s_%s'%(model_name,model_params,
                                 model_cfghash, model_user)
    return model_fname

@torch.no_grad()
def test(model,loader,total,batch_size):
    model.eval()
    correct = 0

    sum_loss = 0
    sum_correct = 0
    sum_truepos = 0
    sum_trueneg = 0
    sum_falsepos = 0
    sum_falseneg = 0
    sum_true = 0
    sum_false = 0
    sum_total = 0
    t = tqdm.tqdm(enumerate(loader),total=total/batch_size)
    for i,data in t:
        data = data.to(device)
        batch_target = data.y
        batch_output = model(data)
        batch_loss_item = F.binary_cross_entropy(batch_output, batch_target).item()
        sum_loss += batch_loss_item
        matches = ((batch_output > 0.5) == (batch_target > 0.5))
        true_pos = ((batch_output > 0.5) & (batch_target > 0.5))
        true_neg = ((batch_output < 0.5) & (batch_target < 0.5))
        false_pos = ((batch_output > 0.5) & (batch_target < 0.5))
        false_neg = ((batch_output < 0.5) & (batch_target > 0.5))
        sum_truepos += true_pos.sum().item()
        sum_trueneg += true_neg.sum().item()
        sum_falsepos += false_pos.sum().item()
        sum_falseneg += false_neg.sum().item()
        sum_correct += matches.sum().item()
        sum_true += batch_target.sum().item()
        sum_false += (batch_target < 0.5).sum().item()
        sum_total += matches.numel()
        t.set_description("batch loss = %.5f" % (batch_loss_item))
        t.refresh() # to show immediately the update


    print('scor', sum_correct,
          'stru', sum_true,
          'stp', sum_truepos,
          'stn', sum_trueneg,
          'sfp', sum_falsepos,
          'sfn', sum_falseneg,
          'stot', sum_total)
    return sum_loss/(i+1), sum_correct / sum_total, sum_truepos/sum_true, sum_falsepos / sum_false, sum_falseneg / sum_true, sum_truepos/(sum_truepos+sum_falsepos + 1e-6)

def train(model, optimizer, epoch, loader, total, batch_size):
    model.train()
    model_fname = get_model_fname(model)

    sum_loss = 0.
    t = tqdm.tqdm(enumerate(loader),total=total/batch_size)
    for i,data in t:
        data = data.to(device)
        optimizer.zero_grad()
        batch_target = data.y        
        batch_output = model(data)
        batch_loss = F.binary_cross_entropy(batch_output, batch_target)
        batch_loss.backward()
        batch_loss_item = batch_loss.item()
        t.set_description("batch loss = %.5f" % batch_loss_item)
        t.refresh() # to show immediately the update
        sum_loss += batch_loss_item
        optimizer.step()
    # to save every epoch
    #modpath = osp.join(os.getcwd(),model_fname+'.%d.pth'%epoch)
    #torch.save(model.state_dict(),modpath)
    
    return sum_loss/(i+1)


def main(args): 

    full_dataset = GraphDataset(root='data/')
    
    data = full_dataset.get(0)
    input_dim = data.x.shape[1]
    fulllen = len(full_dataset)
    tv_frac = 0.10
    tv_num = math.ceil(fulllen*tv_frac)
    splits = np.cumsum([fulllen-2*tv_num,tv_num,tv_num])
    batch_size = 128
    n_epochs = 100
    lr = 0.01
    patience = 10
    hidden_dim = 32
    n_iters = 1

    train_dataset = torch.utils.data.Subset(full_dataset,np.arange(start=0,stop=splits[0]))
    valid_dataset = torch.utils.data.Subset(full_dataset,np.arange(start=splits[1],stop=splits[2]))
    test_dataset = torch.utils.data.Subset(full_dataset,np.arange(start=splits[0],stop=splits[1]))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, pin_memory=True, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, pin_memory=True, shuffle=False)

    train_samples = len(train_dataset)
    valid_samples = len(valid_dataset)
    test_samples = len(test_dataset)
    
    model = EdgeNet(input_dim=input_dim,hidden_dim=hidden_dim,n_iters=n_iters).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    model_fname = get_model_fname(model)

    best_valid_loss = 99999
    print('Training with %s samples'%train_samples)
    print('Validating with %s samples'%valid_samples)

    stale_epochs = 0
    for epoch in range(0, n_epochs):
        epoch_loss = train(model, optimizer, epoch, train_loader, train_samples, batch_size)
        valid_loss, valid_acc, valid_eff, valid_fp, valid_fn, valid_pur = test(model, valid_loader, valid_samples, batch_size)
        print('Epoch: {:02d}, Training Loss: {:.4f}'.format(epoch, epoch_loss))
        print('               Validation Loss: {:.4f}, Acc.: {:.4f}, Eff.: {:.4f}, FalsePos: {:.4f}, FalseNeg: {:.4f}, Purity: {:,.4f}'.format(valid_loss, 
                                                                                                                                               valid_acc,
                                                                                                                                               valid_eff,
                                                                                                                                               valid_fp,
                                                                                                                                               valid_fn,
                                                                                                                                               valid_pur))

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            modpath = osp.join(os.getcwd(),model_fname+'.best.pth')
            print('New best model saved to:',modpath)
            torch.save(model.state_dict(),modpath)
            stale_epochs = 0
        else:
            print('Stale epoch')
            stale_epochs += 1
        if stale_epochs >= patience:
            print('Early stopping after %i stale epochs'%patience)
            break

    modpath = osp.join(os.getcwd(),model_fname+'.final.pth')
    print('Final model saved to:',modpath)
    torch.save(model.state_dict(),modpath)

    modpath = osp.join(os.getcwd(),model_fname+'.best.pth')
    model.load_state_dict(torch.load(modpath))
    y_test, predict_test = [], []
    for i,data in enumerate(test_loader):
        y_test.append(data.y.cpu().detach().numpy())
        data = data.to(device)
        predict_test.append(model(data).cpu().detach().numpy())
    y_test = np.concatenate(y_test,axis=0)
    predict_test = np.concatenate(predict_test,axis=0)

    plt.figure()       
    fpr, tpr, threshold = roc_curve(y_test,predict_test)
    auc_test = auc(fpr, tpr)
    
    plt.plot(tpr,fpr,label='LLP muon tagger (GNN), AUC = %.1f%%'%(auc_test*100.))
    plt.semilogy()
    plt.xlabel("Signal Efficiency")
    plt.ylabel("Background Efficiency")
    plt.ylim(0.001,1)
    plt.grid(True)
    plt.legend(loc='upper left')
    plt.figtext(0.25, 0.90,'CMS',fontweight='bold', wrap=True, horizontalalignment='right', fontsize=14)
    plt.figtext(0.35, 0.90,'preliminary', style='italic', wrap=True, horizontalalignment='center', fontsize=14) 
    plt.savefig('ROC_gnn.pdf')
  
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
        
    args = parser.parse_args()
    main(args)
