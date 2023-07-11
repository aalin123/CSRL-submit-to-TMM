import os
import time
import argparse
import warnings
import random


import torch
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter

from Loss import* #HAFN, SAFN
from Utils import *

parser = argparse.ArgumentParser(description='Expression Classification Training')

parser.add_argument('--Log_Name', type=str, default='batch_64_FonlyMMD_Dual_ZL_CD_PES_RAF_CK+_ST_4_05_CT_4_05_MMD_0_005_CrossTrans_RES18_95_',help='Log Name')
parser.add_argument('--OutputPath', type=str, default='trainmodel',help='Output Path')
parser.add_argument('--Backbone', type=str, default='ResNet18', choices=['ResNet18', 'ResNet50', 'VGGNet', 'MobileNet'])
parser.add_argument('--Resume_Model', type=str, help='Resume_Model', default='ms1m_res18.pkl')
parser.add_argument('--Resume_Model_all', type=str, help='Resume_Model_all', default='None') 
parser.add_argument('--GPU_ID', default='0', type=str, help='CUDA_VISIBLE_DEVICES')

parser.add_argument('--faceScale', type=int, default=224, help='Scale of face (default: 112)')
parser.add_argument('--sourceDataset', type=str, default='RAF', choices=['RAF', 'AFED'])
parser.add_argument('--targetDataset', type=str, default='CK+', choices=['RAF', 'CK+', 'CK_all', 'JAFFE', 'MMI', 'Oulu-CASIA', 'SFEW', 'FER2013', 'ExpW', 'AFED', 'WFED'])
parser.add_argument('--train_batch_size', type=int, default=64, help='input batch size for training (default: 64)')
parser.add_argument('--test_batch_size', type=int, default=64, help='input batch size for testing (default: 64)')


parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--epochs', type=int, default=30,help='number of epochs to train (default: 10)')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum (default: 0.5)') #0.9
parser.add_argument('--weight_decay', type=float, default=0.00000,help='SGD weight decay (default: 0.0005)') #0.00005

parser.add_argument('--isTest', type=str2bool, default=False, help='whether to test model')
parser.add_argument('--showFeature', type=str2bool, default=False, help='whether to show feature')


parser.add_argument('--class_num', type=int, default=7, help='number of class (default: 7)')
parser.add_argument('--seed', type=int, default=2019, help='random seed (default: 1)')
parser.add_argument('--Num_Workers', default=12, type=int, help='Number of Workers')
parser.add_argument('--DataParallel', default=False, type=str2bool, help='Data Parallel')

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
     

def Train(args, model, train_source_dataloader, train_target_dataloader, optimizer, epoch, writer):
    """Train."""

    model.train()
    #torch.autograd.set_detect_anomaly(True)

    acc, prec, recall = [AverageMeter() for i in range(7)], [AverageMeter() for i in range(7)], [AverageMeter() for i in range(7)]
    loss, global_cls_loss, data_time, batch_time =  AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()
   
    
     # Get Source/Target Dataloader iterator
    iter_source_dataloader = iter(train_source_dataloader)
    iter_target_dataloader = iter(train_target_dataloader)

   
    num_iter = len(train_source_dataloader) if (len(train_source_dataloader) > len(train_target_dataloader)) else len(train_target_dataloader)

    #model.train()
    optimizer, lr = Adjust_Learning_Rate(optimizer, epoch, args.lr)
    
    end = time.time()


    if epoch<=15:
        for batch_index in range(num_iter):
            try:
                data_source, label_source = iter_source_dataloader.next()
            except:
                iter_source_dataloader = iter(train_source_dataloader)
                data_source, label_source = iter_source_dataloader.next()

            try:
                data_target, label_target = iter_target_dataloader.next()
            except:
                iter_target_dataloader = iter(train_target_dataloader)
                data_target, label_target = iter_target_dataloader.next()
        

            data_time.update(time.time()-end)
            data_source,index_s,ID_s=data_source
            data_target,index_t,ID_t=data_target
            data_source,  label_source = data_source.cuda(),  label_source.cuda()
            data_target,  label_target = data_target.cuda(),  label_target.cuda()

            # Forward propagation
            end = time.time()
        
            Trag=False
            output,S_feature,T_feature = model(data_source,data_target,Trag)
            #output=S_output
            batch_time.update(time.time()-end)
            
            label_source1=torch.cat((label_source,label_source),0)
            label_source2=torch.cat((label_source1,label_source),0)
            # Compute Loss
            global_cls_loss_ = nn.CrossEntropyLoss()(output, label_source2) 
            C_MMDloss_=mmd_rbf_noaccelerate(S_feature, T_feature) #
            loss_ = global_cls_loss_  + 0.005*C_MMDloss_ 

            # Back Propagation
            optimizer.zero_grad()
            
            #with torch.autograd.detect_anomaly():
            loss_.backward()

            optimizer.step()

            # Decay Learn Rate
            #optimizer, lr = lr_scheduler_withoutDecay(optimizer, lr=args.lr, weight_decay=args.weight_decay) 

            # Compute accuracy, recall and loss
            Compute_Accuracy(args,output.narrow(0, 0, data_source.size(0)), label_source, acc, prec, recall)

            # Log loss
            loss.update(float(loss_.cpu().data.item()))
            global_cls_loss.update(float(global_cls_loss_.cpu().data.item()))
        
            end = time.time()


    elif epoch>15:
        global na
        if epoch%15==1: #%15
            name=ZL_GetIndexFromDataset(args, epoch, model, train_target_dataloader)
            na=name
        
        pes_train_target_dataloader= Get_Pselabel_dataloader(args,na)
        iter_pes_target_dataloader = iter(pes_train_target_dataloader)

        num_iter = len(train_source_dataloader) if (len(train_source_dataloader) > len(pes_train_target_dataloader)) else len(pes_train_target_dataloader)
        for batch_index in range(num_iter):
        
            try:
                data_source, label_source = iter_source_dataloader.next()
            except:
                iter_source_dataloader = iter(train_source_dataloader)
                data_source, label_source = iter_source_dataloader.next()

            try:
                data_target, label_target = iter_pes_target_dataloader.next()
            except:
                iter_pes_target_dataloader = iter(pes_train_target_dataloader)
                data_target, label_target = iter_pes_target_dataloader.next()
        

            data_time.update(time.time()-end)
            data_source,index_s,ID_s=data_source
            data_target,index_t,ID_t=data_target
            data_source,  label_source = data_source.cuda(),  label_source.cuda()
            data_target,  label_target = data_target.cuda(),  label_target.cuda()

            # Forward propagation
            end = time.time()
        
            Trag=True
            output,S_feature,T_feature = model(data_source,data_target,Trag)
            #output=S_output
            batch_time.update(time.time()-end)
            
            label_source1=torch.cat((label_source,label_source),0)
            label_source2=torch.cat((label_source1,label_source),0)
            label_source3=torch.cat((label_source2,label_target),0)
            label_target1=torch.cat((label_target,label_target),0)
            label_all=torch.cat((label_source3,label_target1),0)
            label_cc=torch.cat((label_source,label_target),0)
            # Compute Loss
            global_cls_loss_ = nn.CrossEntropyLoss()(output, label_all) 
            C_MMDloss_=mmd_rbf_noaccelerate(S_feature, T_feature) #
            loss_ = global_cls_loss_  + 0.005*C_MMDloss_ 

            # Back Propagation
            optimizer.zero_grad()
            
            #with torch.autograd.detect_anomaly():
            loss_.backward()

            optimizer.step()

            # Decay Learn Rate
            #optimizer, lr = lr_scheduler_withoutDecay(optimizer, lr=args.lr, weight_decay=args.weight_decay) 

            # Compute accuracy, recall and loss
            Compute_Accuracy(args,output.narrow(0, 0, data_source.size(0)), label_source, acc, prec, recall)

            # Log loss
            loss.update(float(loss_.cpu().data.item()))
            global_cls_loss.update(float(global_cls_loss_.cpu().data.item()))
        
            end = time.time()





    AccuracyInfo, acc_avg, prec_avg, recall_avg, f1_avg = Show_Accuracy(acc, prec, recall, args.class_num)

    writer.add_scalar('Accuracy', acc_avg, epoch)
    writer.add_scalar('Precision', prec_avg, epoch)
    writer.add_scalar('Recall', recall_avg, epoch)
    writer.add_scalar('F1', f1_avg, epoch)

    writer.add_scalar('Global_Cls_Loss', global_cls_loss.avg, epoch)
    #writer.add_scalar('Local_Cls_Loss', local_cls_loss.avg, epoch)
    #writer.add_scalar('AFN_Loss', afn_loss.avg, epoch)

    LoggerInfo = '''
    [Tain]: 
    Epoch {0}
    Data Time {data_time.sum:.4f} ({data_time.avg:.4f})
    Batch Time {batch_time.sum:.4f} ({batch_time.avg:.4f})
    Learning Rate {1}\n'''.format(epoch, lr, data_time=data_time, batch_time=batch_time)

    LoggerInfo+=AccuracyInfo

    LoggerInfo+='''    Acc_avg {0:.4f} Prec_avg {1:.4f} Recall_avg {2:.4f} F1_avg {3:.4f}
    Total Loss {loss:.4f} Global Cls Loss {global_cls_loss:.4f}'''.format(acc_avg, prec_avg, recall_avg, f1_avg,
                                                                        loss=loss.avg, 
                                                                                                                                        global_cls_loss=global_cls_loss.avg, 
                                                                                                                                        )

    print(LoggerInfo)

def Test(args, model, test_source_dataloader,Best_acc):
    """Test."""

    model.eval()
    torch.autograd.set_detect_anomaly(True)

    iter_source_dataloader = iter(test_source_dataloader)
    #iter_target_dataloader = iter(test_target_dataloader)

    # Test on Source Domain
    acc, prec, recall = [AverageMeter() for i in range(7)], [AverageMeter() for i in range(7)], [AverageMeter() for i in range(7)]
    loss, global_cls_loss, data_time, batch_time =  AverageMeter(), AverageMeter(), AverageMeter(), AverageMeter()

    end = time.time()
    for step, (input,  label) in enumerate(iter_source_dataloader):
        input,index,ID=input
        input,  label = input.cuda(),  label.cuda()
        data_time.update(time.time()-end)
        Trag=False
        # Forward Propagation
        with torch.no_grad():
            end = time.time()
            output,fe1,fe2 = model(input,input,Trag)
            batch_time.update(time.time()-end)
        
        # Compute Loss
        global_cls_loss_ = nn.CrossEntropyLoss()(output, label)
        loss_ = global_cls_loss_ 

        # Compute accuracy, precision and recall
        Compute_Accuracy(args, output, label, acc, prec, recall)

        # Log loss
        loss.update(float(loss_.cpu().data.item()))
        global_cls_loss.update(float(global_cls_loss_.cpu().data.item()))
        
        end = time.time()

    AccuracyInfo, acc_avg, prec_avg, recall_avg, f1_avg = Show_Accuracy(acc, prec, recall, args.class_num)

    LoggerInfo = '''
    [Test (Target Domain)]: 
    Data Time {data_time.sum:.4f} ({data_time.avg:.4f})
    Batch Time {batch_time.sum:.4f} ({batch_time.avg:.4f})
    Learning Rate {0}\n'''.format(args.lr, data_time=data_time, batch_time=batch_time)
    #字符串，三个单引号表示回车

    LoggerInfo+=AccuracyInfo 

    LoggerInfo+='''    Acc_avg {0:.4f} Prec_avg {1:.4f} Recall_avg {2:.4f} F1_avg {3:.4f}
    Total Loss {loss:.4f} Global Cls Loss {global_cls_loss:.4f}'''.format(acc_avg, prec_avg, recall_avg, f1_avg,\
                                                                            loss=loss.avg, 
                                                                            global_cls_loss=global_cls_loss.avg,
                                                                            )

    print(LoggerInfo)

    # Save Checkpoints
    if acc_avg > Best_acc:
        Best_acc = acc_avg
        print('[Save] Best ACC: %.4f.' % Best_acc)
        #print('[Save] Best Recall: %.4f.' % Best_Recall)

        if isinstance(model, nn.DataParallel):
            torch.save(model.module.state_dict(), os.path.join(args.OutputPath, '{}.pkl'.format(args.Log_Name)))
        else:
            torch.save(model.state_dict(), os.path.join(args.OutputPath, '{}.pkl'.format(args.Log_Name)))

    # Test on Target Domain

    return Best_acc

def main():
    """Main."""
 
    # Parse Argument
    args = parser.parse_args() #调用上面所有属性
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_ID
    setup_seed(args.seed)
    
    # Experiment Information
    print('Log Name: %s' % args.Log_Name)
    print('Output Path: %s' % args.OutputPath)
    print('Backbone: %s' % args.Backbone)
    print('Resume Model: %s' % args.Resume_Model)
    print('CUDA_VISIBLE_DEVICES: %s' % args.GPU_ID)

    print('================================================')

    print('Use {} * {} Image'.format(args.faceScale, args.faceScale))
    print('SourceDataset: %s' % args.sourceDataset)
    print('TargetDataset: %s' % args.targetDataset)
    print('Train Batch Size: %d' % args.train_batch_size)
    print('Test Batch Size: %d' % args.test_batch_size)

    print('================================================')

    if args.showFeature:
        print('Show Visualiza Result of Feature.')

    if args.isTest:
        print('Test Model.')
    else:
        print('Train Epoch: %d' % args.epochs)
        print('Learning Rate: %f' % args.lr)
        print('Momentum: %f' % args.momentum)
        print('Weight Decay: %f' % args.weight_decay)

    print('================================================')

    print('Number of classes : %d' % args.class_num)

    print('================================================')

    # Bulid Dataloder
    print("Building Train and Test Dataloader...")
    train_source_dataloader = BulidDataloader(args, flag1='train', flag2='source')
    #train_target_dataloader = BulidDataloader(args, flag1='test', flag2='target')
    train_target_dataloader = BulidDataloader(args, flag1='train', flag2='target')
    test_source_dataloader = BulidDataloader(args, flag1='test', flag2='source')
    test_target_dataloader = BulidDataloader(args, flag1='test', flag2='target')
    print('Done!')

    print('================================================')

    # Bulid Model
    print('Building Model...')
    model = Bulid_Model(args)
    print('Done!')

    print('================================================')


    # Set Optimizer
    print('Building Optimizer...')
    param_optim = Set_Param_Optim(model)
    optimizer = Set_Criterion_Optimizer(args, param_optim)
    print('Done!')

    print('================================================')
  
    # Save Best Checkpoint
    Best_acc = 0
    Best_acc1 = 0
    # Running Experiment
    print("Run Experiment...")
    writer = SummaryWriter(os.path.join(args.OutputPath, args.Log_Name))

    for epoch in range(1, args.epochs + 1):

        Train(args, model, train_source_dataloader, train_target_dataloader, optimizer, epoch,  writer)
        Best_acc1 = Test(args, model, test_target_dataloader,  Best_acc1)

        torch.cuda.empty_cache()

    print('Best_target_Accuracy %.4f' % Best_acc1)

    writer.close()

if __name__ == '__main__':
    main()
