import torch
import argparse, threading, time
import numpy as np
from tqdm import tqdm
from model import *
from utils import *
from tensorboardX import SummaryWriter
import os  
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'  # 可以尝试不同的值

# Constants
MODEL_DIR = 'saved_models'
NUM_EPOCHS = 1000

# 输出有俩，当前帧的预测掩码和当前帧与前一帧掩码的编码结果，用于传入下一帧预测
def Propagate_MS(ms, model, F2, P2):
    h, w = F2.size()[2], F2.size()[3]
    
    msv_F2, msv_P2 = ToCudaVariable([F2, P2])
    if hasattr(torch.cuda, 'empty_cache'):
        torch.cuda.empty_cache()
    r5, r4, r3, r2  = model.Encoder(msv_F2, msv_P2)
    e2 = model.Decoder(r5, ms, r4, r3, r2)

    return F.softmax(e2[0], dim=1)[:,1], r5

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='RGMP')
    parser.add_argument('--epochs', dest='num_epochs',
                      help='number of epochs to train',
                      default=NUM_EPOCHS, type=int)
    parser.add_argument('--bs', dest='bs',
                      help='batch_size',
                      default=1, type=int)
    parser.add_argument('--num_workers', dest='num_workers',
                      help='num_workers',
                      default=1, type=int)
    parser.add_argument('--disp_interval', dest='disp_interval',
                      help='display interval',
                      default=10, type=int)
    parser.add_argument('--eval_epoch', dest='eval_epoch',
                      help='interval of epochs to perform validation',
                      default=10, type=int)
    parser.add_argument('--output_dir', dest='output_dir',
                      help='output directory',
                      default=MODEL_DIR, type=str)
    
    # BPTT
    parser.add_argument('--bptt', dest='bptt_len',
                      help='length of BPTT',
                      default=12, type=int)
    parser.add_argument('--bptt_step', dest='bptt_step',
                      help='step of truncated BPTT',
                      default=4, type=int)
    
    # config optimization
    parser.add_argument('--o', dest='optimizer',
                      help='training optimizer',
                      default="sgd", type=str)
    parser.add_argument('--lr', dest='lr',
                      help='starting learning rate',
                      default=1e-3, type=float)
    parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                      help='step to do learning rate decay, unit is epoch',
                      default=5, type=int)
    parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                      help='learning rate decay ratio',
                      default=0.1, type=float)

    # resume trained model
    parser.add_argument('--loadepoch', dest='loadepoch',
                      help='epoch to load model',
                      default=-1, type=int)
    
    args = parser.parse_args()
    return args    
def log_mask(frames, masks, info, writer, F_name='Train/frames', M_name='Train/masks'):
    print('[tensorboard] Updating mask..')
    _,C,T,H,W = masks.size()
#     (lh,uh), (lw,uw) = info['pad'] 
    vid = np.zeros((1,3,T,H,W))
    print_dbg('[mask] mean: {}, max: {}'.format(torch.mean(masks[:,:,1]), torch.max(masks[:,:,1])))
    masks = torch.cat((1-masks, masks), dim=1)
    for f in range(T):
        E = masks[0,:,f].cpu().data.numpy()
        E = ToLabel(E) # E.shape = [H, W], 每个像素点的值表示该点所属的类别, 最后将输出转化为uint8格式
        img_E = Image.fromarray(E)
        img_E.putpalette(PALETTE)
        arr_E = np.array(E)
        vid[0,:,f,:,:] = np.array(img_E)  
    vid_tensor = torch.FloatTensor(vid)
    writer.add_video(F_name, vid_tensor=frames)
    writer.add_video(M_name, vid_tensor=vid_tensor)
    print('[tensorboard] Mask updated')     
if __name__ == '__main__':
    args = parse_args()
    Trainset = DAVIS(DAVIS_ROOT, imset='2016/train.txt')
    Trainloader = data.DataLoader(Trainset, batch_size=1, shuffle=True, num_workers=2)
    Testset = DAVIS(DAVIS_ROOT, imset='2016/val.txt')
    Testloader = data.DataLoader(Testset, batch_size=1, shuffle=True, num_workers=2)
    model = RGMP()
    devive = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device=devive)
    # if torch.cuda.is_available():
    #     model.cuda()
    writer = SummaryWriter()
    start_epoch = 0
    params = []
    for key, value in dict(model.named_parameters()).items():
      if value.requires_grad:
        params += [{'params':[value],'lr':args.lr, 'weight_decay': 4e-5}]
    criterion = torch.nn.BCELoss()
    criterion.to(device=devive)
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9) 
    iters_per_epoch = len(Trainloader)
    for epoch in range(start_epoch, args.num_epochs):        
        if epoch % args.eval_epoch == 1:
            with torch.no_grad():
                print('[Val] Epoch {}{}{}'.format(font.BOLD, epoch, font.END))
                model.eval()
                loss = 0
                iOU = 0
                pbar = tqdm.tqdm(total=len(Testloader))
                for i, (all_F, all_M, info) in enumerate(Testloader):
                    pbar.update(1)
                    all_F = all_F.to(devive)
                    all_M = all_M.to(devive)
                    all_F, all_M = all_F[0], all_M[0]
                    seq_name = info['name'][0]
                    num_frames = info['num_frames'][0]
                    num_objects = info['num_objects'][0]
                    B,C,T,H,W = all_M.shape
                    all_E = torch.zeros(B,C,T,H,W)
                    all_E[:,0,0] = all_M[:,:,0]
                    msv_F1, msv_P1, all_M = ToCudaVariable([all_F[:,:,0], all_E[:,0,0], all_M])
                    ms = model.Encoder(msv_F1, msv_P1)[0]
                    for f in range(0, all_M.shape[2] - 1):
                        output, ms = Propagate_MS(ms, model, all_F[:,:,f+1], all_E[:,0,f])
                        all_E[:,0,f+1] = output.clone()
                        loss = loss + criterion(output, all_M[:,0,f+1].float()) / all_M.size(2)
                    iOU = iOU + iou(torch.cat((1-all_E, all_E), dim=1), all_M)
                pbar.close()
                loss = loss / len(Testloader)
                iOU = iOU / len(Testloader)
                writer.add_scalar('Val/BCE', loss, epoch)
                writer.add_scalar('Val/IOU', iOU, epoch)
                print('loss: {}'.format(loss))
                print('IoU: {}'.format(iOU))
                video_thread = threading.Thread(target=log_mask, args=(all_F,all_E,info,writer,'Val/frames', 'Val/masks'))
                video_thread.start()
        model.train()
        print('[Train] Epoch {}{}{}'.format(font.BOLD, epoch, font.END))
        for i, (all_F, all_M, info) in enumerate(Trainloader):
            all_F = all_F.to(devive)
            all_M = all_M.to(devive)
            optimizer.zero_grad()
            all_F, all_M = all_F[0], all_M[0]
            seq_name = info['name'][0]
            num_frames = info['num_frames'][0]
            num_objects = info['num_objects'][0]
            if (args.bptt_len < num_frames):
                start_frame = random.randint(0, num_frames - args.bptt_len)
                all_F = all_F[:,:, start_frame : start_frame + args.bptt_len]
                all_M = all_M[:,:, start_frame : start_frame + args.bptt_len]
            tt = time.time()
            B,C,T,H,W = all_M.shape
            all_E = torch.zeros(B,C,T,H,W)
            all_E[:,0,0] = all_M[:,:,0]
            msv_F1, msv_P1, all_M = ToCudaVariable([all_F[:,:,0], all_E[:,0,0], all_M])
            ms = model.Encoder(msv_F1, msv_P1)[0] # ms为 1/32, 512, 为第一帧的编码结果
            num_bptt = all_M.shape[2]
            loss = 0
            counter = 0
            for f in range(0, num_bptt - 1):
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
                output, ms = Propagate_MS(ms, model, all_F[:,:,f+1], all_E[:,0,f])
                all_E[:,0,f+1] = output.clone()
                loss = loss + criterion(output, all_M[:,0,f+1].float())
                counter += 1
                if (f+1) % args.bptt_step == 0:
                    optimizer.zero_grad()
                    loss.backward(retain_graph=True)
                    optimizer.step()
                    output.clone()
                    if f < num_bptt - 2:
                        loss = 0
                        counter = 0
            if loss > 0:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if (i+1) % args.disp_interval == 0:
                writer.add_scalar('Train/BCE', loss/counter, i + epoch * iters_per_epoch)
                writer.add_scalar('Train/IOU', iou(torch.cat((1-all_E, all_E), dim=1), all_M), i + epoch * iters_per_epoch)
                print('loss: {}'.format(loss/counter))
            if epoch % 10 == 1 and i == 0:
                video_thread = threading.Thread(target=log_mask, args=(all_F,all_E,info,writer))
                video_thread.start()
        if epoch % 10 == 0 and epoch > 0:
            save_name = '{}/{}.pth'.format(MODEL_DIR, epoch)
            torch.save({'epoch': epoch,
                        'model': model.state_dict(), 
                        'optimizer': optimizer.state_dict(),
                       },
                       save_name)