import cv2
import numpy as np
import torch
from sklearn import metrics
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from visdom import Visdom

import option
from Utils.old.utils import Prepare_logger
from dataset import CTDD
from Models.HTN import HTN
from Utils.utils import AverageMeter, calculate_iou, accuracy, calculate_coss, calculate_psnr, calculate_dice, \
    calculate_iinc


def plot_image(masks, name):
    for j in range(masks.size(0)):
        image = np.array(masks[j].cpu())
        image[image > 0.5] = 1
        image[image <= 0.5] = 0
        image = image.astype(np.uint8) * 255
        img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        cv2.imwrite('Masks/' + name[j], img)

def test(data_loader, model, cls_criterion, seg_criterion):
    total_acc = AverageMeter('Acc@1', ':6.2f')
    total_loss = AverageMeter('Loss', ':.5f')
    total_seg_loss = AverageMeter('segLoss', ':.5f')
    total_cls_loss = AverageMeter('clsLoss', ':.5f')
    total_coss = AverageMeter('coss', ':.4e')
    total_psnr = AverageMeter('psnr', ':.4e')
    total_dice = AverageMeter('dice', ':.4e')
    total_iinc = AverageMeter('iinc', ':.4e')
    total_iou = AverageMeter('Iou', ':.4e')
    total_pbca = AverageMeter('pbca', ':.4e')
    metric = {}
    with torch.no_grad():
        model.eval()
        y_trues = []
        y_preds = []
        for test_data in tqdm(data_loader):
            image, mask, name, label = test_data
            image = image.cuda()
            mask = mask.cuda()
            label = label.cuda().long()
            pred_seg, pred_cls = model(image)

            # --- localization metric ---
            coss = calculate_coss(pred_seg.cpu(), mask.cpu())
            psnr = calculate_psnr(pred_seg, mask)
            dice = calculate_dice(pred_seg, mask)
            iinc = calculate_iinc(pred_seg, mask)
            iou = calculate_iou(pred_seg, mask)
            #plot_image(pred_seg[0].squeeze(1), name)

            seg_loss = torch.mean(seg_criterion(pred_seg, mask))
            cls_loss = cls_criterion(pred_cls, label)
            loss = seg_loss + cls_loss

            total_iou.update(iou.item(), image.size(0))
            total_iinc.update(iinc, image.size(0))
            total_dice.update(dice, image.size(0))
            total_psnr.update(psnr, image.size(0))
            total_coss.update(coss, image.size(0))

            acc = accuracy(pred_cls, label, topk=(1,))[0]
            total_acc.update(acc, image.size(0))
            total_loss.update(loss.item(), image.size(0))
            total_seg_loss.update(seg_loss.item(), image.size(0))
            total_cls_loss.update(cls_loss, image.size(0))

            y_trues.extend(label.cpu().numpy())
            prob = 1 - torch.softmax(pred_cls, dim=1)[:, 0].cpu().numpy()
            y_preds.extend(prob)

        fpr, tpr, thresholds = metrics.roc_curve(y_trues, y_preds, pos_label=1)
        auc = metrics.auc(fpr, tpr) * 100
        eer = fpr[np.nanargmin(np.abs(tpr - (1 - fpr)))] * 100

        metric['iou'] = total_iou.avg
        metric['iinc'] = total_iinc.avg
        metric['dice'] = total_dice.avg
        metric['psnr'] = total_psnr.avg
        metric['coss'] = total_coss.avg

    return total_loss.avg, total_seg_loss.avg, total_cls_loss.avg, total_acc.avg, auc, eer, metric

if __name__ == '__main__':

    args = option.parser.parse_args()
    logger = Prepare_logger(eval=False)
    logger.info(args)
    cls_criterion = nn.CrossEntropyLoss().cuda()
    seg_criterion = nn.BCELoss(reduction='none').cuda()
    viz = Visdom()

    viz.line([0.],
             [0.],
             win=args.dataset,
             opts=dict(title=args.dataset, legend=['ROC curve(area=89.69%)'], xlabel='False positive Rate'
                       , ylabel='True Positive Rate')
             )
    test_dataset = CTDD(args, state=2, transforms=None)
    test_loader = DataLoader(test_dataset,
                             batch_size=args.batch_size, shuffle=True,
                             num_workers=args.workers, pin_memory=True)

    logger.info('TestSet Number:{} Positive Number:{} Negative Number:{}'.
                format(test_dataset.num, test_dataset.positive, test_dataset.negative))


    model = HTN(args).cuda()
    checkpoint = torch.load(r'Checkpoint\c23-Deepfakes\FF++_240225221535.tar')
    model.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_seg_loss, test_cls_loss, test_acc, test_auc, test_eer, metric, tpr, fpr = \
        test(test_loader, model, cls_criterion, seg_criterion)
    logger.info('Epoch {}/{}: TES loss:{:.4} segloss:{:.4} clsloss:{:.4} acc:{:.4} auc:{:.4} eer:{:.4}'.
                format(1, args.epoch, test_loss, test_seg_loss, test_cls_loss, test_acc, test_auc, test_eer))

    viz.line(tpr,  
             fpr,  
             win=args.dataset,  
             update='append'  
             )



