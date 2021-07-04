import os
import argparse
import datetime
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch_util import *
from torch.utils.data import DataLoader
import faiss
import time

from data_loader.train_data_loader import Image_Motion_Data_Training
from data_loader.test_data_loader import Image_Motion_Data_Evaluation
from model.VGG_Transformer_VLAD import NetWork
device = torch.device('cuda')

print('Network training script...')
# Argument parsing
parser = argparse.ArgumentParser(description='US place recognition')
parser.add_argument('--checkpoint', type=str, default='/home/engs2191/', help='the path of checkpoint')
parser.add_argument('--dataset_root_dir', type=str, default='/home/engs2191/simulation_PULSE_data_2nd/simulation_PULSE_data', help='the path of dataset')
parser.add_argument('--results_dir', type=str, default='/home/engs2191/PULSE/results/', help='the path of evaluation results')
parser.add_argument('--log_dir', type=str, default='/home/engs2191/PULSE/log', help='the path of log')
parser.add_argument('--image_filetype', type=str, default='png')
parser.add_argument('--motion_filetype', type=str, default='csv')
parser.add_argument('--image_size', type=int, default=(400, 274))
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--batch_num_queries', type=int, default=1, help='Batch Size during training')
parser.add_argument('--positives_per_query', type=int, default=10, help='Number of potential positives in each training tuple')
parser.add_argument('--negatives_per_query', type=int, default=10, help='Number of definite negatives in each training tuple')
parser.add_argument('--learning_rate', type=float, default=1e-5, help='learning rate')
parser.add_argument('--result_model_dir', type=str, default='/home/engs2191/trained_models/', help='path to trained models folder')
parser.add_argument('--num_epochs', type=int, default=15, help='number of training epochs')
parser.add_argument('--log_interval', type=int, default=1, help='interval of loss printing')
parser.add_argument('--evalEvery', type=int, default=1, help='interval of model testing')
parser.add_argument('--decay_rate', type=float, default=0.1, help='the decay rate after each epoch')
parser.add_argument('--logs_path', type=str, default='/home/engs2191/trained_logs/', help='the path of logs folder')
parser.add_argument('--num_workers', type=int, default=0, help='the number of workers for the data loading') # use 0 for debug
parser.add_argument('--emb_dims', type=int, default=512, help='number of embed dimension')
parser.add_argument('--output_dims', type=int, default=4096, help='number of output dimension')
parser.add_argument('--num_clusters', type=int, default=64, help='number of clusters of VLAD')
parser.add_argument('--margin', type=float, default=0.1, help='Margin for triplet loss.')
parser.add_argument('--freeze', type=bool, default=False, help='freeze the feature extraction')
parser.add_argument('--resume', action='store_true', default=False, help='If present, restore checkpoint and resume training')

args = parser.parse_args()

if not os.path.exists(args.results_dir):
    os.mkdir(args.results_dir)

if not os.path.exists(args.log_dir):
    os.mkdir(args.log_dir)

LOG_FOUT = open(os.path.join(args.log_dir, 'log_train.txt'), 'w')
def log_string(out_str):
    LOG_FOUT.write(out_str + '\n')
    LOG_FOUT.flush()
    print(out_str)
log_string(str(args))


def train_epoch(mode, epoch, model, loss_fn, optimizer, dataloader, log_interval, writer):
    epoch_loss = 0
    length = len(dataloader)

    for batch_idx, (query_image, pos_image, neg_image, pos_distance, neg_distance, position, orientation) in enumerate(dataloader):
        input = torch.cat([query_image, pos_image, neg_image], dim=1)
        input = input.to(device)
        pos_distance = pos_distance.to(device)
        neg_distance = neg_distance.to(device)

        image_discriptor = model(input)
        query_discriptor, pos_discriptor, neg_discriptor = \
            torch.split(image_discriptor, [args.batch_num_queries, args.positives_per_query, args.negatives_per_query], dim=0)

        # for triplet loss
        query_discriptor = query_discriptor.repeat(args.negatives_per_query, 1)
        # pos_discriptor = pos_discriptor.repeat(args.negatives_per_query, 1)
        loss = loss_fn(query_discriptor, pos_discriptor, neg_discriptor)

        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()

        epoch_loss += loss
        writer.add_scalar('loss', loss, global_step=epoch*length+batch_idx)

        if batch_idx % log_interval == 0:
            print(mode.capitalize()+' Epoch: {} [{}/{} ({:.0f}%)]\t\tLoss: {:.6f}'.format(
                epoch, batch_idx, length, 100. * batch_idx / length, loss))

    epoch_loss /= max(1, length)
    print(mode.capitalize() + ' set: Average loss of epoch {} : {:.6f}'.format(epoch, epoch_loss))
    log_string(mode.capitalize() + ' set: Average loss of epoch {} : {:.6f}'.format(epoch, epoch_loss))
    writer.add_scalar('epoch_loss', epoch_loss, global_step=epoch)
    return epoch_loss


def test_epoch(mode, epoch, model, dataloader, number_query, writer):
    # feature_size = args.emb_dims * args.num_clusters
    feature_size = args.output_dims
    qFeat = np.empty((number_query, feature_size))
    dbFeat = np.empty((len(dataloader), feature_size))
    gt = np.empty((number_query, 20))

    for batch_idx, (flag, query_image, database_image, indices, distance) in enumerate(dataloader):
        if flag == 0:
            # print(mode.capitalize() + ' processing batch_idx: {}/{}'.format(batch_idx+1, len(dataloader)))
            input = torch.cat([query_image, database_image], dim=1)
            input = input.to(device)

            image_discriptor = model(input)
            query_discriptor, database_discriptor = torch.split(image_discriptor, [1, 1], dim=0)

            qFeat[batch_idx, :] = query_discriptor.detach().cpu().numpy()
            dbFeat[batch_idx, :] = database_discriptor.detach().cpu().numpy()
            gt[batch_idx, :] = indices.detach().cpu().numpy()
        else:
            # print(mode.capitalize() + ' processing batch_idx: {}/{}'.format(batch_idx+1, len(dataloader)))
            input = database_image.to(device)

            torch.cuda.empty_cache()
            start = time.time()
            database_discriptor = model(input)
            end = time.time()
            print("The inference time of index {} is {:.6f}".format(batch_idx, end - start))

            dbFeat[batch_idx, :] = database_discriptor.detach().cpu().numpy()

    qFeat = qFeat.astype('float32')
    dbFeat = dbFeat.astype('float32')

    print('====> Building faiss index')
    faiss_index = faiss.IndexFlatL2(feature_size)
    faiss_index.add(dbFeat)

    print('====> Calculating recall @ N')
    n_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    _, predictions = faiss_index.search(qFeat, max(n_values))
    # for each query get those within threshold distance
    correct_at_n = np.zeros(len(n_values))
    for qIx, pred in enumerate(predictions):
        for i, n in enumerate(n_values):
            # if in top N then also in top NN, where NN > N
            if np.any(np.in1d(pred[:n], gt[qIx])):
                correct_at_n[i:] += 1
                break

    recall_at_n = correct_at_n / len(gt)
    for i, n in enumerate(n_values):
        print("====> Recall@{}: {:.4f}".format(n, recall_at_n[i]))
        writer.add_scalar(mode.capitalize() + 'Recall@' + str(n), recall_at_n[i], global_step=epoch)

    results_outfile = args.results_dir + 'results_' + str(epoch) + '.txt'
    with open(results_outfile, "w") as output:
        output.write("Average Recall @N:\n")
        output.write(str(recall_at_n))
        output.write("\n\n")

    return recall_at_n[1]


def train():
    if os.path.exists(args.logs_path):
         shutil.rmtree(args.logs_path)
    writer = SummaryWriter(log_dir=args.logs_path)

    print('Creating Network ...')
    model = NetWork(output_dim=args.output_dims, emb_dims=args.emb_dims, num_clusters=args.num_clusters, layer_number=3)
    if torch.cuda.is_available():
        model.cuda()

    # print("Loading model:", args.checkpoint)
    # checkpoint = torch.load(args.checkpoint)
    # model.load_state_dict(checkpoint[''])

    # Optimizer
    if args.freeze:
        for param in model.feature_encoder.parameters():
            param.requires_grad = False

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=args.decay_rate)

    if args.resume:
        resume_filename = args.result_model_dir + "checkpoint.pth.tar"
        print("Resuming From ", resume_filename)

        checkpoint = torch.load(resume_filename)
        starting_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        starting_epoch = 0

    loss_fn = torch.nn.TripletMarginLoss(margin=args.margin ** 0.5, p=2, reduction='sum').to(device)

    best_test_recall = 0

    data_train = Image_Motion_Data_Training(dataset_root_dir=args.dataset_root_dir,
                                        image_filetype=args.image_filetype,
                                        motion_filetype=args.motion_filetype,
                                        image_size=args.image_size,
                                        pos_number=args.positives_per_query,
                                        neg_number=args.negatives_per_query,
                                        train=True)

    print('The size of train dataset: ', len(data_train))
    dl_train = DataLoader(data_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    print('The size of train dataset loader: ', len(dl_train))

    data_test = Image_Motion_Data_Evaluation(dataset_root_dir=args.dataset_root_dir,
                                            image_filetype=args.image_filetype,
                                            motion_filetype=args.motion_filetype,
                                            image_size=args.image_size,
                                            topN=20)

    print('The size of test dataset: ', len(data_test))
    dl_test = DataLoader(data_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    number_query = data_test.query_len
    print('The size of test dataset loader, database: ', len(dl_test))
    print('The size of test dataset loader, dataquery: ', number_query)

    train_loss = np.zeros(args.num_epochs)
    top1_recalls = np.zeros(args.num_epochs)

    print('Starting training...')

    for epoch in range(starting_epoch, args.num_epochs):
        for param_group in optimizer.param_groups:
            print("Current learning rate of this epoch is: ", param_group['lr'])
        writer.add_scalar('learning rate', param_group['lr'], global_step=epoch)

        train_loss[epoch] = train_epoch('train', epoch, model, loss_fn, optimizer, dl_train, args.log_interval, writer)
        scheduler.step()

        if (epoch % args.evalEvery) == 0:
            top1_recalls[epoch] = test_epoch('test', epoch, model, dl_test, number_query, writer)
            # remember best recall
            # is_best = top1_recalls[epoch] > best_test_recall
            best_test_recall = max(top1_recalls[epoch], best_test_recall)
            # is_best = False

        # Define checkpoint name
        checkpoint_name = os.path.join(args.result_model_dir, datetime.datetime.now().strftime(
            "%Y-%m-%d_%H:%M") + '_epoch' + str(epoch) + '.pth.tar')
        torch.save({'epoch': epoch,
                    'args': args,
                    'train_loss': train_loss,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict()},
                    checkpoint_name,)
        print("Model Saved As " + checkpoint_name)

    writer.close()
    print('Done!')
    return


if __name__ == '__main__':
    train()
