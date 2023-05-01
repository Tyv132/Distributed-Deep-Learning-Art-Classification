import torch
import torch.nn as nn
import torch.distributed as dist
import torchvision
from torchvision import models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from torch import Generator
import argparse
from datetime import datetime
import os
import torch.multiprocessing as mp
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--master_port', type=str)
    parser.add_argument('-n', '--nodes', type=int, metavar='N')
    parser.add_argument('-g', '--gpus', type=int)
    parser.add_argument('-b', '--batch_size', type=int, metavar='N')
    parser.add_argument('-l', '--learning_rate', type=float)
    parser.add_argument('-t', '--train_size', type=int)
    parser.add_argument('-v', '--val_size', type=int)
    parser.add_argument('-s', '--test_size', type=int)
    parser.add_argument('-e', '--epochs', type=int)
    args = parser.parse_args()

    args.world_size = args.gpus * args.nodes #node
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.master_port
    mp.spawn(train, nprocs=args.gpus, args=(args,))

def train(gpu, args):
    rank = 0 * args.gpus + gpu
    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=rank)
    torch.manual_seed(0)

    # Load model architecture from torchvision
    model = torchvision.models.alexnet(weights=None)
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = torch.nn.Linear(num_ftrs, 14) # 14 classes
    
    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    
    # Define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(gpu)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    
    # Define model
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    # Data loading code
    data_dir = "../art_data/images/"
    transformations = torchvision.transforms.Compose([
        torchvision.transforms.Resize((300, 300)),
        torchvision.transforms.ToTensor()
    ])

    # Download and load the training data
    dataset_all = ImageFolder(data_dir, transformations)

    # Split into Test, Validation, and Training
    print('The size of the training data set is {}'.format(args.train_size))
    test_dataset, val_dataset, train_dataset = random_split(dataset_all, [args.test_size, args.val_size, args.train_size], \
                                                            generator=Generator().manual_seed(42))
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=args.world_size,
        rank=rank
    )
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset,
        num_replicas=args.world_size,
        rank=rank
    )

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=False,
                                               num_workers=0,
                                               pin_memory=True,
                                               sampler=train_sampler)
    
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=0,
                                             pin_memory=True,
                                             sampler=val_sampler)

    # Training
    #if gpu == 0:
    #    times = []
    total_step = len(train_loader)
    for epoch in range(0, args.epochs):
        if gpu == 0:
            print('Start training')
            a = datetime.now()
            
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Print step number
            if (i + 1) % 1 == 0 and gpu == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'\
                      .format(epoch + 1, args.epochs, i + 1, total_step, loss.item()))
                   
        # Save epoch time
        if gpu == 0:
            b = datetime.now()
            delta = b - a
            print('Total time for epoch {}/{} = {} seconds'.format(epoch+1, args.epochs, \
                                                                   delta.total_seconds()))
        #if gpu == 0 and epoch > 0:
        #    times.append(delta.total_seconds())
        #    print('The average of the epoch train time so far is {}'.format(np.array(times).mean()))
        #    print('The stdev of the epoch train time so far is {}'.format(np.array(times).std()))
         
        # Validation
        if gpu == 0:
            print('Start validation')
            model.eval()
            size = len(val_loader.dataset)
            num_batches = len(val_loader)
            test_loss, correct = 0, 0
            with torch.no_grad():
                for X, y in val_loader:
                    X = X.cuda(non_blocking=True)
                    y = y.cuda(non_blocking=True)
            
                    # Compute prediction
                    pred = model(X)
            
                    # Testing loss
                    test_loss += criterion(pred, y).item()
            
                    # Correct predictions 
                    correct += (pred.argmax(1) == y).type(torch.float).sum().item()

                # Test loss
                test_loss /= num_batches
    
                # Test accuracy
                accuracy = correct / size    
        
            # Show validation accuracy
            if gpu == 0:
                print('Validation accuracy is {}%'.format(accuracy*100))
if __name__ == '__main__':
    main()