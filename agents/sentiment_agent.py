import numpy as np

from tqdm import tqdm
import shutil
import random

import torch
from torch import nn
from torch.backends import cudnn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F

from agents.base import BaseAgent

from graphs.models.sentiment_rnn import RNN
from datasets.dataloader import SentimentDataLoader
from tensorboardX import SummaryWriter
from utils.metrics import AverageMeter, AverageMeterList
from utils.misc import print_cuda_statistics


class SentimentAgent(BaseAgent):

    def __init__(self, config):
        super().__init__(config)

        # define data_loader
        self.data_loader = SentimentDataLoader(config=config)

        # define models
        self.model = RNN(input_dim=self.data_loader.training_set.vocab_size,
                         embedding_dim=self.config.embedding_dim,
                         hidden_dim=self.config.hidden_dim,
                         output_dim=self.config.output_dim)

        # define loss
        self.loss_fn = nn.NLLLoss()

        # define optimizer
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.config.learning_rate, momentum=self.config.momentum)

        # initialize counter
        self.current_epoch = 0
        self.current_iteration = 0
        self.best_metric = 0

        # set cuda flag
        self.is_cuda = torch.cuda.is_available()

        if self.is_cuda and not self.config.cuda:
            self.logger.info("WARNING: You have a CUDA device, so you should probably enable CUDA")

        self.cuda = self.is_cuda & self.config.cuda

        # set the manual seed for torch
        self.manual_seed = self.config.seed
        if self.cuda:
            #torch.cuda.manual_seed(self.manual_seed)
            self.device = torch.device("cuda")
            torch.cuda.set_device(self.config.gpu_device)
            self.model = self.model.to(self.device)
            self.loss = self.loss.to(self.device)

            self.logger.info("Program will run on *****GPU-CUDA***** ")
            print_cuda_statistics()
        else:
            self.device = torch.device("cpu")
            torch.manual_seed(self.manual_seed)
            self.logger.info("Program will run on *****CPU*****\n")

        # Model Loading from the latest checkpoint if not found start from scratch.
        self.load_checkpoint(self.config.checkpoint_file)
        # Summary Writer
        self.summary_writer = None

    def load_checkpoint(self, file_name):
        pass

    def save_checkpoint(self, file_name="checkpoint.pth.tar", is_best=0):
        pass

    def run(self):
        try:
            self.train()
        except KeyboardInterrupt:
             self.logger.info("You have entered CTRL+C.. Wait to finalize")

    def train(self):
        for epoch in range(1, self.config.max_epoch + 1):
            self.train_one_epoch()
            self.validate()

            self.current_epoch += 1

    def train_one_epoch(self):
        self.model.train()
        #here it need to to an iterator over vocabulary
        for batch_idx, (data, target) in enumerate(self.data_loader.train_loader):
            #print("TENSOR DATA 1", data.shape)
            data = torch.stack(data)
            print("TENSOR DATA 2", data.shape)
            target = target[0]
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)

            target = target[0]
            output = output.float()
            target = target.float()
            loss = F.mse_loss(output, target)
            loss.backward()
            self.optimizer.step()
            if batch_idx % self.config.log_interval == 0:
                self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    self.current_epoch, batch_idx * len(data),self.data_loader.training_set.vocab_size ,
                        100. * batch_idx / len(self.data_loader.train_loader), loss.item()))
            self.current_iteration += 1

    def validate(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.data_loader.test_loader:
                data = torch.stack(data).T
                target = target[0]
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                output = output.float()
                target = target.float()
                loss = F.mse_loss(output, target)
                test_loss += F.mse_loss(output, target).item() # sum up batch loss
                pred = output.max(1, keepdim=True)[1]
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.data_loader.test_loader.dataset)
        self.logger.info('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(self.data_loader.test_loader.dataset),
            100. * correct / len(self.data_loader.test_loader.dataset)))

    def finalize(self):
        pass