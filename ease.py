import logging
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from utils.inc_net import EaseNet
from models.base import BaseLearner
from utils.toolkit import tensor2numpy
from tqdm import tqdm
import torch.nn.functional as F
from torch import optim
from .gpm import GPM  # Import the GPM class

num_workers = 8

class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = EaseNet(args, True)

        self.args = args
        self.batch_size = args["batch_size"]
        self.init_lr = args["init_lr"]
        self.weight_decay = args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.min_lr = args["min_lr"] if args["min_lr"] is not None else 1e-8
        self.init_cls = args["init_cls"]
        self.inc = args["increment"]

        self.use_exemplars = args["use_old_data"]
        self.use_init_ptm = args["use_init_ptm"]
        self.use_diagonal = args["use_diagonal"]

        self.recalc_sim = args["recalc_sim"]
        self.alpha = args["alpha"]
        self.beta = args["beta"]

        self.moni_adam = args["moni_adam"]
        self.adapter_num = args["adapter_num"]

        self.gpm = GPM(device=self._device, max_components=20, epsilon=1e-5)

    def after_task(self):
        self._known_classes = self._total_classes
        self._network.freeze()
        # Add the trained adapter to the adapter list and create a new one for the next task
        self._network.backbone.add_adapter_to_list()
        # Re-initialize GPM parameters so that it matches the new parameter dimensions
        self.gpm.initialize_params(self._network)


    def get_cls_range(self, task_id):
        if task_id == 0:
            start_cls = 0
            end_cls = self.init_cls
        else:
            start_cls = self.init_cls + (task_id - 1) * self.inc
            end_cls = start_cls + self.inc
        return start_cls, end_cls

    def replace_fc(self, train_loader):
        model = self._network
        model = model.eval()

        with torch.no_grad():
            if self.use_init_ptm:
                start_idx = -1
            else:
                start_idx = 0

            for index in range(start_idx, self._cur_task + 1):
                if self.moni_adam:
                    if index > self.adapter_num - 1:
                        break
                elif self.use_diagonal and index != -1 and index != self._cur_task:
                    continue

                embedding_list, label_list = [], []
                for i, batch in enumerate(train_loader):
                    (_, data, label) = batch
                    data = data.to(self._device)
                    label = label.to(self._device)
                    embedding = model.backbone.forward_proto(data, adapt_index=index)
                    embedding_list.append(embedding.cpu())
                    label_list.append(label.cpu())

                embedding_list = torch.cat(embedding_list, dim=0)
                label_list = torch.cat(label_list, dim=0)

                class_list = np.unique(self.train_dataset_for_protonet.labels)
                for class_index in class_list:
                    data_index = (label_list == class_index).nonzero().squeeze(-1)
                    embedding = embedding_list[data_index]
                    proto = embedding.mean(0)
                    if self.use_init_ptm:
                        model.fc.weight.data[class_index, (index+1)*self._network.out_dim:(index+2)*self._network.out_dim] = proto
                    else:
                        model.fc.weight.data[class_index, index*self._network.out_dim:(index+1)*self._network.out_dim] = proto

            if self.use_exemplars and self._cur_task > 0:
                embedding_list = []
                label_list = []
                dataset = self.data_manager.get_dataset(np.arange(0, self._known_classes), source="train", mode="test", )
                loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
                for i, batch in enumerate(loader):
                    (_, data, label) = batch
                    data = data.to(self._device)
                    label = label.to(self._device)
                    embedding = model.backbone.forward_proto(data, adapt_index=self._cur_task)
                    embedding_list.append(embedding.cpu())
                    label_list.append(label.cpu())
                embedding_list = torch.cat(embedding_list, dim=0)
                label_list = torch.cat(label_list, dim=0)

                class_list = np.unique(dataset.labels)
                for class_index in class_list:
                    data_index = (label_list == class_index).nonzero().squeeze(-1)
                    embedding = embedding_list[data_index]
                    proto = embedding.mean(0)
                    model.fc.weight.data[class_index, -self._network.out_dim:] = proto

        if self.use_diagonal or self.use_exemplars:
            return

        if self.recalc_sim:
            self.solve_sim_reset()
        else:
            self.solve_similarity()

    def get_A_B_Ahat(self, task_id):
        if self.use_init_ptm:
            start_dim = (task_id + 1) * self._network.out_dim
            end_dim = start_dim + self._network.out_dim
        else:
            start_dim = task_id * self._network.out_dim
            end_dim = start_dim + self._network.out_dim

        start_cls, end_cls = self.get_cls_range(task_id)

        A = self._network.fc.weight.data[self._known_classes:, start_dim : end_dim]
        B = self._network.fc.weight.data[self._known_classes:, -self._network.out_dim:]
        A_hat = self._network.fc.weight.data[start_cls : end_cls, start_dim : end_dim]

        return A.cpu(), B.cpu(), A_hat.cpu()

    def solve_similarity(self):
        for task_id in range(self._cur_task):
            start_cls, end_cls = self.get_cls_range(task_id=task_id)

            A, B, A_hat = self.get_A_B_Ahat(task_id=task_id)

            similarity = torch.zeros(len(A_hat), len(A))
            for i in range(len(A_hat)):
                for j in range(len(A)):
                    similarity[i][j] = torch.cosine_similarity(A_hat[i], A[j], dim=0)

            similarity = F.softmax(similarity, dim=1)

            B_hat = torch.zeros(A_hat.shape[0], B.shape[1])
            for i in range(len(A_hat)):
                for j in range(len(A)):
                    B_hat[i] += similarity[i][j] * B[j]

            self._network.fc.weight.data[start_cls : end_cls, -self._network.out_dim:] = B_hat.to(self._device)

    def solve_sim_reset(self):
        for task_id in range(self._cur_task):
            if self.moni_adam and task_id > self.adapter_num - 2:
                break

            if self.use_init_ptm:
                range_dim = range(task_id + 2, self._cur_task + 2)
            else:
                range_dim = range(task_id + 1, self._cur_task + 1)
            for dim_id in range_dim:
                if self.moni_adam and dim_id > self.adapter_num:
                    break

                start_cls, end_cls = self.get_cls_range(task_id=task_id)

                start_dim = dim_id * self._network.out_dim
                end_dim = (dim_id + 1) * self._network.out_dim

                if self.use_init_ptm:
                    start_cls_old = self.init_cls + (dim_id - 2) * self.inc
                    end_cls_old = self._total_classes
                    start_dim_old = (task_id + 1) * self._network.out_dim
                    end_dim_old = (task_id + 2) * self._network.out_dim
                else:
                    start_cls_old = self.init_cls + (dim_id - 1) * self.inc
                    end_cls_old = self._total_classes
                    start_dim_old = task_id * self._network.out_dim
                    end_dim_old = (task_id + 1) * self._network.out_dim

                A = self._network.fc.weight.data[start_cls_old:end_cls_old, start_dim_old:end_dim_old].cpu()
                B = self._network.fc.weight.data[start_cls_old:end_cls_old, start_dim:end_dim].cpu()
                A_hat = self._network.fc.weight.data[start_cls:end_cls, start_dim_old:end_dim_old].cpu()

                similarity = torch.zeros(len(A_hat), len(A))
                for i in range(len(A_hat)):
                    for j in range(len(A)):
                        similarity[i][j] = torch.cosine_similarity(A_hat[i], A[j], dim=0)

                similarity = F.softmax(similarity, dim=1)

                B_hat = torch.zeros(A_hat.shape[0], B.shape[1])
                for i in range(len(A_hat)):
                    for j in range(len(A)):
                        B_hat[i] += similarity[i][j] * B[j]

                self._network.fc.weight.data[start_cls : end_cls, start_dim : end_dim] = B_hat.to(self._device)

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        self.data_manager = data_manager
        self.train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="train", mode="train")
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)

        self.test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test" )
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=num_workers)

        self.train_dataset_for_protonet = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="train", mode="test", )
        self.train_loader_for_protonet = DataLoader(self.train_dataset_for_protonet, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)

        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        self._train(self.train_loader, self.test_loader)

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

        self.replace_fc(self.train_loader_for_protonet)

        criterion = nn.CrossEntropyLoss()
        # Use the corrected update_memory_with_param_space
        # self.gpm.update_memory_with_param_space(self._network, self.train_loader, criterion)
        self.gpm.update_memory(self._network, self.train_loader, criterion, self._known_classes)

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)

        if self._cur_task == 0 or self.init_cls == self.inc:
            optimizer = self.get_optimizer(lr=self.args["init_lr"])
            scheduler = self.get_scheduler(optimizer, self.args["init_epochs"])
            epochs = self.args['init_epochs']
        else:
            if "later_lr" not in self.args or self.args["later_lr"] == 0:
                self.args["later_lr"] = self.args["init_lr"]
            if "later_epochs" not in self.args or self.args["later_epochs"] == 0:
                self.args["later_epochs"] = self.args["init_epochs"]
            optimizer = self.get_optimizer(lr=self.args["later_lr"])
            scheduler = self.get_scheduler(optimizer, self.args["later_epochs"])
            epochs = self.args['later_epochs']

        criterion = nn.CrossEntropyLoss()
        prog_bar = tqdm(range(epochs))

        use_gpm = (self._cur_task > 0)

        if self._cur_task == 0:
            # Initialize param ordering after first training if not done
            if not self.gpm.param_initialized:
                self.gpm.initialize_params(self._network)

        for epoch in prog_bar:
            self._network.train()
            losses = 0.0
            correct, total = 0, 0

            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                aux_targets = targets.clone()
                aux_targets = torch.where(
                    aux_targets - self._known_classes >= 0,
                    aux_targets - self._known_classes,
                    -1,
                )

                optimizer.zero_grad()
                output = self._network(inputs, test=False)
                logits = output["logits"]
                loss = criterion(logits, aux_targets)
                loss.backward()

                if use_gpm:
                    self.gpm.project_gradients(self._network)

                optimizer.step()

                losses += loss.item()
                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(aux_targets.expand_as(preds)).cpu().sum()
                total += len(aux_targets)

            if scheduler:
                scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                self._cur_task,
                epoch + 1,
                epochs,
                losses / len(train_loader),
                train_acc,
            )
            prog_bar.set_description(info)

        logging.info(info)

    def get_optimizer(self, lr):
        if self.args['optimizer'] == 'sgd':
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                momentum=0.9,
                lr=lr,
                weight_decay=self.weight_decay
            )
        elif self.args['optimizer'] == 'adam':
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                lr=lr,
                weight_decay=self.weight_decay
            )
        elif self.args['optimizer'] == 'adamw':
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                lr=lr,
                weight_decay=self.weight_decay
            )
        return optimizer

    def get_scheduler(self, optimizer, epoch):
        if self.args["scheduler"] == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epoch, eta_min=self.min_lr)
        elif self.args["scheduler"] == 'steplr':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=self.args["init_milestones"], gamma=self.args["init_lr_decay"])
        elif self.args["scheduler"] == 'constant':
            scheduler = None
        return scheduler

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model.forward(inputs, test=True)["logits"]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)
        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = self._network.forward(inputs, test=True)["logits"]
            predicts = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[1]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())
        return np.concatenate(y_pred), np.concatenate(y_true)
