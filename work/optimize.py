import os
from collections import Counter
from datetime import datetime
from logging import getLogger
from time import sleep, time
from lib.utils import read_as_int


# config
from args import Config

# tensorboard
from tensorboardX import SummaryWriter

# data
import numpy as np
from lib.board import bit_to_array
from lib.data_helper import PlayDataset
from lib.data_helper import get_play_data_filenames, read_game_data_from_file, \
    get_next_generation_model_dirs

# model
from agent.model import OthelloModel

# torch
import torch
from torch.optim import SGD

# logger
logger = getLogger(__name__)


def start(config: Config):
    return OptimizeWorker(config).start()


class OptimizeWorker:
    def __init__(self, config: Config):
        # model
        self.config = config
        self.model = None  # type: OthelloModel
        self.optimizer = None

        # data
        self.count = 400
        self.loaded_data = {}
        self.dataset = None     # type: Numpyarray
        self.training_count_of_files = Counter()


    def start(self):
        # start hehe
        self.training()

    def load_model_and_optimizer(self):
        # model
        self.load_model()
        # optimizer
        self.optimizer = SGD(params=filter(lambda p: p.requires_grad, self.model.parameters()) ,lr=1e-2, momentum=0.9)


    def training(self):
        # tensorboard usage
        if self.config.trainer.use_tensorboard:
            self.writer = SummaryWriter(comment='Alpha_go', log_dir=self.config.resource.tensorboard_log_dir)
        total_steps = self.config.trainer.start_total_steps

        # train!
        self.load_model_and_optimizer()
        while True:
            if not self.load_play_data():
                sleep(300)
                continue
            self.update_learning_rate(total_steps)
            total_steps += self.train_epoch(self.config.trainer.epoch_to_checkpoint)
            self.count_up_training_count_and_delete_self_play_data_files()

    def load_play_data(self):
        # get game_data
        filenames = get_play_data_filenames(self.config.resource)

        # load all the files that's been updatad
        updated = False
        for filename in filenames:
            if filename in self.loaded_data.keys():
                continue
            self.load_data_from_file(filename)
            updated = True

        # unload the files that has been delete
        for filename in (set(self.loaded_data.keys()) - set(filenames)):
            self.unload_data_of_file(filename)
            self.count += 1
            updated = True

        # return dataset
        if updated:
            logger.debug("updating training dataset")
            self.set_dataset()

        # if small gg
        if self.count < 400 or self.dataset_size < self.config.trainer.min_data_size_to_learn:
            logger.info(f"new={self.count} is less than 400")
            return False
        self.count = 0
        return True


    def load_data_from_file(self, filename):
        logger.debug(f"loading data from {filename}")
        self.loaded_data[filename] = self.convert_to_training_data(read_game_data_from_file(filename))


    def unload_data_of_file(self, filename):
        logger.debug(f"removing data about {filename} from training set")
        if filename in self.loaded_data:
            del self.loaded_data[filename]
        if filename in self.training_count_of_files:
            del self.training_count_of_files[filename]

    def set_dataset(self):
        state_ary_list, policy_ary_list, z_ary_list = [], [], []
        for s_ary, p_ary, z_ary_ in self.loaded_data.values():
            state_ary_list.append(s_ary)
            policy_ary_list.append(p_ary)
            z_ary_list.append(z_ary_)
        if state_ary_list:
            state_ary = np.concatenate(state_ary_list)
            policy_ary = np.concatenate(policy_ary_list)
            z_ary = np.concatenate(z_ary_list)
            # print(state_ary.shape, policy_ary.shape, z_ary.shape)
            self.dataset = PlayDataset(state_ary, policy_ary, z_ary)

    def train_epoch(self, epochs):
        # dataset
        train_id = read_as_int(self.config.resource.self_play_game_idx_file)
        current_train_epoch = read_as_int(self.config.resource.current_train_epoch) or -1
        tc = self.config.trainer
        train_dataset_loader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=tc.batch_size,
            shuffle=True,
            num_workers=4)

        # train
        best_loss = 1e10
        for _ in range(current_train_epoch+1, epochs):
            loss_list = []
            for batch_idx, data in enumerate(train_dataset_loader):
                # train process
                self.optimizer.zero_grad()
                loss = self.model.loss(*self.model(data[0].float()), data[1].float(), data[2].float())
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2)
                self.optimizer.step()
                loss_list.append(loss.cpu().item())
            # save model and tensor board
            loss = sum(loss_list)/len(self.dataset)
            if loss<=best_loss:
                best_loss = loss
                self.save_current_model(train_id)
            # self.writer.add_scalar('Train', loss, _)
            logger.debug('Epoch {}/{}, loss:{:.4f}'.format(_ + 1, epochs, loss))
            with open(self.config.resource.current_train_epoch, "wt") as f:
                f.write(str(_))

        self.writer.close()
        return (len(self.dataset)//tc.batch_size)*epochs


    def update_learning_rate(self, total_steps):
        # The deepmind paper says
        # ~400k: 1e-2
        # 400k~600k: 1e-3
        # 600k~: 1e-4
        lr = self._decide_learning_rate(total_steps)
        if lr:
            self.optimizer.lr = lr
            logger.debug(f"total step={total_steps}, set learning rate to {lr}")

    def _decide_learning_rate(self, total_steps):
        # no lr file
        ret = None
        for step, lr in self.config.trainer.lr_schedules:
            if total_steps >= step:
                ret = lr
        return ret

    def load_model(self):
        # load best_model
        self.model = OthelloModel(self.config)
        self.model.train()
        dirs = get_next_generation_model_dirs(self.config.resource)
        if not dirs:
            logger.debug(f"loading best model")
            self.model.load(self.model.config.resource.model_best_weight_path)
        else:  # load next generation
            logger.debug(f"loading latest model")
            self.model.load(os.path.join(dirs[-1], self.config.resource.next_generation_model_weight_filename))

    def save_current_model(self,train_id):
        # save current model
        rc = self.config.resource
        # model_dir=
        model_dir = os.path.join(rc.next_generation_model_dir, rc.next_generation_model_dirname_tmpl % train_id)
        os.makedirs(model_dir, exist_ok=True)
        # save model_dir_file
        weight_path = os.path.join(model_dir, rc.next_generation_model_weight_filename)
        self.model.save(weight_path)

    def count_up_training_count_and_delete_self_play_data_files(self):
        # if delete self_play
        limit = self.config.trainer.delete_self_play_after_number_of_training
        if not limit:
            return
        else:
            for filename in self.loaded_data.keys():
                self.training_count_of_files[filename] += 1
                if self.training_count_of_files[filename] >= limit:
                    if os.path.exists(filename):
                        try:
                            logger.debug(f"remove {filename}")
                            os.remove(filename)
                        except Exception as e:
                            logger.warning(e)


    @property
    def dataset_size(self):
        if self.dataset is None:
            return 0
        return len(self.dataset)

    @staticmethod
    def convert_to_training_data(data):
        """
        :param data: format is SelfPlayWorker.buffer
            list of [(own: bitboard, enemy: bitboard), [policy: float 64 items], z: number]
        :return:
        """
        state_list = []
        policy_list = []
        z_list = []
        for state, policy, z in data:
            # state = [64,64]
            # state_list = [(array,array)]
            # value = +1
            own, enemy = bit_to_array(state[0], 64).reshape((8, 8)), bit_to_array(state[1], 64).reshape((8, 8))
            state_list.append([own, enemy])
            policy_list.append(policy)
            z_list.append(z)
        return np.array(state_list), np.array(policy_list), np.array(z_list)

