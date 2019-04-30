import argparse
import torch
import os
import yaml
from moke_config import ConfigBase
from moke_config import create_config

# src
def _project_dir():
    return os.path.dirname(os.path.abspath(__file__))

# data
def _data_dir():
    return os.path.join(_project_dir(), "data")

class Config(ConfigBase):
    def __init__(self):
        self.type = "default"
        self.opts = Options()
        self.resource = ResourceConfig()
        self.model = ModelConfig()
        self.play = PlayConfig()
        self.play_data = PlayDataConfig()
        self.trainer = TrainerConfig()
        self.eval = EvaluateConfig()
        self.play_with_human = EnvGuiConfig()


class Options(ConfigBase):
    new = False
    device = None
    do = None


class ResourceConfig(ConfigBase):
    def __init__(self):
        self.project_dir = os.environ.get("PROJECT_DIR", _project_dir())
        self.data_dir = os.environ.get("DATA_DIR", _data_dir())
        self.model_dir = os.environ.get("MODEL_DIR", os.path.join(self.data_dir, "model"))
        #models
        self.model_best_weight_dir = os.path.join(self.model_dir, "best_weight")
        self.model_best_weight_path = os.path.join(self.model_best_weight_dir, "model_best_weight.h5")

        #models
        self.next_generation_model_dir = os.path.join(self.model_dir, "next_generation")
        self.next_generation_model_dirname_tmpl = "model_%s"
        self.next_generation_model_weight_filename = "model_weight.h5"

        #data
        self.play_data_dir = os.path.join(self.data_dir, "play_data")
        self.play_data_filename_tmpl = "play_%s.json"
        self.self_play_game_idx_file = os.path.join(self.data_dir, ".self-play-game-idx")
        self.current_train_epoch = os.path.join(self.data_dir, ".train_epoch")

        #logs
        self.log_dir = os.path.join(self.project_dir, "logs")
        self.main_log_path = os.path.join(self.log_dir, "main.log")
        self.tensorboard_log_dir = os.path.join(self.log_dir, 'tensorboard')

    def create_directories(self):
        dirs = [self.project_dir, self.data_dir, self.model_dir, self.play_data_dir, self.log_dir,
                self.next_generation_model_dir, self.model_best_weight_dir]
        for d in dirs:
            if not os.path.exists(d):
                os.makedirs(d)


class EnvGuiConfig(ConfigBase):
    def __init__(self):
        self.parallel_search_num = 8
        self.noise_eps = 0
        self.change_tau_turn = 0
        self.resign_threshold = None
        self.use_newest_next_generation_model = True


class EvaluateConfig(ConfigBase):
    def __init__(self):
        self.game_num = 15  # 400
        self.replace_rate = 0.55
        self.play_config = PlayConfig()
        self.play_config.simulation_num_per_move = 400
        self.play_config.thinking_loop = 1
        self.play_config.change_tau_turn = 0
        self.play_config.noise_eps = 0
        self.play_config.disable_resignation_rate = 0
        self.evaluate_latest_first = True


class PlayDataConfig(ConfigBase):
    def __init__(self):
        # Max Training Data Size = nb_game_in_file * max_file_num * 8
        self.multi_process_num = 16
        self.nb_game_in_file = 2
        self.max_file_num = 800
        self.save_policy_of_tau_1 = True


class PlayConfig(ConfigBase):
    def __init__(self):
        self.simulation_num_per_move = 200
        self.share_mtcs_info_in_self_play = True
        self.reset_mtcs_info_per_game = 1
        self.thinking_loop = 3
        self.required_visit_to_decide_action = 400
        self.start_rethinking_turn = 8
        self.noise_eps = 0.25
        self.dirichlet_alpha = 0.5
        self.change_tau_turn = 4
        self.virtual_loss = 3
        self.prediction_queue_size = 16
        self.parallel_search_num = 8
        self.prediction_worker_sleep_sec  = 0.0001
        self.wait_for_expanding_sleep_sec = 0.00001
        self.resign_threshold = -0.9
        self.allowed_resign_turn = 20
        self.policy_decay_turn = 60  # not used
        self.policy_decay_power = 3

        # Using a solver is a kind of cheating!
        self.use_solver_turn = 50
        self.use_solver_turn_in_simulation = 50

        #
        self.schedule_of_simulation_num_per_move = [
            (0, 8),
            (300, 50),
            (2000, 200),
        ]

        # True means evaluating 'AlphaZero' method (disable 'eval' worker).
        # Please change to False if you want to evaluate 'AlphaGo Zero' method.
        self.use_newest_next_generation_model = False


class TrainerConfig(ConfigBase):
    def __init__(self):
        self.wait_after_save_model_ratio = 1  # wait after saving model
        self.batch_size = 256  # 2048
        self.min_data_size_to_learn = 200
        self.epoch_to_checkpoint = 10
        self.start_total_steps = 0
        self.save_model_steps = 200
        self.use_tensorboard = True
        self.logging_per_steps = 100
        self.delete_self_play_after_number_of_training = 0  # control ratio of train:self data.
        self.lr_schedules = [
            (0, 0.0001),
            (150000, 0.001),
            (300000, 0.0001),
            (400000, 1e-3),
            (600000, 1e-4),
        ]


class ModelConfig(ConfigBase):
    def __init__(self):
        self.cnn_filter_num = 256
        self.cnn_filter_size = 3
        self.res_layer_num = 10
        self.l2_reg = 1e-4
        self.value_fc_size = 256
from logging import StreamHandler, basicConfig, DEBUG, getLogger, Formatter

def setup_logger(log_filename):
    format_str = '%(asctime)s@%(name)s %(levelname)s # %(message)s'
    basicConfig(filename=log_filename, level=DEBUG, format=format_str)
    stream_handler = StreamHandler()
    stream_handler.setFormatter(Formatter(format_str))
    getLogger().addHandler(stream_handler)

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cmd", help="what to do", default='play_gui', choices=['self_play', 'optimize', 'make_best', 'play_gui'])
    parser.add_argument("--config", help="specify config yaml", dest="config_file")
    parser.add_argument("--new", help="run from new best model", action="store_true")
    parser.add_argument("--total_step", help="set TrainerConfig.start_total_steps", default=0, type=int)
    parser.add_argument("--no_cuda", help="no cuda", action="store_false", default=True, dest='cuda')
    return parser.parse_args()

def get_everything():
    # get args
    args = create_parser()
    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')

    # get config
    if args.config_file:
        with open(args.config_file, "rt") as f:
            config = create_config(Config, yaml.load(f))
    else:
        config = create_config(Config)

    # update config
    config.opts.new = args.new
    config.opts.device = device
    config.opts.do = args.cmd
    config.trainer.start_total_steps = args.total_step
    config.resource.create_directories()
    setup_logger(config.resource.main_log_path)
    return config