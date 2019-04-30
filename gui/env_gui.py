import enum
from logging import getLogger
import os

from agent.player import LastAcNQ
from agent.player import OthelloPlayer
from args import Config
from env.env import Stone, OthelloEnv
from lib.board import find_correct_moves
from lib.data_helper import get_next_generation_model_dirs
from agent.model import OthelloModel
import time
import numpy as np
logger = getLogger(__name__)
WEIGHT_TABLE2= np.array([ [48,   6,  6,  6,  6,  6,   6, 48,],
           [ 6, -24, -4, -4, -4, -4, -24,  6,],
           [ 6,  -4,  1,  1,  1,  1,  -4,  6,],
           [ 6,  -4,  1,  1,  1,  1,  -4,  6,],
           [ 6,  -4,  1,  1,  1,  1,  -4,  6,],
           [ 6,  -4,  1,  1,  1,  1,  -4,  6,],
           [ 6, -24, -4, -4, -4, -4, -24,  6,],
           [48,   6,  6,  6,  6,  6,   6, 48,],]).reshape(64,)

WEIGHT_TABLE = np.array([[90,-60,10,10,10,10,-60,90],
                         [-60,-80,5,5,5,5,-80,-60],
                         [10,5,1,1,1,1,5,10],
                        [10,5,1,1,1,1,5,10],
                        [10,5,1,1,1,1,5,10],
                        [10,5,1,1,1,1,5,10],
                        [-60,-80,5,5,5,5,-80,-60],
                        [90,-60,10,10,10,10,-60,90]]).reshape(64,)

def _load_model(config: Config):
    # load
    model = OthelloModel(config)
    # update
    if config.play.use_newest_next_generation_model:
        pc = get_next_generation_model_dirs(config.resource)[-1]
        weight_path = os.path.join(pc, config.resource.next_generation_model_weight_filename)
        model.load(weight_path)
    else:
        model.load(config.resource.model_best_weight_path)
    return model

class EnvGui:
    def __init__(self, config: Config):
        self.config = config
        self.env = OthelloEnv().reset()
        self.ai = OthelloPlayer(self.config, _load_model(self.config), weight_table=WEIGHT_TABLE/3, c=20, mc=True)  # type: OthelloPlayer

        self.human_stone = None

        self.rev_function = None
        self.count_one_step = 0
        self.count_all_step = 0
        self.last_evaluation = None
        self.last_history = None  # type: LastAcNQ
        self.last_ava = None
        self.action = None

    def start_game(self, human_is_black):
        # set color and env
        self.__init__(self.config)
        self.human_stone = Stone.black if human_is_black else Stone.white

    def play_next_turn(self):
        # update + ai_move / over
        self._do_move(1)

        # do over
        if self.env.done:
            self._do_move(3)
            return

        # do ai_move
        if self.env.next_to_play != self.human_stone:
            self._do_move(2)

    def _do_move(self, event):
        self.rev_function[event]()

    def add_observer(self, ob_map):
        self.rev_function = ob_map

    def stone(self, px, py):
        """left top=(0, 0), right bottom=(7,7)"""
        action = int(py * 8 + px)
        if self.env.chessboard.black & (1 << action):
            return 2
        elif self.env.chessboard.white & (1 << action):
            return 1
        else:
            return 0

    def available(self, px, py):
        own, enemy = (self.env.chessboard.black, self.env.chessboard.white) if self.env.next_to_play == Stone.black else (self.env.chessboard.white, self.env.chessboard.black)
        action = int(py * 8 + px)
        if action < 0 or 64 <= action or (1<<action) & self.env.chessboard.black or (1<<action) & self.env.chessboard.white\
                or not (1<<action) & find_correct_moves(own, enemy):
            return False
        return 1

    def move(self, px, py):
        self.env.do(int(py * 8 + px))

    def move_by_ai(self):
        own, enemy = (self.env.chessboard.black, self.env.chessboard.white) if self.env.next_to_play == Stone.black else (self.env.chessboard.white, self.env.chessboard.black)
        start = time.time()
        self.action = self.ai.think_and_play(own, enemy).action
        end = time.time()
        self.count_one_step = end-start
        self.count_all_step += self.count_one_step
        self.env.do(self.action)

        # notations
        self.last_history = self.ai.thinking_history
        self.last_evaluation = self.last_history.values[self.last_history.action]
        self.last_ava = self.ai.avalable

