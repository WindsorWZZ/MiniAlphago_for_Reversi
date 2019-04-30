from time import time

from logging import getLogger

from env.env import OthelloEnv, Stone
from lib.board import find_correct_moves
import numpy as np

logger = getLogger(__name__)


class Timeout(Exception):
    pass


class OthelloSolver:
    """ min-max 树搜索"""
    def __init__(self):
        self.cache = {}
        self.timeout = 30
        self.last_is_exactly = False


    def solve(self, black, white, next_to_play, exactly=False):
        # set stuff
        self.start_time = time()
        if not self.last_is_exactly and exactly:  # exactly时候要注意去掉上次的cache
            self.cache = {}
        self.last_is_exactly = exactly

        # try searching process
        try:
            move, score = self._find_winning_move_and_score(OthelloEnv().update(black, white, next_to_play),
                                                           exactly=exactly)
            return move, score if next_to_play == Stone.black else -score
        except Timeout:
            return None, None


    def _find_winning_move_and_score(self, env: OthelloEnv, exactly=True):
        # end
        if env.done:
            b, w = env.chessboard.black_white
            return None, b - w

        # restored
        key = black, white, next_to_play = env.chessboard.black, env.chessboard.white, env.next_to_play
        if key in self.cache: # store leaf node
            return self.cache[key]

        # timeout
        if time() - self.start_time > self.timeout:
            logger.debug("timeout!")
            raise Timeout()

        # recursive
        legal_moves = find_correct_moves(*(white, black) if not next_to_play == Stone.black else (black, white))
        action_list = [idx for idx in range(64) if legal_moves & (1 << idx)] # 遍历所有解
        score_list = np.zeros(len(action_list), dtype=int)
        record_turn = env.epoch
        for i, action in enumerate(action_list):
            env.chessboard.black = black
            env.chessboard.white = white
            env.next_to_play = next_to_play
            env.epoch = record_turn
            env.done = False
            env.Result = None
            env.do(action)
            _, score = self._find_winning_move_and_score(env, exactly=exactly)
            score_list[i] = score

            if not exactly:
                if next_to_play == Stone.black and score > 0:#  找到一个就得
                    break
                elif next_to_play == Stone.white and score < 0:
                    break

        best_action, best_score = (action_list[int(np.argmax(score_list))], np.max(score_list)) if next_to_play == Stone.black else (action_list[int(np.argmin(score_list))], np.min(score_list))
        self.cache[key] = (best_action, best_score)
        return best_action, best_score




