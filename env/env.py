import enum

from logging import getLogger

from lib.board import calc_flip, bit_count, find_correct_moves

# logger
logger = getLogger(__name__)
class Stone(enum.Enum):
    black = 1
    white = 2
class Result(enum.Enum):
    black = 1
    white = 2
    draw = 3

def switch_sides(player: Stone):
    """switch player player"""
    if type(player)==int:
        return Stone.white if player == 1 else Stone.black
    else:
        return Stone.white if player == Stone.black else Stone.black


class OthelloEnv:
    def __init__(self):
        pass

    def reset(self):
        self.chessboard = ChessBoard()
        self.next_to_play = Stone.black
        self.epoch = 0
        self.done = False
        self.result = None
        return self

    def update(self, black, white, next_to_play):
        self.chessboard = ChessBoard(black, white)
        self.next_to_play = next_to_play
        self.epoch = sum(self.chessboard.black_white) - 4
        self.done = False
        self.result = None
        return self

    def do(self, ac):
        """
        :param int|None action: move pos=0 ~ 63 (0=top left, 7 top right, 63 bottom right), None is resign
        :return:
        """
        # check error
        assert ac is None or 0 <= ac <= 63, f"Illegal ac={ac} {self.epoch}"

        # action None branch
        if ac is None:
            logger.warning(f"SITUATION: resigned {self.epoch}")
            self._other_win()
            return self.chessboard
        else:
            # own = next_move_color
            own, opp = (self.chessboard.black, self.chessboard.white) if self.next_to_play == Stone.black else (self.chessboard.white, self.chessboard.black)
            # flipped=after move own
            flipped = calc_flip(ac, own, opp)

            # if not flipped
            if bit_count(flipped) == 0:
                logger.warning(f"SITUATION: Illegal ac={ac}, No Flipped, Set {switch_sides(self.next_to_play)} win {self.epoch}")
                self._other_win()
                return self.chessboard
            else:
                # flip the board
                own, opp = self._do_flip(own, opp, ac, flipped)
                self.chessboard.black, self.chessboard.white = (own, opp) if self.next_to_play == Stone.black else (opp, own)
                # if there's still way to go
                if bit_count(find_correct_moves(opp, own)) > 0:  # there are legal moves for opp.
                    self.next_to_play = switch_sides(self.next_to_play)
                elif bit_count(find_correct_moves(own, opp)) > 0:  # there are legal moves for me but opp.
                    pass
                else:  # there is no legal moves for me and opp.
                    # logger.warning(f"SITUATION: won till game over {self.epoch}")
                    self._game_over()

        return self.chessboard

    def _do_flip(self, own, opp, ac, flipped):
        own ^= flipped
        opp ^= flipped
        own |= 1 << ac
        self.epoch += 1
        return own, opp

    def _other_win(self):
        # get out of the game when no moves of own
        self.__set_other_player_win()
        self.__set_game_over()
        self.__check_Result()

    def __set_other_player_win(self):
        # set self.winner = another player
        win_player = switch_sides(self.next_to_play)  # type: Stone
        self.Result = Result.black if win_player == Stone.black else Result.white

    def __set_game_over(self):
        # set self.done=True
        self.done = True

    def __check_Result(self):
        # correct winner
        black_num, white_num = self.chessboard.black_white
        if black_num > white_num:
            result = Result.black
        elif black_num < white_num:
            result = Result.white
        else:
            result = Result.draw

        # check Result
        if self.result is None:
            self.result = result


    def _game_over(self):
        # get out of the game when no moves of both
        self.__set_game_over()
        self.__check_Result()


class ChessBoard:
    """
    Board include 2 bit strings, each of which include
    a 64_length onehot coding of the 8x8 board.
    """
    def __init__(self, black=None, white=None):
        # init
        self.black, self.white = (0b10000<<24 | 0b1000<<32), (0b1000<<24 | 0b10000<<32)
        # self.black, self.white = (0b100000 << 24 | 0b1000 << 32), (0b1000 << 24 | 0b100000 << 32)
        # from lib.board import flip_vertical,flip_diag_a1h8,rotate180
        # self.black = flip_vertical(self.black)
        if black and white:
            self.black, self.white = black, white

    @property
    def black_white(self):
        """return count of black and white"""
        return bit_count(self.black), bit_count(self.white)
