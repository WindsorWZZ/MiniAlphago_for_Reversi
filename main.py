import os
import sys
from logging import getLogger

from args import get_everything


# gogogo
def alpha_go(config):
    do = config.opts.do
    if do == "play_gui":
        from gui import gui
        return gui.start(config)
    elif do == 'optimize':
        from work import optimize
        return optimize.start(config)
    elif do == 'make_best':
        import make_best
        return make_best.start(config)
    # elif do == 'self_play':
    #     from work import self_play
    #     return self_play.start(config)


# add path
def add_path():
    _PATH_ = os.path.dirname(__file__)
    if _PATH_ not in sys.path:
        sys.path.append(_PATH_)

# add path
add_path()
# logger
logger = getLogger(__name__)
# args
config = get_everything()

if __name__=="__main__":
    # ok to go
    alpha_go(config)
