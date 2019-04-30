from _asyncio import Future
from asyncio.queues import Queue
from collections import defaultdict, namedtuple
from logging import getLogger
import asyncio

import numpy as np
import time

from args import Config
from env.env import OthelloEnv, Stone, Result, switch_sides
from lib.board import find_correct_moves, bit_to_array, dirichlet_noise_of_mask, flip_and_rotate_right, bit_count
from lib.board import flip_and_rotate_result, flip_and_rotate_board_to_array
from lib.search import OthelloSolver



class QItem:
    def __init__(self, state, future):
        self.state = state
        self.future = future

class LastAcNQ:
    def __init__(self, action, policy, values, visit, enemy_values, enemy_visit):
        self.action = action
        self.policy = policy
        self.values = values
        self.visit = visit
        self.enemy_values = enemy_values
        self.enemy_visit = enemy_visit

class LastAva:
    def __init__(self, current, next):
        self.current = current
        self.next = next

class Trees:
    def __init__(self, num_tree, win_tree, policy_tree):
        self.num_tree = num_tree
        self.win_tree = win_tree
        self.policy_tree = policy_tree

def createTrees():
    tree = Trees(defaultdict(lambda: np.zeros((64,))),
                    defaultdict(lambda: np.zeros((64,))),
                    defaultdict(lambda: np.zeros((64,))))
    return tree.num_tree, tree.win_tree, tree.policy_tree

class AcNQ:
    def __init__(self, action, n, q):
        self.action = action
        self.n = n
        self.q = q

TreeNode = namedtuple("TreeNode", ['black', 'white', 'next_to_play'])
def create_node(env: OthelloEnv):
    return TreeNode(env.chessboard.black, env.chessboard.white, env.next_to_play.value)

def create_another_side_node(env: OthelloEnv):
    return TreeNode(env.chessboard.white, env.chessboard.black, switch_sides(env.next_to_play).value)

def create_another_side_node_from_node(node:TreeNode):
    return TreeNode(node.white, node.black, switch_sides(node.next_to_play).value)

def create_both_nodes(env: OthelloEnv):
    return create_node(env), create_another_side_node(env)

def update_num_tree_with_one_or_moresides(tree:defaultdict, node, action, do, value):
    nodes = [node, create_another_side_node_from_node(node)] if len(do)==2 else [node]
    for d,v,n in zip(do, value, nodes):
        if d=='plus':
            tree[n][action] += v
        elif d=='set':
            tree[n][action] = v
        
def update_win_tree_with_one_or_moresides(tree:defaultdict, node, action, do, value):
    nodes = [node, create_another_side_node_from_node(node)] if len(do) == 2 else [node]
    for d, v, n in zip(do, value, nodes):
        if d == 'plus':
            tree[n][action] += v
        elif d == 'set':
            tree[n][action] = v
        elif d == 'minus':
            tree[n][action] -= v
        
def update_policy_tree_with_one_or_moresides(tree:defaultdict, node, do, value):
    nodes = [node, create_another_side_node_from_node(node)] if len(do) == 2 else [node]
    for d, v, n in zip(do, value, nodes):
        if d == 'plus':
            tree[n] += v
        elif d == 'set':
            tree[n] = v
        elif d == 'minus':
            tree[n] -= v

def normalize(p, t=1):
    pp = np.power(p, t)
    return pp / np.sum(pp)

logger = getLogger(__name__)
map = {}
i = 0

class OthelloPlayer:
    def __init__(self, config: Config, client, mode="gui", weight_table=0, c=10, mc=False):
        """
        :param config:
        :param agent.model.OthelloModel|None model:
        :param TreeNode mtcs_info:
        :parameter OthelloModelAPI api:
        """
        self.config = config
        self.client = client
        self.mode = mode
        self.play_config = self.config.play
        self.weight_table = weight_table
        self.c = c
        self.mc = mc

        # mc_tree
        self.num_tree, self.win_tree, self.policy_tree = createTrees()

        # expanded
        self.expanded = set() #expanded存p（dict）的set形式
        self.now_expanding = set()

        # threads
        self.prediction_queue = Queue(self.play_config.prediction_queue_size) #并行计算的信息队列queue大小
        self.sem = asyncio.Semaphore(self.play_config.parallel_search_num)  #限制并行搜索的线程数
        self.loop = asyncio.get_event_loop()

        # for gui
        if self.mode == 'gui':
            self.thinking_history = None  # for fun
            self.avalable = None
            self.allow_resign = False
        elif self.mode == 'self_play':
            self.moves = []
            self.allow_resign = True
        self.test_mode=False
        # params
        self.running_simulation_num = 0

        # solver
        self.solver = OthelloSolver()  #引入minmax树类

    def win_rate(self, node):
        return self.win_tree[node] / (self.num_tree[node] + 1e-5)

# think_and_play
    def think_and_play(self, own, enemy):
        """play tmd：方案：50步以前使用深度學習mctree，若tree到達50步深度后再用minmaxtree； 50步以後直接用minmaxtree
        若搜不到/超時再用之前構建的樹"""
        # renew env
        self.start_time = time.time()
        env = OthelloEnv().update(own, enemy, next_to_play=Stone.black)
        node = create_node(env)

        #五十步之后直接minmax树搜索，若搜索不到，再用深度學習
        if env.epoch >= self.play_config.use_solver_turn:
            logger.warning(f"Entering minmax_tree process")
            ret = self._solver(node)
            if ret:  # not save move as play data
                return ret
        else: # 五十步之前直接用深度學習
            for t1 in range(self.play_config.thinking_loop):# search moves for 3 times
                logger.warning(f"Entering {t1} thinking_loop")
                self._expand_tree(env, node)
                policy, action, value_diff = self._calc_policy_and_action(node)
                # if action 足够大 + n足够大 \ turn 很小
                if env.epoch <= self.play_config.start_rethinking_turn or \
                        (value_diff > -0.01 and self.num_tree[node][action] >= self.play_config.required_visit_to_decide_action):
                    break

            # record or return
            if self.mode == 'gui':
                self._update_thinking_history(own, enemy, action, policy)
                self._update_avalable(own, enemy, action, policy)
            elif self.mode == 'self_play':
                if self.allow_resign:# resign win_rate 太小没有胜率。
                    if self.play_config.resign_threshold is not None and\
                        np.max(self.win_rate(node)-(self.num_tree[node]==0)*10) <= self.play_config.resign_threshold:
                        if env.epoch >= self.config.play.allowed_resign_turn:
                            return AcNQ(None, 0, 0)  # means resign
                        else:
                            logger.debug(f"Want to resign but disallowed turn {env.epoch} < {self.config.play.allowed_resign_turn}")
                # save fuckers
                saved_policy = self.__calc_policy_by_prob(node) if self.config.play_data.save_policy_of_tau_1 else policy
                self.__save_data_to_moves(own, enemy, saved_policy)
            return AcNQ(action=action, n=self.num_tree[node][action], q=self.win_rate(node)[action])


    def _solver(self, node):
        # use solver to do minmax搜索
        action, point = self.solver.solve(node.black, node.white, Stone(node.next_to_play), exactly=True)
        if action is None: #如果沒搜索到是不可以返回None的因爲要
            return None
        else:
            policy = np.zeros(64)
            policy[action] = 1
            update_num_tree_with_one_or_moresides(self.num_tree, node, action, ["set"], [999])
            update_win_tree_with_one_or_moresides(self.win_tree, node, action, ["set"], [np.sign(point)*999])
            update_policy_tree_with_one_or_moresides(self.policy_tree, node, ["set"], [policy])
            self._update_thinking_history(node.black, node.white, action, policy)
        return AcNQ(action=action, n=999, q=np.sign(point))


    def _expand_tree(self, env, node):
        if env.epoch > 0:  # 对树进行拓展
            self._expand_tree_2(env.chessboard.black, env.chessboard.white)
        else:
            self._set_first_move(node)


    def _expand_tree_2(self, own, enemy):
        # params
        loop = self.loop
        self.running_simulation_num = 0
        # n simulation/move
        coroutine_list = []
        # 200 simulations
        for it in range(self.play_config.simulation_num_per_move):
            coroutine_list.append(self.__start_search_my_move(own, enemy))
        coroutine_list.append(self.__prediction_worker())
        loop.run_until_complete(asyncio.gather(*coroutine_list))

    async def __start_search_my_move(self, own, enemy):
        # set parmas
        self.running_simulation_num += 1
        # wait sems
        with await self.sem:  # 8綫程
            env = OthelloEnv().update(own, enemy, Stone.black)
            leaf_v = await self.___recursive_simulation(env, is_root_node=True)
            self.running_simulation_num -= 1
            return leaf_v

    async def ___recursive_simulation(self, env: OthelloEnv, is_root_node=False):
        "fertilize tree process"
        # get both keys
        node, another_side_node = create_both_nodes(env)
        if self.test_mode:
            if (node not in map.keys()):
                map[node] = env.epoch

        # return condition 1
        if env.done:
            if env.result == Result.black:
                return 1
            elif env.result == Result.white:
                return -1
            else:
                return 0

        # return condition 2 : get solver（大于50步，minmax）
        if env.epoch >= self.config.play.use_solver_turn_in_simulation:
            action, point = self.solver.solve(node.black, node.white, Stone(node.next_to_play), exactly=False)
            if action:
                point = point if env.next_to_play == Stone.black else -point
                leaf_v = np.sign(point)
                leaf_p = np.zeros(64)
                leaf_p[action] = 1
                # update tree
                update_num_tree_with_one_or_moresides(self.num_tree, node, action, ["plus", "plus"], [1, 1])#走过的位置+1
                update_win_tree_with_one_or_moresides(self.win_tree, node, action, ["plus", "minus"],[leaf_v, leaf_v])#走此步赢的次数+-1（win）
                update_policy_tree_with_one_or_moresides(self.policy_tree, node, ["set", "set"], [leaf_p, leaf_p])#此节点应该走的位置（position）
                return np.sign(point)
            if time.time()-self.start_time >= 55:
                return 0
        #return condition 3 : expand tree（小於等於50步，用深度學習）
        while node in self.now_expanding: # 兩個搜索綫程遇到同一個node，會有衝突的問題
            await asyncio.sleep(self.config.play.wait_for_expanding_sleep_sec)
        # is leaf
        if node not in self.expanded:  # reach leaf node
            leaf_v = await self.____expand_leaf_node(env)
            if env.next_to_play == Stone.black:
                return leaf_v  # Value for black
            else:
                return -leaf_v  # Value for white == -Value for black
        else: # not leaf do
            virtual_loss_for_w = self.config.play.virtual_loss if env.next_to_play == Stone.black else -self.config.play.virtual_loss
            action_t = self.____decide_action(env, is_root_node)  #UCB公式
            update_num_tree_with_one_or_moresides(self.num_tree, node, action_t, ["plus"], [self.config.play.virtual_loss])
            update_win_tree_with_one_or_moresides(self.win_tree, node, action_t, ["minus"], [virtual_loss_for_w])
            env.do(action_t)
            leaf_v = await self.___recursive_simulation(env)  # next move
            # on returning search path
            update_num_tree_with_one_or_moresides(self.num_tree, node, action_t, ["plus", "plus"], [-self.config.play.virtual_loss+1, 1])
            update_win_tree_with_one_or_moresides(self.win_tree, node, action_t, ["plus", "minus"], [virtual_loss_for_w+leaf_v, leaf_v])
            if self.test_mode:
                logger.warning(map[node], leaf_v)
        return leaf_v

    async def ____expand_leaf_node(self, env):
        "use to expand new leaf"
        node, another_side_node = create_both_nodes(env)
        self.now_expanding.add(node)

        # flip + rotate
        rotate_right_num, is_flip_vertical, black_ary, white_ary = flip_and_rotate_board_to_array(env.chessboard.black, env.chessboard.white)

        # predict
        state = [white_ary, black_ary] if env.next_to_play == Stone.white else [black_ary, white_ary]
        future = await self.predict(np.array(state))  # type: Future
        await future
        leaf_p, leaf_v = future.result()

        # reverse rotate and flip about leaf_p
        leaf_p = flip_and_rotate_result(leaf_p, rotate_right_num, is_flip_vertical)
        if self.mc:
            black = env.chessboard.black
            leaf_v += np.sum(bit_to_array(black,64)*self.weight_table)

        # update
        update_policy_tree_with_one_or_moresides(self.policy_tree, node, ["set", "set"], [leaf_p, leaf_p])
        self.expanded.add(node)
        self.now_expanding.remove(node)
        return leaf_v

    def ____decide_action(self, env, is_root_node):
        # find correct moves
        node = create_node(env)
        legal_moves = find_correct_moves(node.black, node.white) if env.next_to_play == Stone.black else find_correct_moves(node.white, node.black)

        # vn = formula here
        vn = max(np.sqrt(np.sum(self.num_tree[node])), 1)  # SQRT of sum(N(s, b); for all b)

        # p = formula here  re-normalize in legal moves
        vp = self.policy_tree[node]
        vp = vp * bit_to_array(legal_moves, 64)
        temperature = 1
        if np.sum(vp) > 0:
            temperature = min(np.exp(1 - np.power(env.epoch / self.config.play.policy_decay_turn, self.config.play.policy_decay_power)), 1)
            vp = normalize(vp, temperature)
        # add noise 0.75*p + 0.25*noise  
        if is_root_node and self.play_config.noise_eps > 0:  # Is it correct?? -> (1-e)p + e*Dir(alpha)
            noise = dirichlet_noise_of_mask(legal_moves, self.play_config.dirichlet_alpha)
            vp = (1 - self.play_config.noise_eps) * vp + self.play_config.noise_eps * noise

        # u_ = formula here
        vpn = vp * vn / (1 + self.num_tree[node])
        if env.next_to_play == Stone.black:
            vpn_with_weight = (self.win_rate(node)*self.c + vpn + 1000 + self.weight_table) * bit_to_array(legal_moves, 64)
        else:
            vpn_with_weight = (-self.win_rate(node)*self.c + vpn + 1000 + self.weight_table) * bit_to_array(legal_moves, 64)
        action_t = int(np.argmax(vpn_with_weight))
        return action_t


    async def __prediction_worker(self):
        " do prediction in this worker"
        margin = 10  # wait for at most 10 epochs x 0.0001
        while self.running_simulation_num > 0 or margin > 0:
            if self.prediction_queue.empty():
                if margin > 0:
                    margin -= 1
                await asyncio.sleep(self.config.play.prediction_worker_sleep_sec)
                continue
            item_list = [self.prediction_queue.get_nowait() for _ in range(self.prediction_queue.qsize())]  # type: list[QItem]
            data = np.array([x.state for x in item_list])
            policy_ary, value_ary = self.client.forward(data)  # shape=(N, 2, 8, 8)
            for p, v, item in zip(policy_ary, value_ary, item_list):
                item.future.set_result((p, v))

    def _set_first_move(self, node):
        # chose the random num_tree = [1] policy_tree = [每个可能的地方都是1/n]
        legal_array = bit_to_array(find_correct_moves(node.black, node.white), 64)
        action = np.argmax(legal_array)
        update_num_tree_with_one_or_moresides(self.num_tree, node, action, ["set"], [1])
        update_win_tree_with_one_or_moresides(self.win_tree, node, action, ["set"], [0])
        update_policy_tree_with_one_or_moresides(self.policy_tree, node, ["set"], [legal_array/np.sum(legal_array)])

    def _calc_policy_and_action(self, node):
        policy = self._calc_policy(node.black, node.white) # 先验 最大的n,  前四步p[n]=num_tree[key][n]，后面p矩阵只有var_numb最大位置为1，其余为0.
        action = int(np.random.choice(range(64), p=policy))   #随机走一个点，p为随机取的各点概率（先验）
        action_by_value = int(np.argmax(self.win_rate(node) + (self.num_tree[node] > 0)*100)) #选走过的、q（胜率）最大的那个位置
        value_diff = self.win_rate(node)[action] - self.win_rate(node)[action_by_value] #
        return policy, action, value_diff

    def _calc_policy(self, own, enemy):
        env = OthelloEnv().update(own, enemy, Stone.black)
        node = create_node(env)
        # if turn < 4
        if env.epoch < self.play_config.change_tau_turn:
            return self.__calc_policy_by_prob(node) # p value
        else:
            return self.__calc_policy_by_max(node)

    def __calc_policy_by_prob(self, node):
        return self.num_tree[node] / np.sum(self.num_tree[node])  # tau = 1

    def __calc_policy_by_max(self, node):
        action = np.argmax(self.num_tree[node])  # tau = 0
        ret = np.zeros(64)  # one hot
        ret[action] = 1
        return ret

    def _update_thinking_history(self, black, white, action, policy):
        node = TreeNode(black, white, Stone.black.value)
        next_key = self.__get_next_key(black, white, action)
        self.thinking_history = \
            LastAcNQ(action, policy, list(self.win_rate(node)), list(self.num_tree[node]),
                        list(self.win_rate(next_key)), list(self.num_tree[next_key]))

    def _update_avalable(self, black, white, action, policy):
        node = TreeNode(black, white, Stone.black.value)
        next_key = self.__get_next_key(black, white, action)
        self.avalable = LastAva(find_correct_moves(node.black, node.white),
                                                   find_correct_moves(next_key.white, next_key.black))

    def __get_next_key(self, own, enemy, action):
        env = OthelloEnv().update(own, enemy, Stone.black)
        env.do(action)
        return create_node(env)

    def __save_data_to_moves(self, own, enemy, policy):
        for flip in [False, True]:
            for rot_right in range(4):
                self.moves.append(flip_and_rotate_right(flip, rot_right, own, enemy, policy))

    async def predict(self, x):
        future = self.loop.create_future()
        await self.prediction_queue.put(QItem(x, future))
        return future



