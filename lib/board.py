import numpy as np
from random import random
#参考 http://primenumber.hatenadiary.jp/entry/2016/12/26/063226
#参考 https://primenumber.hatenadiary.jp/entry/2016/12/03/203823


def find_correct_moves(own, enemy):
    left_right_mask = 0x7e7e7e7e7e7e7e7e  # Both most left-right edge are 0, else 1
    top_bottom_mask = 0x00ffffffffffff00  # Both most top-bottom edge are 0, else 1
    mask = left_right_mask & top_bottom_mask
    masks = [left_right_mask, mask, top_bottom_mask, mask, left_right_mask, mask, top_bottom_mask, mask]
    funcs = [_search_offset_left]*4 + [_search_offset_right]*4
    offsets = [1, 9, 8, 7 ,1, 9, 8, 7]
    mobility = 0
    for m,o,f in zip(masks, offsets, funcs):
        mobility |= f(own, enemy, m, o)  # Left
    return mobility

def _search_offset_left(own, enemy, mask, offset):
    e = enemy & mask # enemy and
    t = e & (own >> offset)
    for i in range(5):
        t |= e & (t >> offset)    # Up to six stones can be turned at once
    return ~(own | enemy) & (t >> offset)  # Only the blank squares can be started

def _search_offset_right(own, enemy, mask, offset):
    e = enemy & mask              # enemy with mask
    t = e & (own << offset)
    for i in range(5):
        t |= e & (t << offset)    # Up to six stones can be turned at once
    return ~(own | enemy) & (t << offset)  # Only the blank squares can be started

def calc_flip(pos, own, enemy):
    assert 0 <= pos <= 63, f"pos={pos}"
    f1 = _calc_flip_half(pos, own, enemy)
    f2 = _calc_flip_half(63 - pos, rotate180(own),rotate180(enemy))
    return f1 | rotate180(f2)

def _calc_flip_half(pos, own, enemy):
    el = [enemy, enemy & 0x7e7e7e7e7e7e7e7e, enemy & 0x7e7e7e7e7e7e7e7e, enemy & 0x7e7e7e7e7e7e7e7e]
    masks = [b64(0x0101010101010100 << pos), b64(0x00000000000000fe << pos), b64(0x0002040810204080 << pos), b64(0x8040201008040200 << pos)]
    flipped = 0
    for e, mask in zip(el, masks):
        outflank = mask & ((e | ~mask) + 1) & own
        flipped |= (outflank - (outflank != 0)) & mask
    return flipped

# 上下翻转
def flip_vertical(x):
    k1 = 0x00FF00FF00FF00FF
    k2 = 0x0000FFFF0000FFFF
    x = ((x >> 8) & k1) | ((x & k1) << 8)
    x = ((x >> 16) & k2) | ((x & k2) << 16)
    x = (x >> 32) | b64(x << 32)
    return x


# 右对角线翻转
def flip_diag_a1h8(x):
    k1 = 0x5500550055005500
    k2 = 0x3333000033330000
    k4 = 0x0f0f0f0f00000000
    masks = [k4, k2, k1]
    offsets = [28, 14 ,7]
    for m,o in zip(masks, offsets):
        t = m & (x ^ b64(x<<o))
        x ^= t ^ (t >> o)
    return x

# 向右翻转90
def rotate90(x):
    return flip_diag_a1h8(flip_vertical(x))

# 向右翻转180
def rotate180(x):
    return rotate90(rotate90(x))

def dirichlet_noise_of_mask(mask, alpha):
    num_1 = bit_count(mask)
    noise = list(np.random.dirichlet([alpha] * num_1))
    ret_list = []
    for i in range(64):
        if (1 << i) & mask:
            ret_list.append(noise.pop(0))
        else:
            ret_list.append(0)
    return np.array(ret_list)

def b64(x):
    return x & 0xFFFFFFFFFFFFFFFF


def bit_count(x):
    return bin(x).count('1')


def bit_to_array(x, size):
    """bit_to_array(0b0010, 4) -> array([0, 1, 0, 0])"""
    return np.array(list(reversed((("0" * size) + bin(x)[2:])[-size:])), dtype=np.uint8)


def flip_and_rotate_result(leaf_p, rotate_right_num, is_flip_vertical):
    if rotate_right_num > 0 or is_flip_vertical:  # reverse rotation and flip. rot -> flip.
        leaf_p = leaf_p.reshape((8, 8))
        if rotate_right_num > 0:
            leaf_p = np.rot90(leaf_p, k=rotate_right_num)  # rot90: rotate matrix LEFT k times
        if is_flip_vertical:
            leaf_p = np.flipud(leaf_p)
        leaf_p = leaf_p.reshape((64,))
    return leaf_p

def flip_and_rotate_board_to_array(black, white):
    is_flip_vertical = random() < 0.5
    rotate_right_num = int(random() * 4)
    if is_flip_vertical:
        black, white = flip_vertical(black), flip_vertical(white)
    for i in range(rotate_right_num):
        black, white = rotate90(black), rotate90(white)  # rotate90: rotate bitboard RIGHT 1 time
    return rotate_right_num, is_flip_vertical, bit_to_array(black, 64).reshape((8, 8)), bit_to_array(white, 64).reshape((8, 8))

def flip_and_rotate_right(flip, rot_right, own, enemy, policy):
    own_saved, enemy_saved, policy_saved = own, enemy, policy.reshape((8, 8))
    if flip:
        own_saved = flip_vertical(own_saved)
        enemy_saved = flip_vertical(enemy_saved)
        policy_saved = np.flipud(policy_saved)
    if rot_right:
        for _ in range(rot_right):
            own_saved = rotate90(own_saved)
            enemy_saved = rotate90(enemy_saved)
        policy_saved = np.rot90(policy_saved, k=-rot_right)
        # [ (64b, 64b),[64] +-1]
    return [(own_saved, enemy_saved), list(policy_saved.reshape((64, )))]

if __name__ == '__main__':
    black = (0b00100000 << 24 | 0b00001000 << 32)
    white = (0b00001000 << 24 | 0b00010000 << 32)
    print(bit_to_array(black, 64).reshape((8, 8)))