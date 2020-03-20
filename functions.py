import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tables
from textwrap import wrap
import binascii


def permute(block, box):
    """permutes sequence according to permutation tables"""
    return ''.join([block[i] for i in box])


def rotate_left(block, i):
    """Makes left shifts according to shifts table"""
    return bin(int(block, 2) << i & 0x0fffffff | int(block, 2) >> 28 - i)[2:].zfill(28)


def input_text(string):
    """Accepts input text and converts it to a hexadecimal list of 8 bytes for each object(because of UTF-8 encoding)"""
    lalis = [i.zfill(16) for i in wrap(''.join(hex(int(string.encode().hex(), 16))[2:]), 16)]
    return lalis


def xor(arg_1, arg_2):
    """Implementation XOR"""
    return ''.join([str(int(i) ^ int(j)) for i, j in zip(arg_1, arg_2)])


def hex_to_binary(s):
    """Converting numbers from hexadecimal to binary"""
    return ''.join([bin(int(i, 16))[2:].zfill(4) for i in s])


def f(block, key):
    """Implementation of F function"""
    output = []
    for j, i in enumerate(wrap(xor(permute(block, tables.permutation_w_expansion), key), 6)):
        temp_box = [
            tables.s_blocks[j][0:16],
            tables.s_blocks[j][16:32],
            tables.s_blocks[j][32:48],
            tables.s_blocks[j][48:64]
        ]
        # print(temp_box)
        output.append(bin(temp_box[int(i[0] + i[-1], 2)][int(i[1:-1], 2)])[2:].zfill(4))

    return permute(''.join(output), tables.permutation_p_block)


def keygen(hex_key, n):
    """generating 16 keys"""
    # convert hexadecimal key to binary
    hex_to_bin = ''.join([bin(int(i, 16))[2:].zfill(4) for i in hex_key])
    # first permutation
    perm = permute(hex_to_bin, tables.permuted_choice_1)
    # Divide the 64 bit key into two parts
    leftblock = perm[:len(perm) // 2]
    rightblock = perm[len(perm) // 2:]
    # generating of keys for 16 rounds, each of which is 48 bits with second permutation
    li = []
    for i in tables.rotates:
        leftblock = rotate_left(leftblock, i)
        rightblock = rotate_left(rightblock, i)
        li.append(permute(leftblock + rightblock, tables.permuted_choice_2))
    return li[0:n]


def des(block, key, n):
    """implementing DES engine"""
    left, right = block[0: len(block) // 2], block[len(block) // 2:]
    for j, i in zip(range(1, n+1), key):
        right, left = xor(f(right, i), left), right
    return wrap(permute(right + left, tables.inv_permutation), 8)


def des_with_avalance_check(block, key, n, bit):
    """ Implementing DES engine for avalanche effect check """
    change_bit = list(block)
    print('Imported bits:           ', ''.join(change_bit))
    if change_bit[bit-1] == '1':
        change_bit[bit-1] = '0'
        change_bit = ''.join(change_bit)
        print('Changed bit from 1 to 0: ', change_bit)
    else:
        change_bit[bit-1] = '1'
        change_bit = ''.join(change_bit)
        print('Changed bit from 0 to 1: ', change_bit)

    lef1, righ1 = change_bit[0: len(block) // 2], block[len(block) // 2:]
    left, right = block[0: len(block) // 2], block[len(block) // 2:]

    normal_list = []
    another_bit_list = []

    for j, i in zip(range(1, n + 1), key):
        right, left = xor(f(right, i), left), right
        normal_list.append(right + left)

    encode_list = (wrap(permute(right + left, tables.inv_permutation), 8))

    for j, i in zip(range(1, n + 1), key):
        righ1, lef1 = xor(f(righ1, i), lef1), righ1
        another_bit_list.append(righ1 + lef1)

    # print(normal_list)
    # print(another_bit_list)
    not_similarity = []
    rounds = []
    rround = 0
    for i, j in zip(normal_list, another_bit_list):
        sim = 0
        for k, l in zip(i, j):
            if k != l:
                sim += 1
        rround += 1
        rounds.append(str(rround))
        not_similarity.append(sim)
    # print('Round :', rounds)
    # print('Count of different bits: ', not_similarity)
    return encode_list, rounds, not_similarity


def plot_build(rounds, not_similarity):
    """ Avalanche chart construction """
    rounds_list = []
    diff_list = []
    for t, h in zip(rounds, not_similarity):
        for k, l in zip(t, h):
            rounds_list.append(k)
            diff_list.append(l)
    fig, ax = plt.subplots()
    x = np.arange(len(rounds_list))
    width = 0.5
    yay = ax.bar(x - width / 2, diff_list, width, label='Different bits')
    ax.set_ylabel('Count of different bits by round')
    ax.set_xlabel('Rounds')
    ax.set_title('Avalanche effect after changing one bit')
    ax.set_xticks(x)
    ax.set_xticklabels(rounds_list)
    ax.legend()
    for var in yay:
        height = var.get_height()
        ax.annotate('{}'.format(height),
                    xy=(var.get_x() + var.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    fig.tight_layout()
    plt.show()


def encode_with_plots(string, hex_key, n, bit):
    """ Encrypting imported string to hex cipher"""
    encrypted_list = []
    list_of_rounds = []
    list_of_non_simil = []
    for i in input_text(string):
        inp_text = permute(hex_to_binary(i), tables.init_permutation)
        encode_list, rounds, not_similarity = des_with_avalance_check(inp_text, keygen(hex_key, n), n, bit)
        # print('This is solo', encode_list)
        # print('This is rounds: ', rounds)
        # print('This is not similarity: ', not_similarity)
        encrypted_list.append(''.join([hex(int(i, 2))[2:].zfill(2).upper() for i in encode_list]))
        list_of_rounds.append(rounds)
        list_of_non_simil.append(not_similarity)
    print(''.join(encrypted_list))
    # print('this is list of rounds', list_of_rounds)
    # print('this is list of non similarity', list_of_non_simil)
    encrypted = ''.join(encrypted_list)
    plot_build(list_of_rounds, list_of_non_simil)
    return encrypted


def encode(string, hex_key, n):
    """ Encrypting imported string to hex cipher"""
    encrypted_list = []
    for i in input_text(string):
        inp_text = permute(hex_to_binary(i), tables.init_permutation)
        dee = des(inp_text, keygen(hex_key, n), n)
        # print('this is dee:', dee)
        encrypted_list.append(''.join([hex(int(i, 2))[2:].zfill(2).upper() for i in dee]))
    print(''.join(encrypted_list))
    return ''.join(encrypted_list)


def decode(block, key, n):
    """ Decrypting imported hex cipher to string """
    decrypted = []
    kur_list = []
    for i in wrap(block, 16):
        bin_block = hex_to_binary(i).zfill(64)
        permute_input = permute(bin_block, tables.init_permutation)
        des1 = des(permute_input, reversed(keygen(key, n)), n)
        decrypted.append(''.join([hex(int(i, 2))[2:].zfill(2).upper() for i in des1]))
    [[kur_list.append(j) for j in wrap(i, 2) if int(j, 16) != 0] for i in decrypted]
    per = ''.join(kur_list)
    dec = ''.join(binascii.unhexlify(per).decode('utf-8'))
    print(dec)
    return dec
