from functions import *


def choice(string):
    """ Implementation menu of choices """
    key = input('Type 16 characters HEX key: ')
    n = int(input('Amount of rounds(Press Enter to leave default 16(It is maximum)): ') or 16)
    try:
        mod = int(input('Select mod( 1 - Encode, 0 - Decode ): '))
        if mod == 1:
            k = int(input('Show graphics of avalanche effect?(1 - yes, 0 - no): ') or 0)
            if k == 1:
                bit = int(input('What bit you want to change to opposite?(1-64): ') or 1)
                if 1 <= bit <= 64:
                    try:
                        encode_with_plots(string, key, n, bit)
                    except ValueError:
                        print('Bad key')
                else:
                    print('Range is only from 1 to 64!')
                return
            elif k == 0:
                try:
                    encode(string, key, n)
                except ValueError:
                    print('Bad key')
                except IndexError:
                    print('Bad key')
                return
            else:
                print('Kidding me? Choose mod!')
            return
        elif mod == 0:
            decode(string, key, n)
            return
        else:
            return print('Choose existing mode(Encode or Decode)!')
    except ValueError:
        print('Choose existing mode(Encode or Decode) or do not try to decode readable text!')


if __name__ == '__main__':
    while True:
        choice(input('Input text: '))
