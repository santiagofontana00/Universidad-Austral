def label_to_int(string_label):
    if string_label == '0': return 1
    if string_label == '1': return 2
    if string_label == '2': return 3
    if string_label == '3': return 4
    if string_label == '4': return 5
    if string_label == '5': return 6
    if string_label == '6': return 7
    if string_label == '7': return 8
    if string_label == '8': return 9
    if string_label == '9': return 10
    if string_label == 'a': return 11
    if string_label == 'b': return 12
    if string_label == 'c': return 13
    if string_label == 'd': return 14
    if string_label == 'e': return 15
    if string_label == 'f': return 16
    if string_label == 'g': return 17
    if string_label == 'h': return 18
    if string_label == 'i': return 19
    if string_label == 'j': return 20
    if string_label == 'k': return 21
    if string_label == 'l': return 22
    if string_label == 'm': return 23
    if string_label == 'n': return 24
    if string_label == 'o': return 25
    if string_label == 'p': return 26
    if string_label == 'q': return 27
    if string_label == 'r': return 28
    if string_label == 's': return 29
    if string_label == 't': return 30
    if string_label == 'u': return 31
    if string_label == 'v': return 32
    if string_label == 'w': return 33
    if string_label == 'x': return 34
    if string_label == 'y': return 35
    if string_label == 'z': return 36

    else:
        raise Exception('unkown class_label')


def int_to_label(string_label):
    if string_label == 1: return '0'
    if string_label == 2: return '1'
    if string_label == 3: return '2'
    if string_label == 4: return '3'
    if string_label == 5: return '4'
    if string_label == 6: return '5'
    if string_label == 7: return '6'
    if string_label == 8: return '7'
    if string_label == 9: return '8'
    if string_label == 10: return '9'
    if string_label == 11: return 'a'
    if string_label == 12: return 'b'
    if string_label == 13: return 'c'
    if string_label == 14: return 'd'
    if string_label == 15: return 'e'
    if string_label == 16: return 'f'
    if string_label == 17: return 'g'
    if string_label == 18: return 'h'
    if string_label == 19: return 'i'
    if string_label == 20: return 'j'
    if string_label == 21: return 'k'
    if string_label == 22: return 'l'
    if string_label == 23: return 'm'
    if string_label == 24: return 'n'
    if string_label == 25: return 'o'
    if string_label == 26: return 'p'
    if string_label == 27: return 'q'
    if string_label == 28: return 'r'
    if string_label == 29: return 's'
    if string_label == 30: return 't'
    if string_label == 31: return 'u'
    if string_label == 32: return 'v'
    if string_label == 33: return 'w'
    if string_label == 34: return 'x'
    if string_label == 35: return 'y'
    if string_label == 36: return 'z'
    else:
        raise Exception('unkown class_label')
