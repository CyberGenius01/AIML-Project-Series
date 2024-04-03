import numpy as np
import pandas as pd
import re
from nltk.stem.porter import PorterStemmer
import matplotlib.pyplot as plt
import pickle
import random
import tensorflow as tf
import kerastuner as kt


dataset = pd.read_csv(r'C:\\users\\ritesh\\Desktop\\chatbot\\Data.tsv', delimiter='\t', encoding='latin-1', header=None)
X = dataset.iloc[:,0].values
y = dataset.iloc[:,1].values

ps = PorterStemmer()

def vocabulary():
    vocab = {}
    i = 1
    vocab['<pad>'] = 0
    for entry in X:
        entry = re.sub('[^a-zA-Z]' , ' ', str(entry))
        entry = entry.lower()
        entry = entry.split()
        entry = [ps.stem(word) for word in entry]
        for word in entry:
            if word not in vocab.keys():
                vocab[word] = i
                i+=1
                   
    return vocab, i

vocab = {'<pad>': 0, 'hi': 1, 'how': 2, 'are': 3, 'you': 4, 'do': 5, 'i': 6, 'm': 7, 'fine': 8, 'about': 9, 'yourself': 10, 'pretti': 11, 'good': 12, 'thank': 13, 'for': 14, 'ask': 15, 'no': 16, 'problem': 17, 'so': 18, 'have': 19, 'been': 20, 've': 21, 'great': 22, 'what': 23, 'in': 24, 'school': 25, 'right': 26, 'now': 27, 'go': 28, 'to': 29, 'pcc': 30, 'like': 31, 'it': 32, 'there': 33, 's': 34, 'okay': 35, 'a': 36, 'realli': 37, 'big': 38, 'campu': 39, 'luck': 40, 'with': 41, 'veri': 42, 'much': 43, 'well': 44, 'never': 45, 'better': 46, 'late': 47, 'actual': 48, 'which': 49, 'attend': 50, 'enjoy': 51, 'not': 52, 'bad': 53, 'lot': 54, 'of': 55, 'peopl': 56, 'that': 57, 'today': 58, 'absolut': 59, 'love': 60, 'everyth': 61, 'haven': 62, 't': 63, 'start': 64, 'recent': 65, 'where': 66, 'far': 67, 'my': 68, 'class': 69, 'wish': 70, 'an': 71, 'ugli': 72, 'day': 73, 'know': 74, 'think': 75, 'may': 76, 'rain': 77, 'the': 78, 'middl': 79, 'summer': 80, 'shouldn': 81, 'would': 82, 'be': 83, 'weird': 84, 'yeah': 85, 'especi': 86, 'sinc': 87, 'nineti': 88, 'degre': 89, 'outsid': 90, 'horribl': 91, 'if': 92, 'and': 93, 'wa': 94, 'hot': 95, 'ye': 96, 'wasn': 97, 'everi': 98, 'me': 99, 'too': 100, 'can': 101, 'wait': 102, 'until': 103, 'winter': 104, 'but': 105, 'sometim': 106, 'get': 107, 'cold': 108, 'd': 109, 'rather': 110, 'than': 111, 'doesn': 112, 'look': 113, 'nice': 114, 're': 115, 'later': 116, 'wouldn': 117, 'seem': 118, 'consid': 119, 'over': 120, 'exactli': 121, 'cool': 122, 'off': 123, 'one': 124, 'feel': 125, 'want': 126, 'come': 127, 'soon': 128, 'mean': 129, 'nicer': 130, 'is': 131, 'true': 132, 'hope': 133, 'weather': 134, 'ani': 135, 'pointless': 136, 'down': 137, 'some': 138, 'didn': 139, 'though': 140, 'deal': 141, 'such': 142, 'doe': 143, 'whi': 144, 'clear': 145, 'air': 146, 'alway': 147, 'smell': 148, 'fresh': 149, 'after': 150, 'night': 151, 'becaus': 152, 'see': 153, 'star': 154, 'perfectli': 155, 'isn': 156, 'will': 157, 'sky': 158, 'same': 159, 'way': 160, 'when': 161, 'closer': 162, 'don': 163, 'out': 164, 'clean': 165, 'understand': 166, 'make': 167, 'cleaner': 168, 'most': 169, 'at': 170, 'more': 171, 'clearli': 172, 'beach': 173, 'thi': 174, 'weekend': 175, 'sound': 176, 'fun': 177, 'heard': 178, 'warm': 179, 'perfect': 180, 'believ': 181, 'california': 182, 'unpredict': 183, 'minut': 184, 'then': 185, 'next': 186, 'just': 187, 'stay': 188, 'we': 189, 'our': 190, 'activ': 191, 'plan': 192, 'ahead': 193, 'time': 194, 'thing': 195, 'easier': 196, 'take': 197, 'trip': 198, 'forecast': 199, 'say': 200, 'on': 201, 'll': 202, 'ruin': 203, 'badli': 204, 'constantli': 205, 'chang': 206, 'could': 207, 'sooner': 208, 'predict': 209, 'life': 210, 'suppos': 211, 'got': 212, 'uncertain': 213, 'imposs': 214, 'happen': 215, 'differ': 216, 'us': 217, 'hello': 218, 'speak': 219, 'alic': 220, 'pleas': 221, 'she': 222, 'tri': 223, 'call': 224, 'all': 225, 'sorri': 226, 'up': 227, 'were': 228, 'oh': 229, 'hang': 230, 'tomorrow': 231, 'sure': 232, 'did': 233, 'mayb': 234, 'movi': 235, 'or': 236, 'someth': 237, 'let': 238, 'goodby': 239, 'answer': 240, 'phone': 241, 'had': 242, 'chore': 243, 'reason': 244, 'your': 245, 'mind': 246, 'talk': 247, 'avail': 248, 'her': 249, 'hundr': 250, 'busi': 251, 'apolog': 252, 'need': 253, 'somewher': 254, 'special': 255, 'seen': 256, 'new': 257, 'girl': 258, 'describ': 259, 'tall': 260, 'five': 261, 'feet': 262, 'even': 263, 'ha': 264, 'light': 265, 'brown': 266, 'eye': 267, 'around': 268, 'yet': 269, 'tell': 270, 'kind': 271, 'short': 272, 'height': 273, 'probabl': 274, 'first': 275, 'notic': 276, 'beauti': 277, 'might': 278, 'bump': 279, 'into': 280, 'befor': 281, 'met': 282, 'prettiest': 283, 'quit': 284, 'onli': 285, 'facial': 286, 'featur': 287, 'who': 288, 'weren': 289, 'yesterday': 290, 'wrong': 291, 'stomach': 292, 'upset': 293, 'anyth': 294, 'alreadi': 295, 'took': 296, 'medicin': 297, 'miss': 298, 'sick': 299, 'stomachach': 300, 'still': 301, 'under': 302, 'earlier': 303, 'home': 304, 'bother': 305, 'littl': 306, 'store': 307, 'pepto': 308, 'bismol': 309, 'hear': 310, 'news': 311, 'promot': 312, 'job': 313, 'serious': 314, 'am': 315, 'excit': 316, 'congratul': 317, 'happi': 318, 'deserv': 319, 'told': 320, 'work': 321, 'week': 322, 'truth': 323, 'seriou': 324, 'boss': 325, 'offer': 326, 'appreci': 327, 'idea': 328, 'real': 329, 'outfit': 330, 'other': 331, 'from': 332, 'maci': 333, 'again': 334, 'these': 335, 'shoe': 336, 'they': 337, 'chuck': 338, 'taylor': 339, 'those': 340, 'cost': 341, 'forti': 342, 'dollar': 343, 'buy': 344, 'myself': 345, 'pair': 346, 'wear': 347, 'bought': 348, 'coupl': 349, 'ago': 350, 'santa': 351, 'anita': 352, 'mall': 353, 'them': 354, 'find': 355, 'own': 356, 'cute': 357, 'brand': 358, 'went': 359, 'pick': 360, 'found': 361, 'spare': 362, 'draw': 363, 'paint': 364, 'learn': 365, 'back': 366, 'high': 367, 'art': 368, 'talent': 369, 'hidden': 370, 'knew': 371, 'onc': 372, 'while': 373, 'long': 374, 'known': 375, 'sort': 376, 'favorit': 377, 'hobbi': 378, 'often': 379, 'taught': 380, 'superbad': 381, 'funniest': 382, 'ever': 383, 'funni': 384, 'saw': 385, 'came': 386, 'theater': 387, 'laugh': 388, 'through': 389, 'whole': 390, 'brought': 391, 'tear': 392, 'mine': 393, 'dvd': 394, 'hous': 395, 'watch': 396, 'honestli': 397, 'hilari': 398, 'couldn': 399, 'help': 400, 'either': 401, 'here': 402, 'cours': 403, 'best': 404, 'super': 405, 'lie': 406, 'made': 407, 'line': 408, 'keep': 409, 'throughout': 410, 'hyster': 411, 'muscl': 412, 'hurt': 413, 'afterward': 414, 'felt': 415, 'type': 416, 'music': 417, 'listen': 418, 'instanc': 419, 'rock': 420, 'r': 421, 'b': 422, 'instrument': 423, 'use': 424, 'excel': 425, 'variou': 426, 'genr': 427, 'both': 428, 'interest': 429, 'certain': 430, 'basketbal': 431, 'game': 432, 'friday': 433, 'won': 434, 'play': 435, 'should': 436, 'score': 437, 'man': 438, 'close': 439, 'abl': 440, 'unabl': 441, 'intens': 442, 'end': 443, 'win': 444, 'team': 445, 'victori': 446, 'free': 447, 'mad': 448, 'definit': 449, 'hard': 450, 'final': 451, 'lost': 452, 'by': 453, 'three': 454, 'point': 455, 'must': 456, 'gone': 457, 'friend': 458, 'noth': 459, 'anoth': 460, 'none': 461, 'invit': 462, 'pass': 463, 'as': 464, 'catch': 465, 'decid': 466, 'whether': 467, 'suck': 468, 'assign': 469, 'english': 470, 'welcom': 471, 'glad': 472, 'ill': 473, 'troubl': 474, 'return': 475, 'favor': 476, 'give': 477, 'mention': 478, 'cousin': 479, 'labor': 480, 'babi': 481, 'last': 482, 'anyon': 483, 'thought': 484, 'somebodi': 485, 'pound': 486, 'ounc': 487, 'god': 488, 'visit': 489, 'debrah': 490, 'nobodi': 491, 'wow': 492, 'sad': 493, 'switch': 494, 'anyway': 495, 'odd': 496, 'subject': 497, 'question': 498, 'stop': 499, 'alon': 500, 'instead': 501, 'random': 502, 'regardless': 503, 'also': 504, 'basic': 505, 'whatev': 506, 'crazi': 507, 'hey': 508, 'jessica': 509, 'parti': 510, 'gave': 511, 'o': 512, 'clock': 513, 'mani': 514, 'given': 515, 'saturday': 516, 'morn': 517, 'guess': 518, 'intend': 519, 'awesom': 520, 'eight': 521, 'year': 522, 'load': 523, 'p': 524, 'hasn': 525, 'throw': 526, 'realiz': 527, 'doubt': 528, 'hate': 529, 'stand': 530, 'dinner': 531, 'famili': 532, 'hold': 533, 'rush': 534, 'nosey': 535, 'harsh': 536, 'convers': 537, 'fast': 538, 'done': 539, 'besid': 540, 'polit': 541, 'readi': 542, 'care': 543, 'worri': 544, 'eat': 545, 'bye': 546, 'set': 547, 'live': 548, 'pasadena': 549, 'northern': 550, 'southern': 551, 'citi': 552, 'lo': 553, 'angel': 554, 'million': 555, 'car': 556, 'honda': 557, 'old': 558, 'wash': 559, 'oil': 560, 'mechan': 561, 'twice': 562, 'girlfriend': 563, 'rich': 564, 'enough': 565, 'guy': 566, 'money': 567, 'neither': 568, 'joke': 569, 'walk': 570, 'dog': 571, 'poodl': 572, 'bark': 573, 'shut': 574, 'mom': 575, 'watchdog': 576, 'borrow': 577, 'lunch': 578, 'wallet': 579, 'empti': 580, 'broke': 581, 'lend': 582, 'pay': 583, 'month': 584, 'almost': 585, 'drown': 586, 'lifeguard': 587, 'dive': 588, 'water': 589, 'he': 590, 'swam': 591, 'turn': 592, 'marri': 593, 'divorc': 594, 'two': 595, 'wife': 596, 'left': 597, 'leav': 598, 'said': 599, 'anymor': 600, 'terribl': 601, 'fell': 602, 'bore': 603, 'tv': 604, 'show': 605, 'togeth': 606, 'agre': 607, 'small': 608, 'rose': 609, 'parad': 610, 'wonder': 611, 'restaur': 612, 'mountain': 613, 'friendli': 614, 'mattress': 615, 'matter': 616, 'comfort': 617, 'toss': 618, 'drink': 619, 'coffe': 620, 'mark': 621, 'arm': 622, 'bite': 623, 'cat': 624, 'bedbug': 625, 'bit': 626, 'laptop': 627, 'slow': 628, 'comput': 629, 'shop': 630, 'window': 631, 'hit': 632, 'someon': 633, 'head': 634, 'pizza': 635, 'everybodi': 636, 'varieti': 637, 'x': 638, 'pepperoni': 639, 'chees': 640, 'salad': 641, 'save': 642, 'expens': 643, 'payment': 644, 'thirti': 645, 'thousand': 646, 'forev': 647, 'penni': 648, 'seven': 649, 'ocean': 650, 'goe': 651, 'deep': 652, 'mile': 653, 'fish': 654, 'bottom': 655, 'top': 656, 'warn': 657, 'boyfriend': 658, 'birthday': 659, 'spend': 660, 'herself': 661, 'ring': 662, 'la': 663, 'vega': 664, 'gambl': 665, 'him': 666, 'anim': 667, 'each': 668, 'els': 669, 'food': 670, 'cloth': 671, 'dirti': 672, 'bathroom': 673, 'easi': 674, 'sink': 675, 'tub': 676, 'counter': 677, 'toilet': 678, 'finish': 679, 'wast': 680, 'sit': 681, 'mouth': 682, 'open': 683, 'volum': 684, 'meant': 685, 'write': 686, 'letter': 687, 'grandma': 688, 'put': 689, 'envelop': 690, 'seal': 691, 'stamp': 692, 'kitchen': 693, 'drawer': 694, 'mail': 695, 'e': 696, 'yawn': 697, 'sleepi': 698, 'bed': 699, 'record': 700, 'tape': 701, 'broken': 702, 'rerun': 703, 'origin': 704, 'asleep': 705, 'commerci': 706, 'zzz': 707, 'sunday': 708, 'forgot': 709, 'church': 710, 'coat': 711, 'tie': 712, 'respect': 713, 'forgiv': 714, 'feed': 715, 'meow': 716, 'hungri': 717, 'homework': 718, 'themselv': 719, 'rid': 720, 'shave': 721, 'cut': 722, 'blade': 723, 'electr': 724, 'shaver': 725, 'nois': 726, 'grow': 727, 'beard': 728, 'stuff': 729, 'stick': 730, 'hmm': 731, 'cream': 732, 'face': 733, 'lick': 734, 'excus': 735, 'read': 736, 'paper': 737, 'rude': 738, 'world': 739, 'percent': 740, 'puppi': 741, 'shot': 742, 'plate': 743, 'veget': 744, 'kitten': 745, 'away': 746, 'black': 747, 'blacki': 748, 'parent': 749, 'trust': 750, 'heaven': 751, 'die': 752, 'unhappi': 753, 'hell': 754, 'husband': 755, 'cell': 756, 'buri': 757, 'batteri': 758, 'thirteenth': 759, 'unlucki': 760, 'hotel': 761, 'mistak': 762, 'floor': 763, 'stole': 764, 'lesson': 765, 'prove': 766, 'mcdonald': 767, 'reserv': 768, 'hassl': 769, 'father': 770, 'mother': 771, 'angri': 772, 'drop': 773, 'move': 774, 'apart': 775, 'begin': 776, 'fruit': 777, 'ripe': 778, 'thrift': 779, 'price': 780, 'born': 781, 'cent': 782, 'complain': 783, 'button': 784, 'shirt': 785, 'lose': 786, 'pant': 787, 'cuff': 788, 'extra': 789, 'sew': 790, 'chocol': 791, 'mirror': 792, 'fat': 793, 'laundri': 794, 'sheet': 795, 'towel': 796, 'pillowcas': 797, 'pillow': 798, 'dri': 799, 'dryer': 800, 'fold': 801, 'radio': 802, 'mostli': 803, 'current': 804, 'event': 805, 'tax': 806, 'fridg': 807, 'market': 808, 'orang': 809, 'appl': 810, 'tasti': 811, 'candi': 812, 'bar': 813, 'sandwich': 814, 'ham': 815, 'bread': 816, 'cabinet': 817, 'mustard': 818, 'potato': 819, 'chip': 820, 'pickl': 821, 'bath': 822, 'young': 823, 'ladi': 824, 'perfum': 825, 'screen': 826, 'drive': 827, 'crash': 828, 'hp': 829, 'file': 830, 'smart': 831, 'plu': 832, 'instal': 833, 'remov': 834, 'replac': 835, 'screw': 836, 'email': 837, 'address': 838, 'bluedog': 839, 'incomplet': 840, 'cherri': 841, 'ca': 842, 'correct': 843, 'street': 844, 'state': 845, 'zip': 846, 'code': 847, 'yahoo': 848, 'com': 849, 'nap': 850, 'unplug': 851, 'wake': 852, 'hour': 853, 'sleep': 854, 'awak': 855, 'nose': 856, 'cook': 857, 'dream': 858, 'tire': 859, 'funer': 860, 'dad': 861, 'son': 862, 'speech': 863, 'yike': 864, 'blow': 865, 'plane': 866, 'loud': 867, 'word': 868, 'eleph': 869, 'deaf': 870, 'twenti': 871, 'few': 872, 'lone': 873, 'share': 874, 'cheat': 875, 'men': 876, 'book': 877, 'cheater': 878, 'poke': 879, 'woman': 880, 'chop': 881, 'toe': 882, 'honey': 883, 'swear': 884, 'meet': 885, 'jerk': 886, 'full': 887, 'everywher': 888, 'yell': 889, 'form': 890, 'mi': 891, 'initi': 892, 'mm': 893, 'dd': 894, 'yy': 895, 'number': 896, 'exampl': 897, 'birth': 898, 'date': 899, 'januari': 900, 'simpl': 901, 'print': 902, 'fill': 903, 'bubbl': 904, 'complet': 905, 'shelter': 906, 'bet': 907, 'drag': 908, 'name': 909, 'woke': 910, 'gray': 911, 'wet': 912, 'umbrella': 913, 'noon': 914, 'hotter': 915, 'heat': 916, 'condition': 917, 'repairman': 918, 'snow': 919, 'snowman': 920, 'carrot': 921, 'bank': 922, 'withdraw': 923, 'atm': 924, 'automat': 925, 'teller': 926, 'machin': 927, 'insert': 928, 'debit': 929, 'card': 930, 'blue': 931, 'bin': 932, 'front': 933, 'recycl': 934, 'truck': 935, 'usual': 936, 'tuesday': 937, 'forget': 938, 'rememb': 939, 'nation': 940, 'digit': 941, 'convert': 942, 'inch': 943, 'channel': 944, 'six': 945, 'korean': 946, 'pilot': 947, 'canada': 948, 'flew': 949, 'u': 950, 'fighter': 951, 'jet': 952, 'follow': 953, 'land': 954, 'highway': 955, 'cop': 956, 'shoot': 957, 'poor': 958, 'polic': 959, 'robber': 960, 'report': 961, 'robberi': 962, 'hair': 963, 'race': 964, 'racist': 965, 'identifi': 966, 'their': 967, 'male': 968, 'femal': 969, 'sexist': 970, 'wipe': 971, 'sleev': 972, 'tissu': 973, 'age': 974, 'daddi': 975, 'boy': 976, 'mommi': 977, 'marriag': 978, 'respons': 979, 'children': 980, 'except': 981, 'afford': 982, 'artist': 983, 'jar': 984, 'pencil': 985, 'picasso': 986, 'famou': 987, 'drew': 988, 'third': 989, 'grade': 990, 'worth': 991, 'aren': 992, 'patient': 993, 'along': 994, 'beer': 995, 'power': 996, 'drug': 997, 'cigarett': 998, 'prefer': 999, 'tough': 1000, 'tast': 1001, 'hole': 1002, 'pocket': 1003, 'carri': 1004, 'pen': 1005, 'onto': 1006, 'lucki': 1007, 'sharp': 1008, 'knife': 1009, 'crimin': 1010, 'fix': 1011, 'iron': 1012, 'patch': 1013, 'glue': 1014, 'melt': 1015, 'ten': 1016, 'ear': 1017, 'languag': 1018, 'women': 1019, 'spanish': 1020, 'teach': 1021, 'ahora': 1022, 'april': 1023, 'anniversari': 1024, 'earth': 1025, 'yearli': 1026, 'remind': 1027, 'planet': 1028, 'reus': 1029, 'green': 1030, 'plastic': 1031, 'bag': 1032, 'shorter': 1033, 'shower': 1034, 'poetri': 1035, 'poem': 1036, 'rhyme': 1037, 'buckl': 1038, 'shakespear': 1039, 'poet': 1040, 'song': 1041, 'without': 1042, 'averag': 1043, 'iq': 1044, 'test': 1045, 'ridicul': 1046, 'whenev': 1047, 'lead': 1048, 'stori': 1049, 'actress': 1050, 'court': 1051, 'licens': 1052, 'second': 1053, 'actor': 1054, 'daughter': 1055, 'bull': 1056, 'chase': 1057, 'supermarket': 1058, 'octo': 1059, 'hire': 1060, 'nanni': 1061, 'infant': 1062, 'death': 1063, 'avoid': 1064, 'cremat': 1065, 'ash': 1066, 'shaken': 1067, 'coffin': 1068, 'space': 1069, 'cemeteri': 1070, 'seldom': 1071, 'dead': 1072, 'figur': 1073, 'kid': 1074, 'mud': 1075, 'carpet': 1076, 'vacuum': 1077, 'till': 1078, 'rais': 1079, 'flag': 1080, 'stripe': 1081, 'war': 1082, 'plant': 1083, 'entir': 1084, 'distanc': 1085, 'servic': 1086, 'dial': 1087, 'colleg': 1088, 'teacher': 1089, 'classmat': 1090, 'check': 1091, 'desk': 1092, 'graviti': 1093, 'import': 1094, 'forc': 1095, 'pull': 1096, 'pour': 1097, 'glass': 1098, 'float': 1099, 'balloon': 1100, 'doctor': 1101, 'prescript': 1102, 'appoint': 1103, 'yellow': 1104, 'page': 1105, 'case': 1106, 'notebook': 1107, 'calcul': 1108, 'permit': 1109, 'dictionari': 1110, 'classroom': 1111, 'magazin': 1112, 'subscrib': 1113, 'cartoon': 1114, 'photo': 1115, 'sale': 1116, 'film': 1117, 'review': 1118, 'section': 1119, 'subscript': 1120, 'cancel': 1121, 'ink': 1122, 'shake': 1123, 'shook': 1124, 'rule': 1125, 'graduat': 1126, 'join': 1127, 'armi': 1128, 'kill': 1129, 'major': 1130, 'total': 1131, 'park': 1132, 'drove': 1133, 'half': 1134, 'spot': 1135, 'huge': 1136, 'librari': 1137, 'room': 1138, 'thiev': 1139, 'belong': 1140, 'backpack': 1141, 'ipod': 1142, 'safe': 1143, 'math': 1144, 'add': 1145, 'stupid': 1146, 'pray': 1147, 'occasion': 1148, 'prayer': 1149, 'driver': 1150, 'ran': 1151, 'student': 1152, 'instantli': 1153, 'hospit': 1154, 'push': 1155, 'hood': 1156, 'gentli': 1157, 'place': 1158, 'fire': 1159, 'depart': 1160, 'nearbi': 1161, 'himself': 1162, 'ride': 1163, 'bu': 1164, 'seat': 1165, 'bring': 1166, 'strang': 1167, 'everyon': 1168, 'faster': 1169, 'buse': 1170, 'run': 1171, 'four': 1172, 'crowd': 1173, 'aisl': 1174, 'unsaf': 1175, 'rob': 1176, 'eleven': 1177, 'person': 1178, 'fair': 1179, 'ticket': 1180, 'cross': 1181, 'crosswalk': 1182, 'red': 1183, 'hand': 1184, 'blink': 1185, 'white': 1186, 'sign': 1187, 'speed': 1188, 'limit': 1189, 'fastest': 1190, 'flat': 1191, 'hurri': 1192, 'fault': 1193, 'accid': 1194, 'drunk': 1195, 'caus': 1196, 'cadillac': 1197, 'luxuri': 1198, 'key': 1199, 'camera': 1200, 'longer': 1201, 'explod': 1202, 'hamburg': 1203, 'exit': 1204, 'tree': 1205, 'steal': 1206, 'tow': 1207, 'traffic': 1208, 'wors': 1209, 'between': 1210, 'honk': 1211, 'horn': 1212, 'bucket': 1213, 'rins': 1214, 'scrub': 1215, 'spong': 1216, 'soap': 1217, 'windi': 1218, 'fli': 1219, 'wind': 1220, 'danger': 1221, 'damag': 1222, 'bird': 1223, 'stone': 1224, 'order': 1225, 'arrow': 1226, 'lane': 1227, 'straight': 1228, 'quicker': 1229, 'cheap': 1230, 'reliabl': 1231, 'low': 1232, 'mileag': 1233, 'anywher': 1234, 'afternoon': 1235, 'offic': 1236, 'registr': 1237, 'sudden': 1238, 'siren': 1239, 'roll': 1240, 'attitud': 1241, 'downtown': 1242, 'jaywalk': 1243, 'enter': 1244, 'fight': 1245, 'wit': 1246, 'near': 1247, 'usc': 1248, 'passeng': 1249, 'injur': 1250, 'continu': 1251, 'jail': 1252, 'dent': 1253, 'cart': 1254, 'metal': 1255, 'titan': 1256, 'twelv': 1257, 'cri': 1258, 'poker': 1259, 'player': 1260, 'lip': 1261, 'nope': 1262, 'quietli': 1263, 'neighbor': 1264, 'lotto': 1265, 'chanc': 1266, 'basebal': 1267, 'rainstorm': 1268, 'dome': 1269, 'season': 1270, 'bitter': 1271, 'sugar': 1272, 'step': 1273, 'chilli': 1274, 'cap': 1275, 'jacket': 1276, 'glove': 1277, 'colder': 1278, 'warmer': 1279, 'sun': 1280, 'per': 1281, 'ga': 1282, 'sofa': 1283, 'pipe': 1284, 'bear': 1285, 'yard': 1286, 'mous': 1287, 'weatherman': 1288, 'temperatur': 1289, 'town': 1290, 'sport': 1291, 'onlin': 1292, 'internet': 1293, 'amaz': 1294, 'travel': 1295, 'china': 1296, 'stood': 1297, 'wall': 1298, 'beatl': 1299, 'group': 1300, 'sing': 1301, 'earli': 1302, 'pursuit': 1303, 'happy': 1304, 'base': 1305, 'starbuck': 1306, 'common': 1307, 'sniff': 1308, 'bill': 1309, 'easter': 1310, 'homeless': 1311, 'wheelchair': 1312, 'color': 1313, 'curs': 1314, 'violenc': 1315, 'action': 1316, 'pb': 1317, 'public': 1318, 'broadcast': 1319, 'system': 1320, 'garden': 1321, 'knit': 1322, 'dozen': 1323, 'ad': 1324, 'donat': 1325, 'judg': 1326, 'judi': 1327, 'lawsuit': 1328, 'ebay': 1329, 'seller': 1330, 'buyer': 1331, 'singer': 1332, 'cd': 1333, 'station': 1334, 'occur': 1335, 'signal': 1336, 'singl': 1337, 'cabl': 1338, 'antenna': 1339, 'rabbit': 1340, 'strong': 1341, 'uh': 1342, 'weigh': 1343, 'french': 1344, 'snail': 1345, 'sight': 1346, 'caught': 1347, 'hug': 1348, 'mood': 1349, 'boob': 1350, 'tonight': 1351, 'blind': 1352, 'behind': 1353, 'checkout': 1354, 'pineappl': 1355, 'gotten': 1356, 'shelf': 1357, 'fact': 1358, 'jazz': 1359, 'club': 1360, 'pleasant': 1361, 'chemistri': 1362, 'worst': 1363, 'taxi': 1364, 'meal': 1365, 'tip': 1366, 'blame': 1367, 'act': 1368, 'furthest': 1369, 'trade': 1370, 'sensit': 1371, 'odor': 1372, 'sneak': 1373, 'stink': 1374, 'appli': 1375, 'custom': 1376, 'chines': 1377, 'delici': 1378, 'popular': 1379, 'slowest': 1380, 'burger': 1381, 'worker': 1382, 'soup': 1383, 'tomato': 1384, 'lemon': 1385, 'butter': 1386, 'bacon': 1387, 'toast': 1388, 'rice': 1389, 'waiter': 1390, 'steak': 1391, 'spit': 1392, 'smile': 1393, 'fingernail': 1394, 'nail': 1395, 'disgust': 1396, 'yuck': 1397, 'charg': 1398, 'piec': 1399, 'serv': 1400, 'main': 1401, 'tabl': 1402, 'chair': 1403, 'silverwar': 1404, 'inspect': 1405, 'examin': 1406, 'germ': 1407, 'focu': 1408, 'oop': 1409, 'manag': 1410, 'send': 1411, 'immedi': 1412, 'adventur': 1413, 'door': 1414, 'sat': 1415, 'andi': 1416, 'warhol': 1417, 'butterfli': 1418, 'flower': 1419, 'napkin': 1420, 'stain': 1421, 'peanut': 1422, 'foul': 1423, 'ball': 1424, 'golf': 1425, 'silli': 1426, 'certainli': 1427, 'ground': 1428, 'golfer': 1429, 'mental': 1430, 'nut': 1431, 'river': 1432, 'lake': 1433, 'rod': 1434, 'bait': 1435, 'slide': 1436, 'yanke': 1437, 'dodger': 1438, 'practic': 1439, 'jog': 1440, 'tiger': 1441, 'greatest': 1442, 'scuba': 1443, 'retir': 1444, 'relax': 1445, 'nervou': 1446, 'tournament': 1447, 'sank': 1448, 'foot': 1449, 'putt': 1450, 'stroke': 1451, 'footer': 1452, 'outer': 1453, 'human': 1454, 'possibl': 1455, 'mar': 1456, 'babe': 1457, 'ruth': 1458, 'cheer': 1459, 'punch': 1460, 'argument': 1461, 'victim': 1462, 'concret': 1463, 'cheaper': 1464, 'outfield': 1465, 'amateur': 1466, 'hitter': 1467, 'fan': 1468, 'leagu': 1469, 'suspend': 1470, 'crime': 1471, 'neighborhood': 1472, 'sidewalk': 1473, 'burn': 1474, 'smoke': 1475, 'alarm': 1476, 'fifteen': 1477, 'law': 1478, 'firebug': 1479, 'sens': 1480, 'latest': 1481, 'murder': 1482, 'seatbelt': 1483, 'protect': 1484, 'uncomfort': 1485, 'tight': 1486, 'breath': 1487, 'bulb': 1488, 'burnt': 1489, 'textbook': 1490, 'slip': 1491, 'fall': 1492, 'break': 1493, 'rest': 1494, 'stepladd': 1495, 'puddl': 1496, 'slick': 1497, 'crack': 1498, 'cone': 1499, 'mop': 1500, 'quickli': 1501, 'north': 1502, 'roster': 1503, 'build': 1504, 'fireman': 1505, 'upstair': 1506, 'stove': 1507, 'burner': 1508, 'gun': 1509, 'receipt': 1510, 'america': 1511, 'articl': 1512, 'mayor': 1513, 'rate': 1514, 'lock': 1515, 'earthquak': 1516, 'destroy': 1517, 'safer': 1518, 'florida': 1519, 'hurrican': 1520, 'june': 1521, 'octob': 1522, 'harmless': 1523, 'andrew': 1524, 'hawaii': 1525, 'vacat': 1526, 'island': 1527, 'swim': 1528, 'sunni': 1529, 'breakfast': 1530, 'egg': 1531, 'sausag': 1532, 'juic': 1533, 'allow': 1534, 'pet': 1535, 'airport': 1536, 'freeway': 1537, 'least': 1538, 'york': 1539, 'dure': 1540, 'christma': 1541, 'holiday': 1542, 'march': 1543, 'sell': 1544, 'flown': 1545, 'jam': 1546, 'cough': 1547, 'sneez': 1548, 'elbow': 1549, 'knee': 1550, 'climb': 1551, 'zoo': 1552, 'row': 1553, 'across': 1554, 'atlant': 1555, 'paid': 1556, 'cruis': 1557, 'ship': 1558, 'depend': 1559, 'cabin': 1560, 'storm': 1561, 'view': 1562, 'sister': 1563, 'secur': 1564, 'altitud': 1565, 'earplug': 1566, 'spring': 1567, 'arizona': 1568, 'grand': 1569, 'canyon': 1570, 'mule': 1571, 'trail': 1572, 'fallen': 1573, 'thin': 1574, 'telephon': 1575, 'snore': 1576, 'housekeep': 1577, 'nonsmok': 1578, 'stunk': 1579, 'elev': 1580, 'ice': 1581, 'phoni': 1582, 'agent': 1583, 'discount': 1584, 'bomb': 1585, 'threat': 1586, 'stuck': 1587, 'someday': 1588, 'washington': 1589, 'c': 1590, 'veteran': 1591, 'privat': 1592, 'organ': 1593, 'soldier': 1594, 'ii': 1595, 'monument': 1596, 'flight': 1597, 'pictur': 1598, 'band': 1599, 'honor': 1600, 'countri': 1601, 'laid': 1602, 'cowork': 1603, 'newspap': 1604, 'interview': 1605, 'doubl': 1606, 'teeth': 1607, 'shine': 1608, 'sock': 1609, 'match': 1610, 'dark': 1611, 'babysitt': 1612, 'diaper': 1613, 'painter': 1614, 'handyman': 1615, 'drip': 1616, 'faucet': 1617, 'part': 1618, 'skill': 1619, 'flip': 1620, 'layoff': 1621, 'king': 1622, 'becom': 1623, 'routin': 1624, 'repeat': 1625, 'supervisor': 1626, 'troublemak': 1627, 'salari': 1628, 'choos': 1629, 'lighter': 1630, 'fortun': 1631, 'guarante': 1632, 'stock': 1633, 'hunch': 1634, 'compani': 1635, 'basket': 1636, 'f': 1637, 'tutor': 1638, 'killer': 1639, 'carrier': 1640, 'exercis': 1641, 'unfriendli': 1642, 'knock': 1643, 'corpor': 1644, 'sore': 1645, 'knuckl': 1646, 'lettuc': 1647, 'celeri': 1648, 'pepper': 1649, 'salt': 1650, 'dress': 1651, 'calori': 1652, 'cow': 1653, 'milk': 1654, 'leather': 1655, 'deli': 1656, 'ate': 1657, 'meat': 1658, 'slice': 1659, 'huh': 1660, 'diet': 1661, 'pasta': 1662, 'process': 1663, 'natur': 1664, 'vitamin': 1665, 'prepar': 1666, 'steam': 1667, 'sauc': 1668, 'sprinkl': 1669, 'banana': 1670, 'pink': 1671, 'peel': 1672, 'sticker': 1673, 'navel': 1674, 'roast': 1675, 'boil': 1676, 'raw': 1677, 'shell': 1678, 'soft': 1679, 'brother': 1680, 'allerg': 1681, 'strict': 1682, 'gain': 1683, 'weight': 1684, 'freezer': 1685, 'stuf': 1686, 'leftov': 1687, 'disappear': 1688, 'reheat': 1689, 'burst': 1690, 'loosen': 1691, 'belt': 1692, 'unbutton': 1693, 'carton': 1694, 'promis': 1695, 'tag': 1696, 'clerk': 1697, 'fit': 1698, 'kept': 1699, 'waist': 1700, 'bigger': 1701, 'elast': 1702, 'waistband': 1703, 'list': 1704, 'nonfat': 1705, 'swiss': 1706, 'purs': 1707, 'stronger': 1708, 'googl': 1709, 'search': 1710, 'handl': 1711, 'shopper': 1712, 'produc': 1713, 'larg': 1714, 'chariti': 1715, 'desktop': 1716, 'pc': 1717, 'mac': 1718, 'anytim': 1719, 'rip': 1720, 'solut': 1721, 'site': 1722, 'sent': 1723, 'credit': 1724, 'sharpen': 1725, 'dine': 1726, 'leg': 1727, 'rubber': 1728, 'suction': 1729, 'cup': 1730, 'stretch': 1731, 'mix': 1732, 'cash': 1733, 'monthli': 1734, 'corner': 1735, 'insid': 1736, 'intersect': 1737, 'collis': 1738, 'less': 1739, 'unit': 1740, 'side': 1741, 'checkbook': 1742, 'rent': 1743, 'doorbel': 1744, 'visitor': 1745, 'bedroom': 1746, 'afraid': 1747, 'road': 1748, 'surviv': 1749, 'fireplac': 1750, 'wood': 1751, 'quiet': 1752, 'lawn': 1753, 'rusti': 1754, 'properti': 1755, 'valu': 1756, 'council': 1757, 'barbara': 1758, 'invad': 1759, 'starv': 1760, 'grass': 1761, 'berri': 1762, 'round': 1763, 'cover': 1764, 'trash': 1765, 'solv': 1766, 'vote': 1767, 'obama': 1768, 'presid': 1769, 'speaker': 1770, 'elect': 1771, 'liar': 1772, 'voter': 1773, 'ballot': 1774, 'bowl': 1775, 'alley': 1776, 'leader': 1777, 'mccain': 1778, 'imagin': 1779, 'stress': 1780, 'duti': 1781, 'reelect': 1782, 'ralph': 1783, 'nader': 1784, 'democrat': 1785, 'republican': 1786, 'candid': 1787, 'unless': 1788, 'senat': 1789, 'offici': 1790, 'explain': 1791, 'honest': 1792, 'former': 1793, 'bush': 1794, 'confer': 1795, 'american': 1796, 'oversea': 1797, 'wound': 1798, 'spoke': 1799, 'plenti': 1800, 'sampl': 1801, 'instruct': 1802, 'against': 1803, 'prison': 1804, 'legisl': 1805, 'spent': 1806, 'measur': 1807, 'improv': 1808, 'increas': 1809, 'opposit': 1810, 'titl': 1811, 'politician': 1812, 'cereal': 1813, 'blood': 1814, 'bleed': 1815, 'soak': 1816, 'finger': 1817, 'pain': 1818, 'bone': 1819, 'cancer': 1820, 'posit': 1821, 'medic': 1822, 'sand': 1823, 'aid': 1824, 'heal': 1825, 'smoker': 1826, 'weak': 1827, 'control': 1828, 'puff': 1829, 'taken': 1830, 'suntan': 1831, 'lotion': 1832, 'tan': 1833, 'pale': 1834, 'skin': 1835, 'argu': 1836, 'surgeri': 1837, 'insur': 1838, 'rub': 1839, 'pack': 1840, 'chain': 1841, 'unbeliev': 1842, 'aliv': 1843, 'heart': 1844, 'lung': 1845, 'healthi': 1846, 'brush': 1847, 'floss': 1848, 'invent': 1849, 'dentist': 1850, 'wild': 1851, 'lizard': 1852, 'hike': 1853, 'goat': 1854, 'sweat': 1855, 'pimpl': 1856, 'gene': 1857, 'pollut': 1858, 'bright': 1859, 'pop': 1860, 'govern': 1861, 'fourth': 1862, 'swine': 1863, 'flu': 1864, 'mexico': 1865, 'frequent': 1866, 'scratch': 1867, 'emerg': 1868, 'remot': 1869, 'filthi': 1870, 'crud': 1871, 'damp': 1872, 'squeez': 1873, 'firmli': 1874, 'swell': 1875, 'normal': 1876, 'bless': 1877, 'modern': 1878, 'ach': 1879}
inv_vocab = {vocab[word]:word for word in vocab.keys()}
vocab_size = 1880

def bag_of_words(listXy):
    bag = []
    for entry in listXy:
        vector = [0]*vocab_size
        entry = re.sub('[^a-zA-Z]' , ' ', str(entry))
        entry = entry.lower()
        entry = entry.split()
        entry = [ps.stem(word) for word in entry]
        for word in entry:
            if word in vocab:
                vector[vocab[word]] = 1
        bag.append(vector)
    return np.array(bag)

X = bag_of_words(X)
y = bag_of_words(y)

#print(X.shape, y.shape)

def split(X,y):
    train = np.array([list(X), list(y)])
    random.shuffle(train)
    training_data = train[:,:2794,:]
    validation_data = train[:,2794:,:]
    classes = [vocab[word] for word in vocab.keys()]
    return np.array(training_data[0]), np.array(validation_data[0]), np.array(training_data[1]), np.array(validation_data[1]), classes

X_train, X_val, y_train, y_val, classes = split(X,y)
#print(X_train.shape, y_train.shape, X_val.shape, y_val.shape)


def tuning():
    def grid_search(hp):
        lrate = hp.Float('lrate', 1e-4, 1e-1, sampling='log')
        l1=0.0
        l2=hp.Choice('l2', values=[0.0,1e-1,1e-2,1e-3,1e-4])
        num_hidden = hp.Int('num_hidden1', 256, 1024, 16)
        num_hidden2 = hp.Int('num_hidden2', 128, 512, 16)
        num_hidden3 = hp.Int('num_hidden3', 64, 256, 16)
        num_hidden4 = hp.Int('num_hidden4', 16, 128, 16)
        drop_rate = hp.Float('drop_rate', 0.1, 0.5, sampling='log')
        regularizer = tf.keras.regularizers.l1_l2(l1,l2)
        layers = []
        layers.extend([tf.keras.layers.Dense(num_hidden,
                                         kernel_regularizer=regularizer,
                                         name='Dense-1',
                                         input_shape = (None, vocab_size)),
                   tf.keras.layers.Dropout(rate=drop_rate, name='Drop-1'),
                   tf.keras.layers.Activation(activation=tf.keras.activations.relu, name='Activation-1')
                   ])
        layers.extend([tf.keras.layers.Dense(num_hidden2,
                                         kernel_regularizer=regularizer,
                                         name='Dense-2'),
                   tf.keras.layers.Dropout(rate=drop_rate, name='Drop-2'),
                   tf.keras.layers.Activation(activation=tf.keras.activations.relu, name='Activation-2')
                   ])
        layers.extend([tf.keras.layers.Dense(num_hidden3,
                                         kernel_regularizer=regularizer,
                                         name='Dense-3'),
                   tf.keras.layers.Dropout(rate=drop_rate, name='Drop-3'),
                   tf.keras.layers.Activation(activation=tf.keras.activations.relu, name='Activation-3')
                   ])
        layers.extend([tf.keras.layers.Dense(num_hidden4,
                                         kernel_regularizer=regularizer,
                                         name='Dense-4'),
                   tf.keras.layers.Dropout(rate=drop_rate, name='Drop-4'),
                   tf.keras.layers.Activation(activation=tf.keras.activations.relu, name='Activation-4')
                   ])
        layers.extend([tf.keras.layers.Dense(vocab_size,
                                         name='Output'),
                   tf.keras.layers.Activation(activation=tf.keras.activations.softmax, name='Activation-out')
                   ])
        model = tf.keras.models.Sequential(layers, name='Chatbot')
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lrate),
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])
        return model

    tuner = kt.BayesianOptimization(
        grid_search,
        objective=kt.Objective('val_accuracy', 'max'),
        max_trials=10,
        num_initial_points=20,
        overwrite=True)

    tuner.search(
        x=X_train, y=y_train,
        validation_data=(X_val, y_val),
        batch_size=32,
        epochs=25)

    topN=1

    for x in range(topN):
        param = tuner.get_best_hyperparameters(topN)[x].values
        print(param)
        print(tuner.get_best_models(topN)[x].summary())
        
    return param

param = tuning()

def build_model(param):
    lrate, regularizer, num_hidden, num_hidden2, num_hidden3, num_hidden4, drop_rate = param['lrate'], tf.keras.regularizers.l1_l2(0.0, param['l2']), param['num_hidden1'], param['num_hidden2'], param['num_hidden3'], param['num_hidden4'], param['drop_rate']
    layers = []
    layers.extend([tf.keras.layers.Dense(num_hidden,
                                         kernel_regularizer=regularizer,
                                         name='Dense-1',
                                         input_shape = (None, vocab_size)),
                   tf.keras.layers.Dropout(rate=drop_rate, name='Drop-1'),
                   tf.keras.layers.Activation(activation=tf.keras.activations.relu, name='Activation-1')
                   ])
    layers.extend([tf.keras.layers.Dense(num_hidden2,
                                         kernel_regularizer=regularizer,
                                         name='Dense-2'),
                   tf.keras.layers.Dropout(rate=drop_rate, name='Drop-2'),
                   tf.keras.layers.Activation(activation=tf.keras.activations.relu, name='Activation-2')
                   ])
    layers.extend([tf.keras.layers.Dense(num_hidden3,
                                         kernel_regularizer=regularizer,
                                         name='Dense-3'),
                   tf.keras.layers.Dropout(rate=drop_rate, name='Drop-3'),
                   tf.keras.layers.Activation(activation=tf.keras.activations.relu, name='Activation-3')
                   ])
    layers.extend([tf.keras.layers.Dense(num_hidden4,
                                         kernel_regularizer=regularizer,
                                         name='Dense-4'),
                   tf.keras.layers.Dropout(rate=drop_rate, name='Drop-4'),
                   tf.keras.layers.Activation(activation=tf.keras.activations.relu, name='Activation-4')
                   ])
    layers.extend([tf.keras.layers.Dense(vocab_size,
                                         name='Output'),
                   tf.keras.layers.Activation(activation=tf.keras.activations.softmax, name='Activation-out')
                   ])
    model = tf.keras.models.Sequential(layers, name='Chatbot')
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lrate),
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])
    with open(r'C:\\users\\ritesh\\Desktop\\chatbot\\param.pickle', 'wb') as fout:
        pickle.dump((lrate, regularizer, num_hidden, num_hidden2, num_hidden3, num_hidden4, drop_rate), fout)
    return model

model = build_model(param)

def train_and_save(model):
    history = model.fit(x=X_train, y=y_train, validation_data=(X_val, y_val), epochs= 500, batch_size=32)
    model.save(r'C:\\users\\ritesh\\Desktop\\chatbot\\model3.h5')
    with open(r'C:\\users\\ritesh\\Desktop\\chatbot\\history3.history', 'wb') as fout:
        pickle.dump(history.history, fout)
    
        
train_and_save(model)

def visualization():
    model = tf.keras.saving.load_model(
    r'C:\\users\\ritesh\\Desktop\\chatbot\\model3.h5', custom_objects=None, compile=True, safe_mode=True)
    print(model.summary())
    with open(r'C:\\users\\ritesh\\Desktop\\chatbot\\history3.history', 'rb') as fin:
        history = pickle.load(fin)
    plt.style.use('ggplot')
    plt.plot(history['val_loss'], ls='dashed', color='blue', label='val_loss')
    plt.plot(history['loss'], color='chocolate', label='loss')
    plt.title('Loss curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.figure()
    plt.plot(history['val_accuracy'], ls='dashed', color='blue', label='val_accuracy')
    plt.plot(history['accuracy'], color='chocolate', label='accuracy')
    plt.title('Accuracy curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

visualization()    