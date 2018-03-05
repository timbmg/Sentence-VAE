import torch
import numpy as np
from torch.autograd import Variable
from collections import defaultdict, Counter, OrderedDict

class OrderedCounter(Counter, OrderedDict):
    'Counter that remembers the order elements are first encountered'

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, OrderedDict(self))

    def __reduce__(self):
        return self.__class__, (OrderedDict(self),)

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)


def idx2word(idx, i2w, pad_idx):

    sent_str = [str()]*len(idx)

    for i, sent in enumerate(idx):

        for word_id in sent:

            if word_id == pad_idx:
                break
            sent_str[i] += i2w[str(word_id)] + " "

        sent_str[i] = sent_str[i].strip()


    return sent_str


def interpolate(start, end, steps):

    interpolation = np.zeros((start.shape[0], steps + 2))

    for dim, (s,e) in enumerate(zip(start,end)):
        interpolation[dim] = np.linspace(s,e,steps+2)

    return interpolation.T
