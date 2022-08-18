from collections import Counter
from math import log
def ngrams(sequence, n, pad_left=False, pad_right=False,
           left_pad_symbol=None, right_pad_symbol=None):
    """
    Return the ngrams generated from a sequence of items, as an iterator.
    For example:
        # >>> from nltk.util import ngrams
        # >>> list(ngrams([1,2,3,4,5], 3))
        [(1, 2, 3), (2, 3, 4), (3, 4, 5)]
    Wrap with list for a list version of this function.  Set pad_left
    or pad_right to true in order to get additional ngrams:
        # >>> list(ngrams([1,2,3,4,5], 2, pad_right=True))
        [(1, 2), (2, 3), (3, 4), (4, 5), (5, None)]
        # >>> list(ngrams([1,2,3,4,5], 2, pad_right=True, right_pad_symbol='</s>'))
        [(1, 2), (2, 3), (3, 4), (4, 5), (5, '</s>')]
        # >>> list(ngrams([1,2,3,4,5], 2, pad_left=True, left_pad_symbol='<s>'))
        [('<s>', 1), (1, 2), (2, 3), (3, 4), (4, 5)]
        # >>> list(ngrams([1,2,3,4,5], 2, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>'))
        [('<s>', 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, '</s>')]
    :param sequence: the source data to be converted into ngrams
    :type sequence: sequence or iter
    :param n: the degree of the ngrams
    :type n: int
    :param pad_left: whether the ngrams should be left-padded
    :type pad_left: bool
    :param pad_right: whether the ngrams should be right-padded
    :type pad_right: bool
    :param left_pad_symbol: the symbol to use for left padding (default is None)
    :type left_pad_symbol: any
    :param right_pad_symbol: the symbol to use for right padding (default is None)
    :type right_pad_symbol: any
    :rtype: sequence or iter
    """

    sequence = iter(sequence)
    history = []
    while n > 1:
        history.append(next(sequence))
        n -= 1
    for item in sequence:
        history.append(item)
        yield tuple(history)
        del history[0]


def distinct_n_sentence_level(sentence, n):
    if len(sentence) == 0:
        return 0.0
    elif len(sentence) < n:
        return 1.0
    distinct_ngrams = set(ngrams(sentence, n))
    return len(distinct_ngrams) / len(sentence)


def distinct_n_corpus_level(sentences, n):
    if len(sentences) == 0:
        return 0.0
    distinct_ngrams = set()
    total_length = 0
    for sentence in sentences:
        if len(sentence) < n:
            distinct_ngrams = distinct_ngrams | set(sentence)
        else:
            distinct_ngrams = distinct_ngrams | set(ngrams(sentence, n))
        total_length += len(sentence)
    return len(distinct_ngrams) / total_length

def entropy(sentences, n):
    if len(sentences) == 0:
        return 0.0
    frequency = Counter()
    for sentence in sentences:
        if len(sentence) < n:
            frequency += Counter([tuple(sentence)])
        else:
            frequency += Counter(list(ngrams(sentence, n)))
    total_frequency = sum(frequency.values())
    log_part = 0
    for key, value in frequency.items():
        log_part += value * log(value / total_frequency)
    return -1 * log_part / total_frequency


if __name__ == '__main__':
    print(entropy([[1, 2, 1], [1, 3, 2], [1, 2, 3, 4]], 2))
    print(distinct_n_sentence_level([1, 2, 1], 4))
    print(distinct_n_corpus_level([[1, 2, 1], [1, 3, 2], [1, 2, 3, 4]], 2))
