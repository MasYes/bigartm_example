import re
import sys
from nltk.corpus import stopwords

stop_words = stopwords.words('english')
stop_words.append('-rrb-')
stop_words.append('-lrb-')
__table = str.maketrans({key: None for key in "!\"#$%&'()*-+,./>?@:[\\]^`{|}~"})
letters = re.compile("[A-z]")

def clear_text(text, exclude_stop_words=True):
    if exclude_stop_words:
        text = ' '.join([token for token in text.split() if token not in stop_words and letters.match(token)])
    else:
        text = ' '.join([token for token in text.split() if letters.match(token)])
    return text.translate(__table)

def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size
