import re


def remove_author(s):
    return collapse_multiple_whitespaces(re.sub(r'^(?:@[\w!\"#$%&\'()*+,-./:;<=>?\\^_`{|}~]+|\s*)+', '', s))


def collapse_multiple_whitespaces(s):
    return re.sub(r'[ \t]+', ' ', s).strip()



def strip_anchor_tags(text):
    """Remove HTML code resulting from anchors like <a href="https://t.co/abc">text</a> and
    replace the anchor by the displayed inner text.
"""
    pattern = r"<a(?:\s+[^>]*)?>(.*?)</a>"
    return re.sub(pattern, lambda m: m.group(1) + " ", text)


def remove_url(s):
    return collapse_multiple_whitespaces(re.sub(r'https://t\.co/\S+', '', s))


def remove_rt(s):
    return collapse_multiple_whitespaces(re.sub(r'^RT\s*', '', s))


def fix_encoding_of_apostrophes(s):
    return s.replace("�", "'")


def convert_to_int(item):
    try:
        return int(item)
    except ValueError:
        return item