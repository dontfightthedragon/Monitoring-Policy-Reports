
import re
def basic_clean(text):
    if not isinstance(text, str): return ""
    text = re.sub(r"[\r\n\t]+"," ", text)
    text = re.sub(r"\s{2,}"," ", text).strip()
    return text
