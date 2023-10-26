import re

from nltk.stem import WordNetLemmatizer

WNL = WordNetLemmatizer()


def remove_no_one_yes_nothing(x):
    if re.match(r"(no one)|(yes)|(nothing)", x):
        return ""
    return x


def lemmatize(x):
    return WNL.lemmatize(x, "n")


ofa_pred["answer"] = ofa_pred["answer"].map(remove_no_one_yes_nothing)
ofa_pred["answer"] = ofa_pred["answer"].map(lemmatize)
