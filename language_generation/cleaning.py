import re

def standard_clean(text):
    text = re.sub(r"\\t", "  ", text)
    text = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", text)
    text = re.sub(r"\'s", " \'s", text) 
    text = re.sub(r"\'ve", " \'ve", text) 
    text = re.sub(r"n\'t", " n\'t", text) 
    text = re.sub(r"\'re", " \'re", text) 
    text = re.sub(r"\'d", " \'d", text) 
    text = re.sub(r"\'ll", " \'ll", text) 
    text = re.sub(r",", " , ", text) 
    text = re.sub(r"!", " ! ", text) 
    text = re.sub(r"\(", " \( ", text) 
    text = re.sub(r"\)", " \) ", text) 
    text = re.sub(r"\?", " \? ", text) 
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip().lower()

def mbti_types_clean(text):
	text = text.lower()
	text = re.sub("infp", "<type>", text)
	text = re.sub("infj", "<type>", text)
	text = re.sub("intp", "<type>", text)
	text = re.sub("intj", "<type>", text)

	text = re.sub("isfp", "<type>", text)
	text = re.sub("isfj", "<type>", text)
	text = re.sub("istp", "<type>", text)
	text = re.sub("istj", "<type>", text)

	text = re.sub("esfp", "<type>", text)
	text = re.sub("esfj", "<type>", text)
	text = re.sub("estp", "<type>", text)
	text = re.sub("estj", "<type>", text)

	text = re.sub("enfp", "<type>", text)
	text = re.sub("enfj", "<type>", text)
	text = re.sub("entp", "<type>", text)
	text = re.sub("entj", "<type>", text)

	return text


def clean(text):
	text = mbti_types_clean(text)
	text = standard_clean(text)
	return text