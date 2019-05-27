import re
from IPython.display import HTML, display
from torch import nn

# regexp for a before or after span tag, with any html attributes inside
b4='(\<span[^\>]+\>)?' # will match e.g. <span style="color: rgba(255,0,0,{})">
after='(\<\/span\>)?' # will match <\span>

replace_list = [
    # punctuation
    ['\s{}\,'.format(b4), r'\1,'],
    ['\s{}\.'.format(b4), r'\1.'],
    ['\s{}\:'.format(b4), r'\1:'],
    ['\s{}\;'.format(b4), r'\1;'],
    ['\s{}\!'.format(b4), r'\1!'],
    ['\s{}\?'.format(b4), r'\1?'],
    ["\s{}\'{}\s?".format(b4,after), r"\1'\2"],
    ["\s{}\-{}\s".format(b4,after), r"\1-\2"],
    ["\“{}\s".format(after), r"“\1"],
    ["\s{}\”".format(b4), r"\1”"],
    ["\s{}\’".format(b4), r"\1’"],
    
    # tokenization
    ['\s?{}\#\#'.format(b4), r'\1'],    

    ["\s?\¿\s?", r""],
#     [UNK]
#     [MASK]
#     [CLS]
]

replace_tokens_list = [
    ['\[CLS\]\s?', ''],
    ['\s?\[SEP\]\s?', ''],
    ["\[\s?PAD\s?\]\s?", r""],
]

def clean_decoded(tokens, clean_tokens=False):
    s = ' '.join(tokens)
    for a, b in replace_list:
        p = re.search(a, s)
        s = re.sub(a, b, s)
    if clean_tokens:
        for a, b in replace_tokens_list:
            p = re.search(a, s)
            s = re.sub(a, b, s)
    return s



def html_clean_decoded_logits(input_ids, logits, input_mask, label_weights, tokenizer):
    """Format model outputs as html, with masked elements in red, with opacity indicating confidence."""
    decoder = {v:k for k,v in tokenizer.wordpiece_tokenizer.vocab.items()}
    log_probs = nn.LogSoftmax(-1)(logits).detach()
    prediction_idxs = log_probs.argmax(-1)
    # join masked an non masked
    y = input_ids *  (1 - label_weights) + prediction_idxs * label_weights
    yd = [decoder[hh.item()] for hh in y]
    html_yd = []
    for i in range(len(yd)):
        if not label_weights[i]:
#             if yd[i] == '[SEP]':
#                 # remove all after the [SEP]
#                 break
            html_yd.append(yd[i])
        else:
            prob = log_probs[i][prediction_idxs[i]].exp()
            prob = prob/2 + 0.5
            html_yd.append('<span style="color: rgba(255,0,0,{})">{}</span>'.format(prob, yd[i]))
    return clean_decoded(html_yd)

def html_clean_decoded(tokens, input_mask, label_weights, tokenizer):
    """Format model outputs as html, with masked elements in red, with opacity indicating confidence."""
    decoder = {v:k for k,v in tokenizer.wordpiece_tokenizer.vocab.items()}
    yd = [decoder[hh.item()] for hh in tokens]
    html_yd = []
    for i in range(len(yd)):
        if not label_weights[i]:
            if yd[i] == '[SEP]':
                # remove all after the [SEP]
                break
            html_yd.append(yd[i])
        else:
            prob = 1
            html_yd.append('<span style="color: rgba(255,0,0,{})">{}</span>'.format(prob, yd[i]))
    return clean_decoded(html_yd)


def text_clean_decoded(tokens, input_mask, label_weights, tokenizer):
    """Format model outputs as html, with masked elements in red, with opacity indicating confidence."""
    decoder = {v:k for k,v in tokenizer.wordpiece_tokenizer.vocab.items()}
    yd = [decoder[hh.item()] for hh in tokens]
    html_yd = []
    return clean_decoded(yd, clean_tokens=True)
