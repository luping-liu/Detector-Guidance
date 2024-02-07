import re
import spacy
import sys
import torch
import open_clip
import numpy as np


def collate_fn_mto(batch):
    bbox = []
    cls = []
    order = []
    batch_idx = []
    for i in range(len(batch)):
        if len(batch[i]['bbox']) > 0:
            bbox.append(batch[i]['bbox'])
            cls += batch[i]['cls']
            order += batch[i]['order']
            # emb.append(batch[i]['embed'])
            batch_idx += [i] * len(batch[i]['bbox'])
        del batch[i]['bbox']
        del batch[i]['order']
        del batch[i]['cls']
        # del batch[i]['embed']
    result = torch.utils.data.default_collate(batch)
    result['bbox'] = torch.from_numpy(np.concatenate(bbox, axis=0))
    result['cls'] = torch.tensor(cls)[..., None]
    result['order'] = torch.tensor(order)[..., None]
    # result['emb'] = torch.from_numpy(np.concatenate(emb, axis=0))
    result['batch_idx'] = torch.tensor(batch_idx)
    return result


def to_yolo_input(task, yolo_input_dict, batch=None, device=None, precision=None):
    if task == 'mto':
        yolo_input_dict['masks'] = batch['seg'].to(device)
        bboxes = batch['bbox'].to(device, precision).clamp(0, 1).reshape(-1, 4)
        center = (bboxes[:, 0:2] + bboxes[:, 2:4]) / 2
        scale = bboxes[:, 2:4] - bboxes[:, 0:2]
        bboxes = torch.cat([center, scale], dim=1)
        yolo_input_dict['bboxes'] = bboxes
        yolo_input_dict['batch_idx'] = batch['batch_idx'].to(device)
        yolo_input_dict['cls'] = batch['cls']

    return yolo_input_dict


my_nlp = None
my_token = None
excluded_nouns = ['top', 'bottom', 'beside', 'towards', 'front', 'left', 'right', 'center', 'middle', 'rear',
                  'edge', 'corner', 'periphery', 'interior', 'exterior', 'upstairs', 'downstairs',
                  'sideways', 'diagonal', 'opposite', 'adjacent', 'parallel', 'north', 'south', 'east', 'west',
                  'northeast', 'southeast', 'southwest', 'downward', 'inward', 'outward', 'lengthwise', 'crosswise',
                  'amidst', 'amongst', 'proximity', 'vicinity']


def prompt_parser(prompt):
    global my_nlp
    global my_token
    if my_nlp is None:
        my_nlp = spacy.load("en_core_web_md")
        my_token = open_clip.SimpleTokenizer()

    prompt = prompt.lower()
    doc = my_nlp(prompt)

    noun = []
    noun_phrase = []
    for noun_chunk in doc.noun_chunks:
        n = [token.text for token in noun_chunk
             if token.pos_ in ('NOUN', 'PROPN') and token.text not in excluded_nouns]
        if len(n) == 0:
            continue
        noun.append(" ".join(n))
        noun_phrase.append(noun_chunk.text)

    main = []
    sub = []
    start1, start2 = 0, 1
    for n, p in zip(noun, noun_phrase):
        for m in re.finditer(p, prompt):
            if m.start() > start1:
                sent = prompt[start1:m.start()]
                prompt = prompt.replace(sent, ' ' * len(sent), 1)
                tk = my_token.encode(sent)
                start2 += len(tk)
            sent2 = prompt[m.start():m.end()]
            prompt = prompt.replace(sent2, ' ' * len(sent2), 1)
            tk1 = my_token.encode(sent2)
            tk2 = my_token.encode(n)
            sub.append(list(range(start2, start2 + len(tk1))))
            main.append(list(range(start2 + len(tk1) - len(tk2), start2 + len(tk1))))
            start1 = m.end()
            start2 += len(tk1)
            break

    return main, sub


if __name__ == "__main__":
    caption = "A white cat plays with a brown dog"

    tokens = np.zeros(77)
    tokens_sub = np.zeros(77)
    ids = prompt_parser(caption)
    for i, j in enumerate(ids[0]):
        tokens[j] = i+1
    for i, j in enumerate(ids[1]):
        tokens_sub[j] = i+1
    import pdb; pdb.set_trace()
    print('pass')

