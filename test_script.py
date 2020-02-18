import numpy as np
import torch
from transformers import BertTokenizer, BertModel, BertForPreTraining, BertConfig

import pprint

# 京大の日本語BERTのモデルの保存場所の指定
# 重すぎるからGitにはあげない
BERT_MODEL_PATH = './bert/Japanese_L-12_H-768_A-12_E-30_BPE_transformers/'
BERT_VOCAB_PATH = './bert/Japanese_L-12_H-768_A-12_E-30_BPE_transformers/vocab.txt'

# https://qiita.com/masaki_sfc/items/1564cf9122db7ed47096 でのmodelの作り方
config = BertConfig(vocab_size_or_config_json_file=32006,
                    hidden_size=768, num_hidden_layers=12,
                    num_attention_heads=12, intermediate_size=3072)
model = BertForPreTraining(config=config)
model.load_state_dict(torch.load(f"{BERT_MODEL_PATH}pytorch_model.bin"))

# https://yag-ays.github.io/project/pytorch_bert_japanese/ でのmodelの作り方
# model = BertModel.from_pretrained(BERT_MODEL_PATH)


bert_tokenizer = BertTokenizer(BERT_VOCAB_PATH, do_lower_case=False, do_basic_tokenize=False)


def getVector(text):
  # print(text)
  text = text.replace('　', ' ').replace('？', '?').replace('??', '[MASK]')
  # print(text)
  # ここで結局bertの辞書に合う粒度でid付けされる
  ids = bert_tokenizer.convert_tokens_to_ids(["[CLS]"] + (bert_tokenizer.tokenize(text))[:126] + ["[SEP]"])
  tokens_tensor = torch.tensor(ids).reshape(1, -1)

  print('@@ token_ids     :', ids)
  print('@@ tokens_tensor :', tokens_tensor)

  model.eval()
  with torch.no_grad():
    all_encoder_layers, _ = model(tokens_tensor)

  # print('@@ debug')
  # pprint.pprint(len(all_encoder_layers[0]))
  # pprint.pprint(all_encoder_layers.size())

  pooling_layer = 0
  embedding = all_encoder_layers[pooling_layer].numpy()[0]
  res = np.mean(embedding, axis=0)
  # res = np.max(embedding, axis=0)

  print('@@ res of encode :', res)

  if text.find('[MASK]') > -1:
    mask_pos = text.split(' ').index('[MASK]') + 1
    print('@@ index of MASK :', mask_pos)
    print('@@ mask predicate:', [bert_tokenizer.ids_to_tokens[i.item()] for i in all_encoder_layers[0][mask_pos].argsort()[-10:]])


try:
  print('[Usage]')
  print('* スペース区切りで文を入力する')
  print('例）吾輩　は　猫　である　。')
  print('* 一箇所を？？にすると、そこに入りそうな候補を出す')
  print('例）今日　は　良い　天気　で　？？　が　し　たく　なり　ます　ね　。')
  while True:
    s = input('>> ')
    getVector(s)
    print('====================')
except KeyboardInterrupt as identifier:
  pass
