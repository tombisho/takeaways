
from fastai import *
from fastai.text import *
import string
from unidecode import unidecode

df = pd.read_csv('07_07_2020_final_ff.csv')

valid_idx = df[df.valid].index

train_idx = df[~df.valid].index

bal_idx = []

for k, v in zip(df.index, df.bal):
	bal_idx += [k]*v

class LetterTokenizer(BaseTokenizer):
    def __init__(self, lang): pass
    def tokenizer(self, t:str) -> List[str]:
        out = []
        i = 0
        while i < len(t):
            if t[i:].startswith(BOS):
                out.append(BOS)
                i += len(BOS)
            else:
                out.append(t[i])
                i += 1
        return out
    def add_special_cases(self, toks:Collection[str]): pass

itos = [UNK, BOS] + list(string.ascii_lowercase + " -'@&)(." +"0123456789")
vocab=Vocab(itos)
tokenizer=Tokenizer(LetterTokenizer, pre_rules=[], post_rules=[])
train_df = df.iloc[train_idx, [3,2]]
bal_df = df.iloc[bal_idx, [3,2]]
valid_df = df.iloc[valid_idx, [3,2]]



data = TextClasDataBunch.from_df(path='.', train_df=bal_df, valid_df=valid_df,
                         tokenizer=tokenizer, vocab=vocab,
                         mark_fields=False, bs=128)

learn = text_classifier_learner(data, AWD_LSTM,drop_mult=1.0)
learn.load_encoder('just_eat_enc_bs128_2')
learn.unfreeze()

lr = 1e-2
moms = (0.7,0.8)
scale = (1**4)
cycles = 20

learn.fit_one_cycle(cycles, lr, moms=moms)
learn.fit_one_cycle(cycles, lr, moms=moms)



learn.save('07_07_2020_final_ff')

