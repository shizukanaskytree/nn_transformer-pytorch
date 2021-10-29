post:
https://pytorch.org/tutorials/beginner/translation_transformer.html

code:
https://github.com/pytorch/tutorials/blob/master/beginner_source/translation_transformer.py

How to run?

1.

```
pip install spacy
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm
```

https://spacy.io/usage/models#download-pip
https://stackoverflow.com/questions/54334304/spacy-cant-find-model-en-core-web-sm-on-windows-10-and-python-3-5-3-anacon

For example, `en_core_web_sm` is a small English pipeline trained on written web text (blogs, news, comments), that includes vocabulary, syntax and entities.

unk: https://machinelearning.wtf/terms/unk/
UNK, unk, <unk> are variants of a symbol in natural language processing and machine translation to indicate an out-of-vocabulary word. Many language models do calculations upon representations of the n most frequent words in the corpus. Words that are less frequent are replaced with the <unk> symbol.Dec 24, 2017

As it is already mentioned in the comments, in tokenizing and NLP when you see UNK token, it is to indicate unknown word with a high chance. for example, if you want to predict a missing word in a sentence.Aug 17, 2017

BOS:
Based on the structure of the code, and my intuition, I would guess those stand for Beginning Of Sentence and End Of Sentence. Specifically, BOS is only appended when i == 0 (i.e., before anything else), and EOS is only appended when i == len(sent) (i.e., after everything else).



2.

3.

4.

5.

