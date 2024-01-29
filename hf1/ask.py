# set hf home to os env
import os
os.environ['HF_HOME'] = './hf1'
os.environ['HF_DATASETS_CACHE'] = './hf1'
# HF_TOKEN
os.environ['HF_TOKEN'] = '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'
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

model = AutoModelForSequenceClassification.from_pretrained('./hf1')
tokenizer = AutoTokenizer.from_pretrained('./hf1') # 替换为你实际使用的预训练模型

pipe = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

print(pipe('Around 0335 GMT , Tab shares were up 19 cents , or 4.4 % , at A $ 4.56 , having earlier set a record high of A $ 4.57 .Tab shares jumped 20 cents , or 4.6 % , to set a record closing high at A $ 4.57 .'))