from datasets import load_dataset

raw_datasets = load_dataset("glue", "mrpc")

for i in range(5):
    print('sentence1')
    print(raw_datasets["train"][i]['sentence1'])
    print('sentence2')
    print(raw_datasets["train"][i]['sentence2'])
    print('label')
    print(raw_datasets["train"][i]['label'])
    print('---------------------------------------')