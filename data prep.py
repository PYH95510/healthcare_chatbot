import pandas as pd
'''
splits = {'train': 'data/train-00000-of-00001.parquet', 'test': 'data/test-00000-of-00001.parquet', 'validation': 'data/validation-00000-of-00001.parquet'}
df = pd.read_parquet("hf://datasets/openlifescienceai/medmcqa/" + splits["train"])

train_df = pd.read_parquet("hf://datasets/openlifescienceai/medmcqa/" + splits["train"])
test_df = pd.read_parquet("hf://datasets/openlifescienceai/medmcqa/" +splits['test'])
validation_df = pd.read_parquet("hf://datasets/openlifescienceai/medmcqa/" + splits['validation'])

combined_df = pd.concat([train_df, validation_df], ignore_index=True)
trained= pd.concat([test_df], ignore_index=True)

combined_df.to_csv("multiple_healthcare_data.csv", index=False)
trained.to_csv("test_multiple healthcare data.csv",index=False)
'''
'''
validation_df = pd.read_csv('./dataset/validation_data_chatbot.csv')
train_df = pd.read_csv('./dataset/train_data_chatbot.csv')

combined = pd.concat([train_df,validation_df], ignore_index=True)
combined.to_csv("kaggle_healthcare_data.csv", index=False)
'''



