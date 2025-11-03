


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from tqdm import tqdm
import time
import torch
from torch.nn.functional import softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification
````



---


```python
nltk.download('punkt', quiet=True)
tqdm.pandas(desc="Processing...")
plt.style.use('ggplot')
pd.set_option('display.max_colwidth', 300)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
start_time = time.time()
```



```python
file_path = 'Amazon Customer Reviews.csv'
try:
    df = pd.read_csv(file_path, index_col='Id', usecols=['Id', 'Score', 'Text'])
    print(f"Loaded {file_path} successfully.")
except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
    raise
except ValueError as e:
    print(f"ValueError: {e}. Attempting to read all columns...")
    df = pd.read_csv(file_path, index_col='Id')
    if not all(col in df.columns for col in ['Score', 'Text']):
        raise ValueError("Required columns 'Score' or 'Text' missing.")
```




---


```python
initial_rows = len(df)
df['Text'] = df['Text'].fillna('')
df['Score'] = (
    pd.to_numeric(df['Score'], errors='coerce')
      .fillna(0)
      .astype(int)
)
df = df[df['Text'].str.strip() != '']
rows_after = len(df)
print(f"Removed {initial_rows - rows_after} rows; Current shape: {df.shape}")
```




---


```python
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
print("BERT model and tokenizer loaded.")
```




---


```python
def get_bert_sentiment(text, max_length=512):
    ...
```



---


```python
bert_scores = df['Text'].progress_apply(get_bert_sentiment)
bert_df = pd.json_normalize(bert_scores)
df = pd.concat([df, bert_df], axis=1)
```





```python
def check_alignment(row):
    ...
df['Alignment_BERT'] = df.progress_apply(check_alignment, axis=1)
```



---


```python
counts = df['Alignment_BERT'].value_counts()
sns.countplot(data=df, y='Alignment_BERT', order=counts.index)
plt.show()
```




```python
neg_examples = df[df['Alignment_BERT'] == 'Disagreement (PosScore-NegText)']
pos_examples = df[df['Alignment_BERT'] == 'Disagreement (NegScore-PosText)']
```



---


```python
df[['Score', 'Text', 'neg', 'neu', 'pos', 'compound', 'Alignment_BERT']].to_csv(output_file)
end_time = time.time()
print(f"--- Script End ---\nElapsed time: {end_time - start_time:.2f} seconds")
```

