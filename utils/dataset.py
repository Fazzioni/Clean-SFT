from torch.utils.data import Dataset
import torch

class ConversationDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=2048, key='messages', chat_template=None):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_lenght = max_length
        self.key = key
        
        self.chat_template = None
        
        if chat_template is not None:
            with open(chat_template, 'r') as f:
                self.chat_template = f.read()
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        tokenized = self.tokenizer.apply_chat_template(self.dataset[idx][self.key],
                                                       tokenize=True,
                                                       return_tensors='pt',
                                                       truncation=True,
                                                       max_length=self.max_lenght,
                                                       return_dict=True,
                                                       return_assistant_tokens_mask=True,
                                                       padding='max_length',
                                                       chat_template=self.chat_template
                                                      )

        assert 'assistant_masks' in tokenized, "tokenizer.apply_chat_template must return assistant_masks when return_assistant_tokens_mask=True"

        train_mask = tokenized['attention_mask'].bool() & tokenized['assistant_masks'].bool()
        tokenized['labels'] = tokenized['input_ids'].clone()
        tokenized['labels'][~train_mask]  = -100
        tokenized['num_itens_sample'] = tokenized['labels'][train_mask].shape[-1]
        del tokenized['assistant_masks']

        tokenized['input_ids'] = tokenized['input_ids'].squeeze(0)
        tokenized['attention_mask'] = tokenized['attention_mask'].squeeze(0)
        tokenized['labels'] = tokenized['labels'].squeeze(0)
        return tokenized


if __name__ == '__main__':
    import transformers 
    dataset = ConversationDataset([{'messages':[{'role':'user','content':"olá"}, {'role':'assistant','content':"olá, tudo bem?"}]}],
                          tokenizer=transformers.AutoTokenizer.from_pretrained("Biatron/biatron-345m"),
                        )
    sample = dataset[0]
    only_answer = sample['labels'][sample['labels'] != -100]
    print(dataset.tokenizer.decode(sample['input_ids'][0]))
    print(dataset.tokenizer.decode(only_answer))
    
    for a,r in zip(sample['input_ids'][0], sample['labels'][0]):
        assert r == -100 or a == r