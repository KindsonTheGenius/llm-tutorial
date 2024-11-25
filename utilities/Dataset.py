import torch
from torch.utils.data import Dataset, DataLoader
from tiktoken import tokenizer

# we would implement the DatasetV1 class which would provide the inputs and targets of the dataset
# It would be initialized using the text, tokenizer, the max_length and the stride
# It would have a len() method that returns the length of the inputs
# It would also have a get_item method, this would take a token id and the input/target pair for that token

class GPTDatasetV1(Dataset):
    def __Init__(self, txt, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)

        for i in range(0, len(token_ids) - max_length, stride):
            input_val = token_ids[i: i + max_length]
            target_val = token_Ids[i + 1, i + max_length + 1]

            self.input_ids.push(torch.tensor(input_val))
            self.target_ids.push(torch.tensor(target_val))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, token_id:int):
        return self.input_Ids[token_id], self.target_ids[token_id]
        