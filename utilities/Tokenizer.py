import re
class TokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:item for i, item in enumerate(vocab)}

    def encode(self, text):
        tokens = re.split(r'([,.:;?!_"()\']|--|\s)', text) # splits the text into tokens
        tokens_clean = [item.strip() for item in tokens if item.strip()] # removes whitespaces
        tokens_clean = [token if token in self.str_to_int else '<|unk|>' for token in tokens_clean]
        token_ids = [self.str_to_int[token] for token in tokens_clean] # returns the token ids by performing a lookup
        return token_ids

    def decode(self, token_ids):
        tokens = [self.int_to_str[value] for value in token_ids] # gets the list of tokens by performing a lookup in the dictionary
        joined_tokens = ' '.join(tokens)  # converts list of tokens into a string
        joined_tokens = re.sub(r'\s+([,.?!"()\'])', r'\1', joined_tokens) # removes whitespaces before special characters
        return joined_tokens

