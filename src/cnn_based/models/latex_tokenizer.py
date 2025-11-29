class LaTeXCharTokenizer:
    def __init__(self):
        self.PAD_TOKEN = '<PAD>'
        self.START_TOKEN = '<START>'
        self.END_TOKEN = '<END>'
        self.UNK_TOKEN = '<UNK>'
        
        self.vocab = [self.PAD_TOKEN, self.START_TOKEN, self.END_TOKEN, self.UNK_TOKEN]
        
        latex_chars = list("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
        latex_chars += list("+-=*/()[]{}\\^_.,;:!?'\" \n\t")
        
        self.vocab.extend(latex_chars)
        
        self.char_to_idx = {char: idx for idx, char in enumerate(self.vocab)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        
    def encode(self, text, max_length=None):
        tokens = [self.char_to_idx.get(char, self.char_to_idx[self.UNK_TOKEN]) 
                  for char in text]
        
        tokens = [self.char_to_idx[self.START_TOKEN]] + tokens + [self.char_to_idx[self.END_TOKEN]]
        
        if max_length:
            if len(tokens) < max_length:
                tokens += [self.char_to_idx[self.PAD_TOKEN]] * (max_length - len(tokens))
            else:
                tokens = tokens[:max_length-1] + [self.char_to_idx[self.END_TOKEN]]
        
        return tokens
    
    def decode(self, tokens):
        chars = [self.idx_to_char.get(idx, self.UNK_TOKEN) for idx in tokens]
        chars = [c for c in chars if c not in [self.PAD_TOKEN, self.START_TOKEN, self.END_TOKEN]]
        return ''.join(chars)
    
    def __call__(self, text, max_length=None):
        return self.encode(text, max_length)
    
    def vocab_size(self):
        return len(self.vocab)