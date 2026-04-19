class MidiLMConfig:
    def __init__(
        self,
        vocab_size=2048,
        max_seq_len=1024,
        n_layers=6,
        n_heads=8,
        d_model=512,
        d_ff=2048,
        dropout=0.1,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    ):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = dropout
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

    def to_dict(self):
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d):
        return cls(**d)
