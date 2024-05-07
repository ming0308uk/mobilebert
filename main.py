from configuration_bert import BertConfig
from modeling_bert import BertForMaskedLM
import torch

config = BertConfig(
    vocab_size=16_000,  # Word vocab size from MobileBERT
    hidden_size=512,  # Hidden size from MobileBERT
    num_hidden_layers=6,  # Num blocks from MobileBERT
    num_attention_heads=4,  # Num attention heads from MobileBERT
    intermediate_size=1024,  # Intermediate size from MobileBERT
    hidden_act='relu',  # Hidden activation from MobileBERT,
    embedding_size=64,
    normalization_type="rms_norm"
)

model = BertForMaskedLM(config=config)

model.eval()

batch_size = 2
seq_length = 10
dummy_input = torch.randint(0, config.vocab_size, (batch_size, seq_length), dtype=torch.long)

with torch.no_grad():
    output = model(input_ids=dummy_input)

expected_shape = (batch_size, seq_length, config.vocab_size)
