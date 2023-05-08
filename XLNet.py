# Construct an XLNet model
xlnet_model = xlnet.XLNetModel(
    xlnet_config=xlnet_config,
    run_config=run_config,
    input_ids=input_ids,
    seg_ids=seg_ids,
    input_mask=input_mask)

# Get a summary of the sequence using the last hidden state
summary = xlnet_model.get_pooled_out(summary_type="last")

# Get a sequence output
seq_out = xlnet_model.get_sequence_output()

# build your applications based on `summary` or `seq_out`

######################Tokenization##########################

import sentencepiece as spm
from prepro_utils import preprocess_text, encode_ids

# some code omitted here...
# initialize FLAGS

text = "An input text string."

sp_model = spm.SentencePieceProcessor()
sp_model.Load(FLAGS.spiece_model_file)
text = preprocess_text(text, lower=FLAGS.uncased)
ids = encode_ids(sp_model, text)