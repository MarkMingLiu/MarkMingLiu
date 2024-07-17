import pyarrow.parquet as pq 
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from rich import progress

pq_file = '/ofs/aml/liuming/gpt/llms_small/data/my_dataset.parquet'
pf = pq.read_table(pq_file)

def get_training_corpus():
    buffer = []
    for prompt, response in progress.track(zip(pf['prompt'], pf['response']), total=pf.num_rows):

        buffer.append(
            f"{prompt.as_py()}\n{response.as_py()}"
        )

        if len(buffer) >= 1000:
             yield buffer
             buffer = []

    if buffer: yield buffer
iter_training_corpus = get_training_corpus()
old_tokenizer = AutoTokenizer.from_pretrained('t5-base')
tokenizer = old_tokenizer.train_new_from_iterator(iter_training_corpus, vocab_size=40960)
tokenizer.save_pretrained('../model_save/my_tokenizer_wiki')
