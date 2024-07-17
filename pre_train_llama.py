# from datasets import load_dataset
# import os
# from transformers import (
#     CONFIG_MAPPING,
#     MODEL_FOR_CAUSAL_LM_MAPPING,
#     AutoConfig,
#     AutoModelForCausalLM,
#     AutoTokenizer,
#     TrainerCallback,
#     TrainerState,
#     TrainerControl,
#     HfArgumentParser,
#     Trainer,
#     TrainingArguments,
#     default_data_collator,
#     is_torch_tpu_available,
#     set_seed,
# )

# from transformers import PreTrainedTokenizerFast
# from config import TrainConfig
# from torchdata.datapipes.iter import IterableWrapper

# def get_dataset(extension,tokenizer,streaming, output_dir, train_files,validation_files,validation_split_percentage):
#     block_size = 1024
#     data_files = {}
#     dataset_args = {}

#     if train_files is not None:
#         data_files["train"] = train_files
#     if validation_files is not None:
#         data_files["validation"] = validation_files
#     raw_datasets = load_dataset(
#             extension,
#             data_files=data_files,
#             streaming=streaming,
#             cache_dir=os.path.join(output_dir,'dataset_cache'),
#             use_auth_token=None,
#             **dataset_args,
#         )
#     if "validation" not in raw_datasets.keys():
#         raw_datasets["validation"] = load_dataset(
#                 extension,
#                 data_files=data_files,
#                 split=f"train[:{validation_split_percentage}%]",
#                 cache_dir=None,
#                 use_auth_token= None,
#                 **dataset_args,
#             )
#         raw_datasets["train"] = load_dataset(
#                 extension,
#                 data_files=data_files,
#                 split=f"train[{validation_split_percentage}%:]",
#                 cache_dir=None,
#                 use_auth_token=None,
#                 **dataset_args,
#             )
#     def tokenize_function(examples):
#         output = tokenizer( [ item for item in examples["text"]])
#         return output
#     tokenized_datasets = raw_datasets.map(
#                 tokenize_function,
#                 batched=True,
#                 num_proc=10,
#                 load_from_cache_file=not False,
#                 desc="Running tokenizer on dataset",
#             )
#     def group_texts(examples):
#         from itertools import chain
        
#         # Concatenate all texts.
#         concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
#         # concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
#         total_length = len(concatenated_examples[list(examples.keys())[0]])
#         # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
#         # customize this part to your needs.
#         if total_length >= block_size:
#             total_length = (total_length // block_size) * block_size
#         # Split by chunks of max_len.
#         result = {
#             k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
#             for k, t in concatenated_examples.items()
#         }
#         # print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))       
#         result["labels"] = result["input_ids"].copy()
#         return result
#     lm_datasets = tokenized_datasets.map(
#                 group_texts,
#                 batched=True,
#                 num_proc=10,
#                 load_from_cache_file=not False,
#                 desc=f"Grouping texts in chunks of {block_size}",
#                 batch_size = 40000,
#             )
#     return lm_datasets

# def pre_train(config: TrainConfig):

#     tokenizer = PreTrainedTokenizerFast.from_pretrained(config.tokenizer_dir)

#     model = AutoModelForCausalLM.from_config(config,trust_remote_code=False)

#     tokenizer = PreTrainedTokenizerFast.from_pretrained(config.tokenizer_dir)
#     lm_dataset = get_dataset('csv', tokenizer, streaming=False, output_dir = config.output_dir, train_files = [config.train_file], validation_files = None, validation_split_percentage = 5)
#     train_dataset = lm_dataset['train']
#     eval_dataset = lm_dataset['validation']
#     trainer = Trainer(
#         model=model,
#         train_dataset= IterableWrapper(train_dataset),
#         tokenizer=tokenizer,
#         # Data collator will default to DataCollatorWithPadding, so we change it.
#         data_collator=default_data_collator,
#         compute_metrics=None,
#         preprocess_logits_for_metrics=None,
#         # callbacks=([SavePeftModelCallback] if isinstance(model, PeftModel) else None),
#     )
#     checkpoint = None
#     train_result = trainer.train(resume_from_checkpoint=checkpoint)
#     trainer.save_model()  # Saves the tokenizer too for easy upload

# if __name__ == '__main__':
#     config = TrainConfig()
#     print('config', config)
#     # tokenizer = PreTrainedTokenizerFast.from_pretrained(config.tokenizer_dir)
#     # tokenizer_dir = '/ofs/aml/liuming/gpt/ChatLM-mini-Chinese-main/model_save/model_save/hf_tokenizer/'
#     # tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)

#     # t5_config = get_Llama_config(T5ModelConfig(), vocab_size=len(tokenizer), decoder_start_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
#     # t5_config = get_Llama_config(T5ModelConfig(), vocab_size=50000, decoder_start_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
#     # print('t5_config',t5_config)
#     # model = SmallLlamaModel(t5_config)
#     # print(model)
#     pre_train(config)
