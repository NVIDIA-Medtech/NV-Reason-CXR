import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, Any, List

import torch

import transformers
from transformers import set_seed, AutoProcessor, AutoModelForImageTextToText
from transformers.trainer_utils import get_last_checkpoint

import datasets
from datasets import  Value
from accelerate import PartialState

from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser
)

from transformers import Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info


logger = logging.getLogger(__name__)

@dataclass
class VLMScriptArguments(ScriptArguments):
    '''
    Additional command line arguments for the SFT training script. For a full list of arguments, see the configs/sft_config.yaml file.
    '''

    dataset_path: str = field(default="grpo.jsonl", metadata={"help": "Path to the dataset json file."})
    dataset_root: str = field(default="datalists/", metadata={"help": "Path to the datalists root directory."})
    image_dir: str = field(default="images/",  metadata={"help": "Path to the image directory."})

    max_image_height: int = field(default=476, metadata={"help": "Max height e.g 476."})
    min_image_height: int = field(default=128, metadata={"help": "Min height e.g 128."})


def get_padding_tokens_ids(tokenizer):
    '''
    Get special tokens ids to mask in the loss computation.
    '''

    tokenizer = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer
    image_tokens = ["<|image|>", "<|vision_start|>",  "<|vision_end|>",  "<|vision_pad|>", "<|image_pad|>", "<|video_pad|>"]
    if hasattr(tokenizer, "image_token"):
        image_tokens = image_tokens + [tokenizer.image_token]

    padding_token_ids = tokenizer.convert_tokens_to_ids(image_tokens)
    if hasattr(tokenizer, "pad_token_id"):
        padding_token_ids.append(tokenizer.pad_token_id)

    padding_token_ids = list(filter(None, padding_token_ids))
    padding_token_ids = list(set(padding_token_ids))
    padding_token_ids = torch.IntTensor(padding_token_ids)
    return padding_token_ids


class VLM_SFT_DataCollator:
    '''
    VLM custom data collator for SFT training.
    '''
    def __init__(self, processor):
        self.processor = processor
        self.padding_token_ids = get_padding_tokens_ids(processor) # token_ids to ignore in loss computation

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:

        ## simplify the structure of the examples
        examples = [e["messages"] for e in examples]
        
        ### remove image field if present
        for messages in examples:
            for message in messages:
                for content in message["content"]:
                    if content["type"] == "image":
                        content.pop("text", None)
                    elif content["type"] == "text":
                        content.pop("image", None) #remove image field if added by load_dataset


        # Get the texts and images, and apply the chat template
        texts = [self.processor.apply_chat_template(example, tokenize=False) for example in examples]  

        image_inputs = [process_vision_info(example)[0] for example in examples]  
        image_inputs = None if image_inputs[0] is None else image_inputs

        batch = self.processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)  
  
        labels = batch["input_ids"].clone()  

        labels[torch.isin(labels, self.padding_token_ids)] = -100  # Mask tokens in labels
        batch["labels"] = labels  

        return batch
       

def print_input_config():
    '''
    Prints the user provided input config.
    '''
    args = sys.argv[1:]
    if "--config" in args and PartialState().is_main_process:
        with open(args[args.index("--config")+1], 'r') as file:
            print("Input config:", file.read().strip())

def vlm_data_format_dict(sample, image_dir):
    '''
    Formats the sample data for SFT training.
    '''
    for message in sample["messages"]:
        for content in message["content"]:
            if content["type"] == "image":
                content["image"] = os.path.join("file://" + image_dir, content["image"]) #update image path
                content.pop("text", None)
            elif content["type"] == "text":
                content.pop("image", None) #remove image field
    
    for check_columns in ['id', 'image', 'subject_id', 'study_id', 'solution']:
        if check_columns not in sample:
            sample[check_columns] = None

    return sample


def main(script_args, training_args, model_args):
    '''
    Main function for the SFT training script.
    '''

    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Data parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

   
    ###############
    # Setup dataset
    ###############
    features = datasets.Features({"id": Value('string'), "image": Value('string'), "subject_id": Value('string'), "study_id": Value('string'), "solution": Value('string'), "messages": datasets.List({'role': Value('string'), 'content': datasets.List({'type': Value('string'), 'text': Value('string'), 'image': Value('string')})})})
   
    my_dataset = datasets.load_dataset("json", data_files=os.path.join(script_args.dataset_root, script_args.dataset_path), split='train', streaming=script_args.dataset_streaming, features=features)
    train_dataset = my_dataset.map(vlm_data_format_dict, fn_kwargs={"image_dir": script_args.image_dir})
    train_dataset = my_dataset.cast(features)  

    # if streaming 
    if isinstance(train_dataset, datasets.IterableDataset):
        training_args.dataloader_drop_last = True
        training_args.accelerator_config.dispatch_batches = False
        training_args.ignore_data_skip = True


    ###############
    # Setup model
    ###############
    model = AutoModelForImageTextToText.from_pretrained(
        pretrained_model_name_or_path=model_args.model_name_or_path,
        device_map=None,
        torch_dtype=model_args.torch_dtype,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
    )


    ######################
    # Setup Processor
    ######################
    processor_config={}
    if isinstance(model, Qwen2_5_VLForConditionalGeneration) and script_args.min_image_height is not None and script_args.max_image_height is not None:
        processor_config = {"min_pixels": script_args.min_image_height**2, "max_pixels": script_args.max_image_height**2}

    processor = AutoProcessor.from_pretrained(
        pretrained_model_name_or_path=model_args.model_name_or_path,
        use_fast=True,
        padding_side="right",
        trust_remote_code=model_args.trust_remote_code,
        **processor_config
    ) 


    ###############
    # Setup trainer
    ###############
    data_collator = VLM_SFT_DataCollator(processor)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        processing_class=processor,
    )

   
    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics


    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {"dataset_name": script_args.dataset_name, "tags": ["cxr"]}
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)


    logger.info(f"All done! Congrats reaching this point! Please consider citing this work and starring the repo if you found it useful: https://github.com/NVIDIA-Medtech/NV-Reason-CXR")


if __name__ == "__main__":

    print_input_config()

    # parse arguments
    parser = TrlParser((VLMScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    # set WANDB run name
    if training_args.run_name is None:
        training_args.run_name = training_args.output_dir.split("/")[-1] 
    
    main(script_args, training_args, model_args)
