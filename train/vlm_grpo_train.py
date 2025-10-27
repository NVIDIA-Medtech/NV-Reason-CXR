import logging
import os
import sys
from typing import Dict, Any, List
from dataclasses import dataclass, field, asdict

import torch
import datasets
from datasets import  Value, disable_caching

import transformers
from transformers import set_seed, AutoModelForImageTextToText
from transformers.trainer_utils import get_last_checkpoint

from accelerate import PartialState

from trl import ModelConfig, ScriptArguments, TrlParser, GRPOTrainer, GRPOConfig
from vlm_rewards import (
    accuracy_reward,
    format_reward,
    tag_count_reward,
    accuracy_reward_hard,
    get_soft_overshort_punishment
)

from qwen_vl_utils import process_vision_info
disable_caching() 


logger = logging.getLogger(__name__)

@dataclass
class VLMScriptArguments(ScriptArguments):
    '''
    Additional command line arguments for the GRPO training script. For a full list of arguments, see the cofings/grpo_config.yaml file.
    '''

    reward_funcs: list[str] = field(default_factory=lambda: ["accuracy_reward_hard", "soft_overshort_punishment"], metadata={"help": "List of reward functions."})

    dataset_path: str = field(default="grpo.jsonl", metadata={"help": "Path to the dataset json file."})
    dataset_root: str = field(default="datalists/", metadata={"help": "Path to the datalists root directory."})
    image_dir: str = field(default="images/",  metadata={"help": "Path to the image directory."})
                                  
    max_image_height: int = field(default=476, metadata={"help": "Max height e.g 512."})
    min_image_height: int = field(default=128, metadata={"help": "Min height e.g 128."})



class VLM_GRPO_DataCollator:
    '''
    VLM custom data collator for GRPO training, mainly to loads and resize images on the fly.
    '''
    def __init__(self, image_dir, min_pixels, max_pixels):
        self.image_dir = image_dir
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:

        for e in examples:
            if isinstance(e["image"], str):
                item = { 
                    "role": "user",
                    "content": [
                        {"type": "image", "image": os.path.join("file://" + self.image_dir, e["image"])},
                    ],
                }
                if self.max_pixels is not None:
                    item["content"][0]["max_pixels"] = self.max_pixels 
                    item["content"][0]["min_pixels"] = self.min_pixels 

                image, _ = process_vision_info([item])
                image = image[0]
                e["image"] = image

        return examples
       

def print_input_config():
    '''
    Prints the user provided input config.
    '''
    args = sys.argv[1:]
    if "--config" in args and PartialState().is_main_process:
        with open(args[args.index("--config")+1], 'r') as file:
            print("Input config:", file.read().strip())


def vlm_data_format_grpo(sample):
    '''
    Formats the sample data for GRPO training.
    '''
    prompt = [{"role": "user", "content": sample["conversations"][0]["value"]}]
    output = {"id": sample["id"], "solution": sample["solution"], "prompt": prompt, "image": sample["image"]}
    # print(f"output: {output}")
    return output


def main(script_args, training_args, model_args):
    '''
    Main function for the GRPO training script.
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


    # Check for the last checkpoint if resuming from a previous run
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    ################
    # Load datasets
    ################
    if  PartialState().num_processes > 8:
        datasets.disable_progress_bars() 

    min_pixels = script_args.min_image_height**2 if script_args.min_image_height is not None else None
    max_pixels = script_args.max_image_height**2 if script_args.max_image_height is not None else None

    my_dataset = datasets.load_dataset("json", data_files=os.path.join(script_args.dataset_root, script_args.dataset_path), split='train', streaming=script_args.dataset_streaming)
    train_dataset = my_dataset.map(vlm_data_format_grpo, remove_columns=["conversations"])

    if not isinstance(train_dataset, datasets.IterableDataset):
        logger.info(f"Created dataset mixture with {len(train_dataset)} examples")

    training_args.accelerator_config.dispatch_batches = False 
    if script_args.dataset_streaming: # streaming is not supported with GRPOTrainer yet, but just in case we use it in the future
        train_dataset = train_dataset.shuffle()
        training_args.dataloader_drop_last = True
        training_args.ignore_data_skip = True 

    #############################
    # Setup model
    #############################
    model = AutoModelForImageTextToText.from_pretrained(
        pretrained_model_name_or_path=model_args.model_name_or_path,
        device_map=None,
        torch_dtype=model_args.torch_dtype,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        use_cache=False
    )

    ########################
    # Setup reward functions
    ########################
    REWARD_FUNCS_REGISTRY = {
        "accuracy": accuracy_reward,
        "format": format_reward,
        "tag_count": tag_count_reward,
        "accuracy_reward_hard": accuracy_reward_hard,
        "soft_overshort_punishment": get_soft_overshort_punishment
    }
    reward_funcs = [REWARD_FUNCS_REGISTRY[func] for func in script_args.reward_funcs]


    ###############
    # GRPO trainer
    ###############
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        reward_funcs=reward_funcs,
        train_dataset=train_dataset,
    )

    # streaming is not supported for grpo yet, so we load images on the fly in custom data_collator
    trainer.data_collator = VLM_GRPO_DataCollator(image_dir=script_args.image_dir, min_pixels=min_pixels, max_pixels=max_pixels) 


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
    metrics["train_samples"] = len(train_dataset)
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
    '''
    Main entry point for the GRPO training script.
    '''

    print_input_config()

    parser = TrlParser((VLMScriptArguments, GRPOConfig, ModelConfig)) # parse arguments
    script_args, training_args, model_args = parser.parse_args_and_config()

    # set WANDB run name
    if training_args.run_name is None:
        training_args.run_name = training_args.output_dir.split("/")[-1] 

    main(script_args, training_args, model_args)
