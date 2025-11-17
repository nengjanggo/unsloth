import os
os.environ['UNSLOTH_VLLM_STANDBY'] = '1'
from unsloth import FastLanguageModel
from accuracy_rewards import accuracy_reward
from datasets import Dataset, load_dataset
from datetime import datetime
from trl import GRPOConfig, GRPOTrainer


MATH_PROMPT = r"""Please reason step by step, and put your final answer within \boxed{}."""
MATH_PROMPT_ENG = r"""Please reason step by step in English, and put your final answer within \boxed{}."""
MATH_PROMPT_KOR = r"""Please reason step by step in Korean, and put your final answer within \boxed{}."""


def shuffle_and_select_data(
    dataset: Dataset,
    num_samples: int | None = None
) -> Dataset:
    dataset = dataset.shuffle()

    if num_samples is not None:
        assert num_samples <= len(dataset)
        dataset = dataset.select(range(num_samples))

    return dataset


def get_dapomath17k(
    num_samples: int | None = None
) -> Dataset:
    dataset = load_dataset('BytedTsinghua-SIA/DAPO-Math-17k')['train']
    dataset = shuffle_and_select_data(dataset, num_samples)
    trailing_prompt_len = len(r'Remember to put your answer on its own line after ’Answer:’.')
    dataset = dataset.map(lambda x: {
            'prompt': [
                {
                    'role': 'user', 
                    'content': x['prompt'][0]['content'][:-trailing_prompt_len] + MATH_PROMPT_ENG
                }
            ], 
            'solution': x['reward_model']['ground_truth']
        }
    )

    return dataset


def main():
    model_name = 'Qwen/Qwen3-4B-Instruct-2507'
    max_prompt_length = 1024
    max_completion_length = 1024 * 2
    max_length = max_prompt_length + max_completion_length
    lora_rank = 4

    time = datetime.now().strftime('%Y%m%d-%H%M%S')
    output_dir = f'output/{model_name.split("/")[1]}/{time}'

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_length,
        load_in_4bit = True,
        full_finetuning = False,
        fast_inference = True,
        gpu_memory_utilization = 0.9,
        max_lora_rank = lora_rank
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = lora_rank,
        target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
        lora_alpha = lora_rank * 2,
        use_gradient_checkpointing = 'unsloth'
    )
    
    dataset = get_dapomath17k(1000)
    split_dataset = dataset.train_test_split(test_size=32*4, seed=42)
    train_dataset = split_dataset['train']
    eval_dataset = split_dataset['test']

    config = GRPOConfig(
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        num_generations=8,
        temperature=1.0,
        top_p=1.0,
        top_k=None,
        use_vllm=True,
        beta=0.0,
        num_iterations=1,
        epsilon_high=0.28,
        scale_rewards='group',
        loss_type='dapo',
        mask_truncated_completions=True,
        top_entropy_quantile=1.0,
        log_completions='rich',
        bf16=True,
        output_dir=output_dir,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        torch_empty_cache_steps=1,
        learning_rate=1e-6,
        max_steps=2,
        lr_scheduler_type='linear',
        warmup_ratio=0.1,
        logging_steps=0.1,
        report_to='wandb',
        # report_to='none',
        eval_strategy='no',
        per_device_eval_batch_size=32,
        bf16_full_eval=True,
        eval_steps=0.2,
        eval_on_start=False
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=accuracy_reward,
        args=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer
    )

    trainer.train()

    model.save_pretrained_merged(output_dir, tokenizer, save_method = 'merged_16bit')


if __name__ == '__main__':
    main()