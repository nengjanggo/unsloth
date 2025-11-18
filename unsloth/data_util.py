from datasets import Dataset, load_dataset


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
    preceding_prompt_len = len(r'Solve the following math problem step by step. The last line of your response should be of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n')
    trailing_prompt_len = len(r'Remember to put your answer on its own line after ’Answer:’.')
    dataset = dataset.map(lambda x: {
            'prompt': [
                {
                    'role': 'user', 
                    'content': x['prompt'][0]['content'][preceding_prompt_len:-trailing_prompt_len] + MATH_PROMPT_ENG
                }
            ], 
            'solution': x['reward_model']['ground_truth']
        }
    )

    return dataset