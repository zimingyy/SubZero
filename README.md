# Source code for paper "SubZero: Random Subspace Zeroth-Order Optimization for Memory-Efficient LLM Fine-Tuning"

This is the implementation for the paper [SubZero: Random Subspace Zeroth-Order Optimization for Memory-Efficient LLM Fine-Tuning](http://arxiv.org/abs/2410.08989). 

In this paper, we propose the random Subspace Zeroth-order (SubZero) optimization to address the challenges posed by LLMs’ high dimensionality. We introduce a low-rank perturbation tailored for LLMs that significantly reduces memory consumption while improving training performance. Additionally, we have successfully applied SubZero to four popular fine-tuning schemes for LLMs, including full parameter tuning, LoRA, prefix tuning, and prompt tuning. This demonstrates SubZero's compatibility and versatility across different tuning approaches. 

Furthermore, we prove that our gradient estimation closely approximates the backpropagation gradient, exhibits lower variance than traditional ZO methods, and ensures convergence when combined with SGD. Experimental results show that SubZero enhances fine-tuning performance and achieves faster convergence compared to standard ZO approaches like [MeZO](https://github.com/princeton-nlp/MeZO) across various language modeling tasks.


<p>
  <img src="./figure/subzero.png?raw=true" alt="Fig" width="100%"/>
  <em>
    Visualization of cosine similarity, relative variance, training loss and GPU memory cost on OPT-1.3B under the prompt tuning scheme. SubZero demonstrates reduced angle error and variance in gradient estimation, while also accelerating convergence with minimal additional memory overhead.
  </em>
</p>

## Getting start
- We use python 3.10 and torch 2.1.0, transformers 4.28.1, and cuda 11.8.0.
- pip install -r requirements.txt

## Usage

Use `run.py` for all functions (zero-shot/ICL/fine-tuning/MeZO/SubZero):
```bash
python run.py {ARGUMENTS}
```

Please read `run.py` for a complete list of arguments. We introduce some of the most important ones below. 
* `--num_train`: Number of training examples. For ICL, this is the number of demonstrations.
* `--num_dev`: Number of validation examples.
* `--num_test`: Number of testing examples.
* `--model_name`: HuggingFace model name or path.
* `--task_name`: Task name.
* `--trainer`: can be `none` (zero-shot/ICL), `regular` (fine-tuning), or `zo_sgd` (MeZO) or `subzero_sgd`(SubZero).
* `--train_as_classification`: turn this on for classification tasks (Cross Entropy over likelihood of each class' label words). Otherwise it is LM-style teacher forcing.
* `--zo_eps`: ZO hyperparameter epsilon
* `--prefix_tuning`: use prefix-tuning. 
* `--lora`: use LoRA.
* `--prompt_tuning`: use prompt-tuning.

## Reproducing Results

We provide an example of the OPT-1.3b model performing prompt tuning on the SST-2 dataset.

### MeZO-SGD
`CUDA_VISIBLE_DEVICES=0 python run.py --task_name=SST2 --model_name=facebook/opt-1.3b --output_dir=result/opt1.3b-SST2-prompt-mezo --num_train_epochs=5 --per_device_train_batch_size=16 --load_best_model_at_end --evaluation_strategy=steps --save_strategy=steps --save_total_limit=1    --eval_steps=1000     --max_steps=20000  --logging_steps=10 --num_eval=1000 --num_train=1000 --num_dev=500 --train_as_classification --perturbation_mode=two_side --trainer=zo_sgd    --train_set_seed=0 --lr_scheduler_type=constant --eval_steps=500 --save_steps=500 --prompt_tuning --num_virtual_tokens=10 --prompt_init_by_real_tokens --learning_rate=1e-3 --zo_eps=1e-2     --weight_decay=0`

### SubZero-SGD
`CUDA_VISIBLE_DEVICES=0 python run.py --task_name=SST2 --model_name=facebook/opt-1.3b --output_dir=result/opt1.3b-SST2-prompt-subzero --num_train_epochs=5    --per_device_train_batch_size=16 --load_best_model_at_end --evaluation_strategy=steps --save_strategy=steps --save_total_limit=1 --eval_steps=1000 --max_steps=20000 --logging_steps=10 --num_eval=1000 --num_train=1000 --num_dev=500 --train_as_classification --perturbation_mode=two_side --trainer=subzero_sgd --train_set_seed=0     --lr_scheduler_type=constant --eval_steps=500 --save_steps=500 --prompt_tuning --num_virtual_tokens=10 --prompt_init_by_real_tokens  --learning_rate=1e-3     --zo_eps=1e-2 --weight_decay=0 --gauss_rank=24 --update_interval=1000`

### FO-SGD
`CUDA_VISIBLE_DEVICES=0 python run.py --task_name=SST2 --model_name=facebook/opt-1.3b --output_dir=result/opt1.3b-SST2-prompt-sgd --num_train_epochs=5 --per_device_train_batch_size=16 --load_best_model_at_end --evaluation_strategy=steps --save_strategy=steps --save_total_limit=1 --eval_steps=1000 --max_steps=20000 --logging_steps=10 --num_eval=1000 --num_train=1000 --num_dev=500 --train_as_classification --perturbation_mode=two_side --trainer=sgd --optimizer=sgd --train_set_seed=0 --lr_scheduler_type=constant --eval_steps=500 --save_steps=500 --prompt_tuning --num_virtual_tokens=10 --prompt_init_by_real_tokens --learning_rate=1e-3 --zo_eps=1e-2 --weight_decay=0`

## Acknowledgment

This project is built upon the foundation laid by [MeZO: Fine-Tuning Language Models with Just Forward Passes](https://github.com/princeton-nlp/MeZO) and [Revisiting Zeroth-Order Optimization for Memory-Efficient LLM Fine-Tuning: A Benchmark](https://github.com/ZO-Bench/ZO-LLM/tree/main). The original code from their project is licensed under the [MIT License](https://github.com/princeton-nlp/MeZO/blob/main/LICENSE) and [License](https://github.com/ZO-Bench/ZO-LLM/blob/main/LICENSE) respectively. We would like to thank the authors for their great work and contributions.
