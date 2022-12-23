# CodeGen--模型FineTune训练参考

原文地址：

- [How to fine tune Codegen](https://github.com/salesforce/CodeGen/issues/16)
- [train_deepspeed](https://github.com/salesforce/CodeGen/blob/main/jaxformer/hf/train_deepspeed.py)
- [How to fine tune Github Copilot?](https://discuss.huggingface.co/t/how-to-fine-tune-fine-tune-github-copilot/18889)
- [How to use CodeGen](https://discuss.huggingface.co/t/how-to-use-codegen/21120)



## Train_deepspeed 事例

>  
>
> **引用--"enijkamp"**
>
> 对于torch，我写了一个deepspeed 的小例子，它可以在~24 GB gpu 上训练16B。您需要对此进行健全性测试，优化配置，插入数据加载器，并将权重保存到磁盘.
>
> 



```python
# Minimal example of training the 16B checkpoint on GPU with CPU offloading using deepspeed.

'''
apt install python3.8 python3.8-venv python3.8-dev
python3.8 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools
pip install torch --extra-index-url https://download.pytorch.org/whl/cu113
pip install transformers==4.21.1 datasets==1.16.1 deepspeed==0.7.0
deepspeed --num_gpus=1 train_deepspeed.py
'''

########################################################################################################
## imports
import os
import argparse
import random
import math
from time import time
import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM
import deepspeed

########################################################################################################
## args

DEEPSPEED_CONFIG = \
{
    'fp16': {'enabled': True, 'loss_scale': 0, 'loss_scale_window': 1000, 'initial_scale_power': 12, 'hysteresis': 2, 'min_loss_scale': 1},
    'optimizer': {'type': 'AdamW', 'params': {'lr': 1e-05, 'betas': [0.9, 0.999], 'eps': 1e-08, 'weight_decay': 0.0}},
    'scheduler': {'type': 'WarmupLR', 'params': {'warmup_min_lr': 0, 'warmup_max_lr': 1e-05, 'warmup_num_steps': 100}},
    'zero_optimization': {
        'stage': 3,
        'offload_optimizer': {'device': 'cpu', 'pin_memory': False},
        'offload_param': {'device': 'cpu', 'pin_memory': False},
        'overlap_comm': True,
        'contiguous_gradients': True,
        'sub_group_size': 1e9,
        'reduce_bucket_size': 16777216,
        'stage3_prefetch_bucket_size': 15099494.4,
        'stage3_param_persistence_threshold': 40960,
        'stage3_max_live_parameters': 1e9,
        'stage3_max_reuse_distance': 1e9,
        'stage3_gather_fp16_weights_on_model_save': True
    },
    'train_batch_size': 32,
    'train_micro_batch_size_per_gpu': 2,
    'gradient_accumulation_steps': 16,
    'gradient_clipping': 1.0,
    'steps_per_print': 8,
    'wall_clock_breakdown': False,
    'compression_training': {'weight_quantization': {'shared_parameters': {}, 'different_groups': {}}, 'activation_quantization': {'shared_parameters': {}, 'different_groups': {}}, 'sparse_pruning': {'shared_parameters': {}, 'different_groups': {}}, 'row_pruning': {'shared_parameters': {}, 'different_groups': {}}, 'head_pruning': {'shared_parameters': {}, 'different_groups': {}}, 'channel_pruning': {'shared_parameters': {}, 'different_groups': {}}}
}


def create_args(args=argparse.Namespace()):
    args.seed = 42
    args.model = 'Salesforce/codegen-16B-mono'
    args.deepspeed_config = DEEPSPEED_CONFIG
    args.opt_steps_train = 1000
    return args

########################################################################################################
## train

def train(args):
    #######################
    ## preamble
    set_seed(args.seed)

    #######################
    ## model
    print('initializing model')
    config = AutoConfig.from_pretrained(args.model)
    config.gradient_checkpointing = True
    config.use_cache = False

    model = AutoModelForCausalLM.from_pretrained(args.model, config=config)

    model.train()
    # TODO(enijkamp): we need to set this flag twice?
    model.gradient_checkpointing_enable()

    #######################
    ## deepspeed
    print('initializing deepspeed')
    model_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    model_engine, optimizer, _, _ = deepspeed.initialize(config=args.deepspeed_config, model=model, model_parameters=model_parameters)
    torch.cuda.empty_cache()

    #######################
    ## train
    print('starting training')
    input_ids = torch.randint(low=0, high=10, size=[args.deepspeed_config['train_micro_batch_size_per_gpu'], 1024], dtype=torch.int64).cuda()

    for step in range(args.opt_steps_train+1):
        loss = model_engine(input_ids=input_ids, labels=input_ids).loss
        model_engine.backward(loss)
        model_engine.step()
        print(f'{step} {loss:8.3f}')

########################################################################################################
## preamble

def set_gpus(gpu):
    torch.cuda.set_device(gpu)


def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def set_cuda(deterministic=True):
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic


def get_exp_id(file):
    return os.path.splitext(os.path.basename(file))[0]


def get_output_dir(exp_id):
    import datetime
    t = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    output_dir = os.path.join('output/' + exp_id, t)
    return output_dir


def copy_source(file, output_dir):
    import shutil
    shutil.copyfile(file, os.path.join(output_dir, os.path.basename(file)))

########################################################################################################
## main
def main():
    # preamble
    exp_id = get_exp_id(__file__)
    output_dir = get_output_dir(exp_id)

    # args
    args = create_args()
    args.output_dir = output_dir
    args.exp_id = exp_id

    # output
    os.makedirs(args.output_dir, exist_ok=True)
    copy_source(__file__, args.output_dir)
    
    # train
    train(args=args)

if __name__ == '__main__':
    main()
```





## How to fine tune Codegen

- [How to fine tune Github Copilot?](https://discuss.huggingface.co/t/how-to-fine-tune-fine-tune-github-copilot/18889)
- [transformers](https://github.com/huggingface/transformers)/[examples](https://github.com/huggingface/transformers/tree/main/examples)/[pytorch](https://github.com/huggingface/transformers/tree/main/examples/pytorch)/**language-modeling**/

> **Ivwerra:**
>
> 你可以看一下语言建模的例子。这应该适用于任何自动回归模型，如GPT-2或CodeGen。

### **Language model training**

Fine-tuning (or training from scratch) the library models for language modeling on a text dataset for GPT, GPT-2, ALBERT, BERT, DistilBERT, RoBERTa, XLNet... GPT and GPT-2 are trained or fine-tuned using a causal language modeling (CLM) loss while ALBERT, BERT, DistilBERT and RoBERTa are trained or fine-tuned using a masked language modeling (MLM) loss. XLNet uses permutation language modeling (PLM), you can find more information about the differences between those objectives in our [model summary](https://huggingface.co/transformers/model_summary.html).

There are two sets of scripts provided. The first set leverages the Trainer API. The second set with `no_trainer` in the suffix uses a custom training loop and leverages the 🤗 Accelerate library . Both sets use the 🤗 Datasets library. You can easily customize them to your needs if you need extra processing on your datasets.

**Note:** The old script `run_language_modeling.py` is still available [here](https://github.com/huggingface/transformers/blob/main/examples/legacy/run_language_modeling.py).

The following examples, will run on datasets hosted on our [hub](https://huggingface.co/datasets) or with your own text files for training and validation. We give examples of both below.

### GPT-2/GPT and causal language modeling

The following example fine-tunes GPT-2 on WikiText-2. We're using the raw WikiText-2 (no tokens were replaced before the tokenization). The loss here is that of causal language modeling.

```shell
python run_clm.py \
    --model_name_or_path gpt2 \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --output_dir /tmp/test-clm
```

This takes about half an hour to train on a single K80 GPU and about one minute for the evaluation to run. It reaches a score of ~20 perplexity once fine-tuned on the dataset.

To run on your own training and validation files, use the following command:

```shell
python run_clm.py \
    --model_name_or_path gpt2 \
    --train_file path_to_train_file \
    --validation_file path_to_validation_file \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --output_dir /tmp/test-clm
```

This uses the built in HuggingFace `Trainer` for training. If you want to use a custom training loop, you can utilize or adapt the `run_clm_no_trainer.py` script. Take a look at the script for a list of supported arguments. An example is shown below:

```shell
python run_clm_no_trainer.py \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --model_name_or_path gpt2 \
    --output_dir /tmp/test-clm
```

### RoBERTa/BERT/DistilBERT and masked language modeling

The following example fine-tunes RoBERTa on WikiText-2. Here too, we're using the raw WikiText-2. The loss is different as BERT/RoBERTa have a bidirectional mechanism; we're therefore using the same loss that was used during their pre-training: masked language modeling.

In accordance to the RoBERTa paper, we use dynamic masking rather than static masking. The model may, therefore, converge slightly slower (over-fitting takes more epochs).

```shell
python run_mlm.py \
    --model_name_or_path roberta-base \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --output_dir /tmp/test-mlm
```

To run on your own training and validation files, use the following command:

```shell
python run_mlm.py \
    --model_name_or_path roberta-base \
    --train_file path_to_train_file \
    --validation_file path_to_validation_file \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --output_dir /tmp/test-mlm
```

If your dataset is organized with one sample per line, you can use the `--line_by_line` flag (otherwise the script concatenates all texts and then splits them in blocks of the same length).

This uses the built in HuggingFace `Trainer` for training. If you want to use a custom training loop, you can utilize or adapt the `run_mlm_no_trainer.py` script. Take a look at the script for a list of supported arguments. An example is shown below:

```shell
python run_mlm_no_trainer.py \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --model_name_or_path roberta-base \
    --output_dir /tmp/test-mlm
```

**Note:** On TPU, you should use the flag `--pad_to_max_length` in conjunction with the `--line_by_line` flag to make sure all your batches have the same length.

### Whole word masking

This part was moved to `examples/research_projects/mlm_wwm`.

### XLNet and permutation language modeling

XLNet uses a different training objective, which is permutation language modeling. It is an autoregressive method to learn bidirectional contexts by maximizing the expected likelihood over all permutations of the input sequence factorization order.

We use the `--plm_probability` flag to define the ratio of length of a span of masked tokens to surrounding context length for permutation language modeling.

The `--max_span_length` flag may also be used to limit the length of a span of masked tokens used for permutation language modeling.

Here is how to fine-tune XLNet on wikitext-2:

```shell
python run_plm.py \
    --model_name_or_path=xlnet-base-cased \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --output_dir /tmp/test-plm
```

To fine-tune it on your own training and validation file, run:

```shell
python run_plm.py \
    --model_name_or_path=xlnet-base-cased \
    --train_file path_to_train_file \
    --validation_file path_to_validation_file \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --output_dir /tmp/test-plm
```

If your dataset is organized with one sample per line, you can use the `--line_by_line` flag (otherwise the script concatenates all texts and then splits them in blocks of the same length).

**Note:** On TPU, you should use the flag `--pad_to_max_length` in conjunction with the `--line_by_line` flag to make sure all your batches have the same length.



## How to use CodeGen

- [How to use CodeGen](https://discuss.huggingface.co/t/how-to-use-codegen/21120)

**[rwheel](https://discuss.huggingface.co/u/rwheel)**

Hi!

There are different ways to solve your problem, but I recommend using the Dataset library, it allows you to tokenize the whole corpus in an easy way.
Note that in the code below I read both train/test data at the same time. Then you can get each of them as a dict (dataset[“train”], dataset[“test”]).

Here is a summary of the steps:

1. Read the data with datasets library.
2. Define a preprocess function to tokenize your data in your way. Due to you want to get the data in two different lists (**sentences** that contains *question*, *input_output* and *difficulty* and **labels** that contains *solutions*) I defined two different functions. The function is quite easy, it only join the desired data and tokenize it.
3. Tokenize all data using map function from datasets.



```python
import transformers
from datasets import load_dataset
from transformers import AutoTokenizer

dataset = load_dataset("codeparrot/apps", "all")

def preprocess_function_sentences(data):
    return tokenizer([x + y + z + " " for x, y, z in zip(data["question"], data["input_output"], data["difficulty"])], truncation=True, max_length=128)

def preprocess_function_label(data):
    return tokenizer([" ".join(x) for x in data["solutions"]], truncation=True, max_length=128)

tokenized_dataset_sentence = dataset.map(preprocess_function_sentences,
                                 batched=True,
                                 num_proc=4,
                                 remove_columns=dataset["train"].column_names)

tokenized_dataset_label = dataset.map(preprocess_function_label,
                                 batched=True,
                                 num_proc=4,
                                 remove_columns=dataset["train"].column_names)
```

Hope this can help you!

PS: In the <img src="https://emoji.discourse-cdn.com/apple/hugs.png?v=12" alt=":hugs:" style="zoom:25%;" /> docs you can find more detailed information ([Process text data](https://huggingface.co/docs/datasets/v2.4.0/en/nlp_process#process-text-data)).