# 预训练模型的 fine-tuning 方案研究



技术方案参考：

- [Easy GPT-2  fine-tuning with Hugging Face and PyTorch](http://reyfarhan.com/posts/easy-gpt2-finetuning-huggingface/)



## Easy GPT-2 fine-tuning with Hugging Face and PyTorch

**author:**   Rey Farhan;  *30. August 2020*

我正在分享一个 [Colab 笔记本](https://colab.research.google.com/drive/13dZVYEOMhXhkXWfvSMVM1TTtUDrT6Aeh?usp=sharing#scrollTo=vCPohrZ-CTWu)，它说明了使用 Hugging Face 的 [Transformers](https://huggingface.co/docs/transformers/index) 库和 [PyTorch](https://pytorch.org/) 微调 GPT2 过程的基础知识。它旨在作为一个易于理解的介绍如何将 Transformers 与 PyTorch 结合使用，并介绍基本组件和结构，特别是考虑到 GPT2。有很多方法可以让 PyTorch 和 Hugging Face 一起工作，但我想要的东西不会偏离 PyTorch 教程中显示的方法太远。

在开始之前，你应该了解 PyTorch 的基础知识以及训练循环的工作原理。如果你不这样做，这个官方 [PyTorch 教程](https://pytorch.org/tutorials/beginner/nn_tutorial.html)可以作为一个可靠的介绍。熟悉 GPT2 的工作原理可能很有用，但不是必需的。我从 [Chris McCormick 的 BERT 微调教程](https://mccormickml.com/2019/07/22/BERT-fine-tuning/)、Ian Porter 的 [GPT2](https://snappishproductions.com/blog/2020/03/01/chapter-9.5-text-generation-with-gpt-2-and-only-pytorch.html.html) 教程和 [Hugging Face Language 模型微调脚本](https://huggingface.co/transformers/v2.0.0/examples.html#language-model-fine-tuning)中大量借鉴了他们的全部功劳。 Chris 的代码实际上为这个脚本提供了基础——你应该查看他的[教程系列](https://mccormickml.com/tutorials/)以获得更多关于转换器和 NLP 的精彩内容。

我应该提到该脚本未涵盖的内容：

1. 使用 [nlp](https://huggingface.co/) 库加载数据集并设置训练工作流程，这看起来可以很好地简化事情；
2. [Accumulated gradients--累积梯度](https://medium.com/huggingface/training-larger-batches-practical-tips-on-1-gpu-multi-gpu-distributed-setups-ec88c3e51255) - 这提供了比 Colab 允许的更大的有效批量大小(GPT2 是一个大模型，任何超过 2 的批量大小都足以让 Colab 出现 CUDA 内存不足错误)；
3. [Freezing layers--冻结图层](https://github.com/huggingface/transformers/issues/1431)，这是一个只改变所选层的参数的过程，由[ULMFit](https://arxiv.org/abs/1801.06146)过程而闻名；
4. 生成文本时使用“过去”，这在生成连续的文本项时采用先前的状态。我不需要它;
5. [Tensor packing--张量打包](https://snappishproductions.com/blog/2020/03/01/chapter-9.5-text-generation-with-gpt-2-and-only-pytorch.html.html)。这是一种在每个批次中装入尽可能多的训练数据的巧妙方法；
6. [Hyperparameter search--超参数搜索](https://discuss.huggingface.co/t/using-hyperparameter-search-in-trainer/785/10)。我很快就确定了似乎产生体面价值的数值，而没有检查它们是否是最优的。



即使这对你现在没有任何意义，你以后可能会发现自己想知道其中的一些。

最后，值得注意的是，Transformers 库可以发生很大变化，而文档中没有太多警告或记录。如果某些内容与您在文档中看到的内容不符，则很可能事情已经发生了变化。



## GPT-2 Fine-Tuning Tutorial with PyTorch & Huggingface in Colab

**目录**

- Setup
- Create Training Set
- GPT2 Tokenizer
- PyTorch Datasets & Dataloaders
- Finetune the GPT2 Language Model
- Display Model Info
- Saving & Loading Fine-Tuned Model
- Generate Text



### **Setup**

```python
!pip install transformers

import os
import time
import google.colab import drive

import pandas as pd
import seaborn as sns
import numpy as np
import random

import matplotlib.pyplot as plt
% matplotlib inline

import torch
form torch.utils.data import Dataset, DataLoader, random_split, RandomSampler, SequentiaSample
torch.manual_seed(42)

from transformers import GPT2LMHeadModel,  GPT2Config, GPT2Tokenizer
from transformers import AdamW,  get_linear_schedule_with_warmup

import nltk
nltk.download('punkt')
```



### **Create Training Set**

用于微调语言模型的数据是一组大约1000个DJ传记，目的是以相同的一般格式和风格生成这些传记。这个数据不是公开的，所以如果你想使用这个脚本，你必须要有自己的训练集。

```python
# mount Google Drive directory to access the training data
gdrive_dir = '/content/gdrive/'
data_dir = os.path.join(gdrive, "'nlp'", "'text gen demos'")
filename = 'ra_top_1000_full.csv'
drive.mount(gdrive_dir, force_remount=True)

# copy the data to current Colab working directory
!cp $data_dir/$filename .

# load into a data frame
df = pd.read_csv(filename)
print(df)

# remove NA values
df.dropna(inplace=True)
bios = df.bio_main.copy()
bios
```

我们需要了解一下我们的训练文件的长度。

我不打算使用与GPT2相同的标记器，它是一个字节对编码的标记器。相反，我使用一个简单的，只是为了得到一个大致的了解。

```python
doc_lengths = []
for bio in bios:
    tokens = nltk.word_tokenize(bio)
    doc_lengths.append(len(tokens))
doc_lengths = np.array(doc_lengths)
sns.dispaly(doc_lengths)
```

![Figure_1](D:\Onedrive\Work-Documents\Projects\代码补全方向\imgs\GPT2-finetune\Figure_1.png)

```python
# the max token length
len(doc_lengths[doc_lengths > 768])/len(doc_lengths)
# output = 0.1581081081081081
np.average(doc_lengths)
# output = 491.45405405405404
```

即使这些令牌计数与 BPE 令牌生成器的不匹配，我相信大多数 bios 将适合小型 GPT2 模型的 768 嵌入大小限制。



### **GPT2 Tokenizer**

虽然默认值会处理这个问题，但我想我会展示您可以指定一些特殊标记。

```python
# Load the GPT tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>') #gpt2-medium

print("The max model length is {} for this model, although the actual embedding size for GPT small is 768".format(tokenizer.model_max_length))
print("The beginning of sequence token {} token has the id {}".format(tokenizer.convert_ids_to_tokens(tokenizer.bos_token_id), tokenizer.bos_token_id))
print("The end of sequence token {} has the id {}".format(tokenizer.convert_ids_to_tokens(tokenizer.eos_token_id), tokenizer.eos_token_id))
print("The padding token {} has the id {}".format(tokenizer.convert_ids_to_tokens(tokenizer.pad_token_id), tokenizer.pad_token_id))

# output:  The max model length is 1024 for this model, although the actual embedding size for GPT small is 768
# output:  The beginning of sequence token <|startoftext|> token has the id 50258
# output:  The end of sequence token <|endoftext|> has the id 50256
# output:  The padding token <|pad|> has the id 50257
```



### **PyTorch Datasets & Dataloaders**

GPT2 是一个大型模型。 将批量大小增加到 2 以上会导致内存不足问题。 这可以通过累积梯度来缓解，但这超出了这里的范围。

```python
batch_size = 2
```

我正在使用标准的 PyTorch 方法在[数据集类](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)中加载数据。

我将标记器作为参数传递，但通常我会在类中实例化它。

```python
class GPT2Dataset(Dataset):
    def __init__(self, txt_list, tokenizer, gpt2_type='gpt2', max_length=768):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.attn_masks = []
        
        for txt in txt_list:
            encodings_dict = tokenizer('<|startoftext|>'+ txt + '<|endoftext|>', truncation=True, max_length=max_length, padding="max_length")
            self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
            self.attn_masks.append(torch.tensor(encoding_dict['attention_mask']))
            
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx]
```

要了解我是如何使用分词器的，请阅读 [**文档**](https://huggingface.co/docs/transformers/main_classes/tokenizer) 。 我已经将每个生物都包装在 bos 和 eos 令牌中。

传递给模型的每个张量都应该是相同的长度。

如果 bio 短于 768 个令牌，它将使用填充令牌填充到 768 的长度。 此外，将返回一个注意掩码，需要将其传递给模型以告诉它忽略填充标记。

如果 bio 超过 768 个令牌，它将在没有 eos_token 的情况下被截断。 这不是问题。

```python
dataset = GPT2Dataset(bio, tokenizer, max_length=768)

# Split into training and validation sets
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
print('{:>5,} training samples'.format(train_size))
print('{:>5,} validation samples'.format(train_size))
```

```python
# Create the DataLoaders for our training and validation datasets.
# We'll take the training samples in random order.
train_dataloader = DataLoader(
					train_dataset,		# The yraining samples
					sampler = RandomSampler(train_dataset),   # Select batches randomly
    				batch_size = batch_size  # Trains with batch size
				)
# For validation the order doesn't matter, so we'll just read them sequantially.
validation_dataloader = DataLoader(
					val_dataset,	# The validation samples.
    				sampler = SequentialSampler(val_dataset),   # Pull out batches sequentially.
    				batch_size = batch_size    # Evaluate with this bacth size.
				)
```



### **Finetune GPT2 Language Model**

```python
# I'm not really doing anything with the config buheret
configuration = GPT2Config.from_pretrained("gpt2", config=configuration)

# instantiate the model--将模型实例化
model = GPT2LMHeadModel.from_pretrained("gpt2", config=configuration)

# THis step is necessary because I've added some tokens (bos_token, etc) to the embeddings.
# Otherwise the tokenizer and model tensors won't match up
model.resize_token_embeddings(len(tokenizer))

# Tell pytorch to run this model on the GPU.
device = torch.device("cuda")
model.cuda()

# Set the seed value all over the place to make this reproductible.
seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)
```



```python
# some parameters I cooked up that work reasonably well.
epochs = 5
learning_rate = 5e-4
warmup_steps = 1e2
epsilon = 1e-8

# this produces sample output every 100 steps.
sample_every = 100

# Note: AdamW is a class from the huggingface library (as opposed to pytorch)
optimizer  = AdamW(model.parameters(),
                  					lr = learning_rate,
                  					eps = epsilon
                  				)
# Total number of training steps is [number of batches]  X [number of epoches]
# (Note that this is not the same as the number of training samples).
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
# This changes the learning rate as the training loop progresses.
scheduler = get_linear_schedule_with_warmup(optimizer,
                                           num_warmup_steps = warmup_steps,
                                           num_training_steps = total_steps)
```



```python
def format_time(elapsed):
    return str(datatime.timedelta(seconds=int(round((elapsed)))))
```

```python
total_t0 = time.time()
training_stats = []
model = model.to(device)

for epoch_i in range(0, epochs):
    print("")
    print('========= Epoch {:} / {:} ========'.format(epoch_i+1, epochs))
    print("Training...")
    t0 = time.time()
    total_train_loss = 0
    model.train()
    for step, batch in enumerate(train_dataloader):
        b_input_ids = batch[0].to(device)
        b_labels = batch[0].to(device)
        b_masks = batch[1].to(device)
        
        model.zero_grad()
        outputs = model(
            b_input_ids,
            labels = b_labels,
            attention_mask = b_masks,
            token_type_ids=None
        )
        loss = outputs[0]
        batch_loss = loss.item()
        total_train_loss += batch_loss
        
        # Get sample every x batches
        if step % sample_every == 0 and not step ==0:
            elapsed = format_time(time.time() - t0)
            print('  Batch {:>5,}  of   {:>5,},  Loss:  {:>5,},  Elapsed: {:}.'.format(step, len(train_dataloader), batch_loss, elapsed))
            model.eval()
            
            sample_outputs = model.generate(
                bos_token_id=random.randint(1, 30000),
                do_sample=True,
                top_k=50,
                max_length = 200,
                top_p=0.95,
                num_return_sequances=1
            )
            for i, sample_output in enumerate(sample_outputs):
                print("{}:  {}".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))
            model.train()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
    # Calculate the average loss over all of the batches.
    avg_train_loss = total_train_loss / len(train_dataloader)
    
    # Measure how long this epoch took.
    training_time = format_time(time.time() - t0)
    print("")
    print("   Average training loss: {0:.2f}".format(avg_train_loss))
    print("   Training epoch took: {:}".format(training_time))
    # ========================================
    #               Validation
    # ========================================
    print("")
    print("Running Validation..")
    t0 = time.time()
    model.eval()
    total_eval_loss = 0
    nb_eval_steps = 0
    
    # Evaluate data for one epoch
    for batch in validation_dataloader:
        b_input_ids = batch[0].to(device)
        b_labels = batch[0].to(device)
        b_masks = batch[1].to(device)
        
        with torch.no_grad():
            outputs = model(b_input_ids, 
                           attention_mask = b_masks,
                           labels=b_labels)
            loss = outputs[0]
        batch_loss = loss.item()
        total_eval_loss += batch_loss
    avg_val_loss = total_eval_loss / len(validation_dataloader)
    validation_time = format_time(time.time() - t0)
    
    print("   Validation Loss:  {0:.2f}".format(avg_val_loss))
    print("   Validation took:  {:}".format(validation_time))
    
    # Read all statistics from this epoch.
    train_stats.append(
    	{
            'epoch': epoch_i +1,
            'Training Loss': avg_train_loss,
            'Valid loss': avg_val_loss,
            'Training Time':  training_time,
            'Validation Time': validation_time
        })
print("")
print("Training complete!")
print("Total training took {:} (h:mm:ss)".format(format_time() - total_t0))
```

Let's view the summary of the training process.

```python
# Display floats with two decimal places.
pd.set_option('precision', 2)

# Create a DataFrame from our training statistics.
df_stats = pd.DataFrame(data=training_stats)

# Use the 'epoch' as the row index.
df_stats = df_stats.set_index('epoch')

# A hack to forch the column headers to wrap.
#df = df.style.set_table_styles([dict(selector='th', props=[('max-width', '70px')])])

# Display the table
df_stats
```



```python
# Use plot styling from seaborn.
sns.set(style='darkgrid')

# Increase the plot size and font size.
sns.set(font_scale=1.5)
plt.rcParams["figure.figsize"] = (12, 6)

# Plot the learning curve.
plt.plot(df_stats['Training Loss'], 'b-o', label="Training")
plt.plot(df_stats['valid Loss'], 'g-o', label='validation')

# Label the plot.
plt.title("Training & Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.xtricks([1, 2, 3, 4])
plt.show()
```

<img src="D:\Onedrive\Work-Documents\Projects\代码补全方向\imgs\GPT2-finetune\Figure_2.png" alt="Figure_2" style="zoom: 80%;" />



### **Display Model Info**

```python
# Get all of the model's parameters as a list of tuple.
params = list(model.named_parameters())

print('The GPT-2 model has {:} different named parameters. \n'.format(len(params)))

print('====== Embedding Layers =====\n')

for p in params[0:2]:
    print("{:>55} {:>12}".format(p[0], str(tuple(p[1].size()))))

print('\n ===== First Transformer ====\n')

for p in params[2:14]:
    print("{:<55} {:>12}".format(p[0], str(tuple[p[1].size()])))

print('\n==== Output Layer ====\n')

for p in params[-2:]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
```

```txt
The GPT-2 model has 148 different named parameters.

==== Embedding Layer ====

transformer.wte.weight                                  (50259, 768)
transformer.wpe.weight                                   (1024, 768)

==== First Transformer ====

transformer.h.0.ln_1.weight                                   (768,)
transformer.h.0.ln_1.bias                                     (768,)
transformer.h.0.attn.c_attn.weight                       (768, 2304)
transformer.h.0.attn.c_attn.bias                             (2304,)
transformer.h.0.attn.c_proj.weight                        (768, 768)
transformer.h.0.attn.c_proj.bias                              (768,)
transformer.h.0.ln_2.weight                                   (768,)
transformer.h.0.ln_2.bias                                     (768,)
transformer.h.0.mlp.c_fc.weight                          (768, 3072)
transformer.h.0.mlp.c_fc.bias                                (3072,)
transformer.h.0.mlp.c_proj.weight                        (3072, 768)
transformer.h.0.mlp.c_proj.bias                               (768,)

==== Output Layer ====

transformer.ln_f.weight                                       (768,)
transformer.ln_f.bias                                         (768,)
```



### **Saving & Loading Fine-Tuned Model**

```python
# Saving best-practices: if you use default names for the model, you can reload it using from_pretrained()
output_dir = './model_save/'

# Create output directory if needed.
if not os.path.exits(output_dir):
    os.makedirs(output_dir)
    
print("Saving model to %s" % output_dir)

# Save a trained model, configuration and tokenizer using 'saving_pretrained()'.
# They can then be reloaded using 'from_pretrained()'
model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
model_to_save.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
# Good practice: save your training arguments together with the trained model
# torch.save(args, os.path.join(output_dir, 'training_args.bin'))
```

```txt
Saving model to ./model_save/
('./model_save/vocab.json',
 './model_save/merges.txt',
 './model_save/special_tokens_map.json',
 './model_save/added_tokens.json')
```

```python
!ls -l --block-size=K ./model_save/
```

```txt
total 499796K
-rw-r--r-- 1 root root      1K Aug 27 13:16 added_tokens.json
-rw-r--r-- 1 root root      1K Aug 27 13:16 config.json
-rw-r--r-- 1 root root    446K Aug 27 13:16 merges.txt
-rw-r--r-- 1 root root 498451K Aug 27 13:16 pytorch_model.bin
-rw-r--r-- 1 root root      1K Aug 27 13:16 special_tokens_map.json
-rw-r--r-- 1 root root      1K Aug 27 13:16 tokenizer_config.json
-rw-r--r-- 1 root root    878K Aug 27 13:16 vocab.json
```

```python
!ls -l --block-size=M ./model_save/pytorch_model.bin
```

```txt
-rw-r--r-- 1 root root 487M Aug 27 13:16 ./model_save/pytorch_model.bin
```



```python
# Copy the model files to a directory in your Google Drive.
!cp -r ./model_save/ $data_dir

# Load a trained model and vocabulary that you have fine-tuned
model = GPT2MLHeadModel.from_pretrained(output_dir)
tokenizer = GPT2Tokenizer.from_pretrained(output_dir)
model.to(device)
```



### **Generate Text**

```python
model.eval()
prompt = "<|startoftext|>"
generated = torch.tensor(tokenizer.encode(prompt)).unsquenze(0)
generated = generated.to(device)

print(generated)

sample_outputs = model.generate(
						generated,
    					do_sample=True,
    					top_k=50,
    					max_length=300,
    					top_p=0.95,
    					num_return_sequences=3
					)
for i, sample_output in enumerate(sample_outputs):
    prit("{}: {}\n\n".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))
```







## BERT Fine-Tuning Tutorial with PyTorch

By Chris McCormick and Nick Ryan,  22 Jul 2019

修订于3/20/20 - 切换到tokenizer.encode_plus并增加了验证损失。详见最后的修订历史。

在本教程中，我将向你展示如何使用BERT与huggingface PyTorch库来快速有效地微调一个模型，以获得接近最先进的句子分类性能。更广泛地说，我描述了转移学习在NLP中的实际应用，以最小的努力在一系列NLP任务中创建高性能模型。

这篇文章以两种形式呈现--这里是一篇[博文](https://mccormickml.com/2019/07/22/BERT-fine-tuning/)，这里是[Colab笔记本](https://colab.research.google.com/drive/1pTuQhug6Dhl9XalKB0zUGf4FIdYFlpcX)。

两者的内容相同，但是：

- 博客文章包括一个评论部分供讨论。
- Colab笔记本将允许你运行代码，并在阅读过程中检查它。



### Contents:

- Contents
- Introduction
  - History
  - What is BERT?
  - Advantages of Fine-Tuning
    - A Shift in NLP
- 1、Setup
- 2、Loading CoLA Dataset
  - 2.1、Download & Extract
  - 2.2、Parse
- 3、Tokenization & Input Formatting
  - 3.1、BERT Tokenizer
  - 3.2、Required Formatting
    - Special Tokens
    - Sentence Length & Attention Mask
  - 3.3、Tokenization & Input Formatting
    - 3.1、BERT Tokenizer
    - 3.2、Required Formatting
      - Special Tokens
      - Sentence Length & Attention Mask
    - 3.3、Tokenize Dataset
    - 3.4、Training & Validation Split
  - 4、Train Our Classification Model
    - 4.1、BertForSequenceClassification
    - 4.2、Optimizer & Learning Rate Scheduler
    - 4.3、Training Loop
  - 5、Performance On Test Set
    - 5.1、Data Preparation
    - 5.2、Evaluate on Test Set
  - Conclusion
  - Appendix
    - A1. Saving & Loading Fine-Tuned Model
    - A2. Weight Decay
  - Revision History
    - Further Work
      - Cite



### **Introduction**

#### History

2018 年是 NLP 突破性的一年。迁移学习，尤其是 Allen AI 的 ELMO、OpenAI 的 Open-GPT 和 Google 的 BERT 等模型，允许研究人员通过最少的任务特定微调来打破多个基准，并为 NLP 社区的其他成员提供预训练模型，这些模型可以轻松（使用更少的数据和更少的计算时间）进行微调和实施以产生最先进的结果。不幸的是，对于许多刚开始 NLP 的人，甚至对于一些经验丰富的从业者来说，这些强大模型的理论和实际应用仍然没有得到很好的理解。

What is BERT?

BERT（Bidirectional Encoder Representations from Transformers）于 2018 年底发布，是我们将在本教程中使用的模型，旨在为读者提供更好的理解和实践指导，以便在 NLP 中使用迁移学习模型。 BERT 是一种预训练语言表示的方法，用于创建 NLP 实践者可以免费下载和使用的模型。您可以使用这些模型从文本数据中提取高质量的语言特征，也可以使用自己的数据在特定任务（分类、实体识别、问答等）上微调这些模型，以生成语言状态艺术预测。

这篇文章将解释你如何修改和微调BERT，以创建一个强大的NLP模型，迅速给你带来最先进的结果。



### A Shift in NLP

这种向迁移学习的转变与几年前发生在计算机视觉领域的转变相似。为计算机视觉任务创建一个好的深度学习网络可能需要数百万个参数并且训练成本非常高。研究人员发现，深度网络学习分层特征表示（在最低层的简单特征，如边缘，在较高层逐渐复杂的特征）。与其每次都从头开始训练一个新的网络，一个训练有素的具有广义图像特征的网络的较低层可以被复制并转移到另一个具有不同任务的网络中使用。下载预训练的深度网络并快速重新训练以完成新任务或在顶部添加额外层很快成为一种常见做法——这比从头开始训练网络的昂贵过程更可取。对许多人来说，2018 年深度预训练语言模型（ELMO、BERT、ULMFIT、Open-GPT 等）的引入标志着 NLP 迁移学习的转变与计算机视觉看到的相同。



## <span style='color:brown'>**1、Setup**</span>



### **1.1、Using Colab GPU for Training**

Google Colab提供免费的GPU和TPU! 由于我们将训练一个大型的神经网络，所以最好利用这个优势（在这种情况下，我们将附加一个GPU），否则训练将花费很长的时间。

```python
import tensorflow as tf

# Get the GPU device name.
device_name = tf.test.gpu_device_name()

# The device name should look like the following:
if device_name == '/device:GPU:0':
    print('Found GPU at {}'.format(device_name))
else:
    raise SystenError('GPU device not found')
```

为了让Torch使用GPU，我们需要识别并指定GPU为设备。稍后，在我们的训练循环中，我们将把数据加载到该设备上。

```python
import torch

# If there's a GPU available...
if torch.cuda.is_available():
    # Tell PyTorch to use the GPU.
    device = torch.device('cuda')
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU: ', torch.cuda.get_device_name(0))
# If not ...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")
```



### 1.2、Installing the Hugging Face Library

接下来，让我们安装来自Hugging Face的transformers包，它将为我们提供一个pytorch接口，以便与BERT一起工作。（这个库包含其他预训练语言模型的接口，如OpenAI的GPT和GPT-2）。我们之所以选择pytorch接口，是因为它在高级API（易于使用，但不能深入了解事物的工作原理）和tensorflow代码（包含很多细节，但经常使我们偏离关于tensorflow的课程，而这里的目的是BERT！）之间取得了良好的平衡。

目前，Hugging Face库似乎是使用BERT工作的最广泛接受和强大的pytorch接口。除了支持各种不同的预训练转化器模型外，该库还包括适合你的特定任务的这些模型的预建修改。例如，在本教程中，我们将使用`BertForSequenceClassification`。

该库还包括特定任务的类，用于标记分类、问题回答、下句预测等。使用这些预建的类可以简化为你的目的而修改BERT的过程。

这个 notebook 中的代码实际上是来自 huggingface 的 run_glue.py 示例脚本的简化版本。

run_glue.py 是一个有用的实用程序，它允许您选择要运行的 GLUE 基准测试任务，以及要使用的预训练模型（您可以在此处查看可能模型的列表）。它还支持使用 CPU、单个 GPU 或多个 GPU。如果您想进一步加快速度，它甚至支持使用 16 位精度。不幸的是，所有这些可配置性都以可读性为代价。在这本 Notebook 中，我们大大简化了代码并添加了大量注释，以明确发生了什么。



## <span style='color:brown'>**2、Loading CoLA Dataset**</span>

我们将使用 [The Corpus of Linguistic Acceptability](https://nyu-mll.github.io/CoLA/)（CoLA）数据集进行单句分类。这是一组标记为语法正确或不正确的句子。它首次发布于2018年5月，是 "GLUE基准 "中包含的测试之一，BERT等模型在此基础上进行竞争。



### 2.1、Download & Extract



```python
!pip install wget
```



```python
import wget
import os

print('Dpwnloading dataset...')

# The URL for the dataset zip file.
url = 'https://nyu-mll.github.io/CoLA/cola_public_1.1.zip'

# Download the file (if we haven't already)
if not os.path.exists('./cola_public_1.1.zip'):
    wget.download(url, './cola_public_1.1.zip')
```

将数据集解压到文件系统中。你可以在左边的侧边栏中浏览Colab实例的文件系统。

```python
# Unzip the dataset (if we haven't already)
if not os.path.exists('./cola_public/'):
    !unzip cola_public_1.1.zip
```



### **2.2、Parse**

我们可以从文件名中看到，`tokenized-标记化` 和 `raw-原始版本` 的数据都是可用的。

我们不能使用预标记的版本，因为为了应用预训练的BERT，我们必须使用模型提供的标记器。这是因为（1）模型有一个特定的、固定的词汇，（2）BERT标记器有一个处理词汇外的特殊方式。

我们将使用pandas来解析 "域内 "训练集，并看一下它的一些属性和数据点。

```python
import pandas as pd

# Load the dataset into a pandas dataframe.
df = pd.read_csv("./cola_public/raw/in_domain_train.tsv", delimiter='\t', header=None, names=['sentence_source', 'label', 'label_notes', 'sentence'])

# Report the number of sentences.
print('Number of training sentences: {:,}\n'.format(df.shape[0]))

# Display 10 random rows from the data.
df.sample(10)
```



|      | sentence_source | label | label_notes |                                          sentence |
| :--- | --------------: | ----: | ----------: | ------------------------------------------------: |
| 8200 |            ad03 |     1 |         NaN |                            They kicked themselves |
| 3862 |            ks08 |     1 |         NaN |            A big green insect flew into the soup. |
| 8298 |            ad03 |     1 |         NaN |                              I often have a cold. |
| 6542 |            g_81 |     0 |           * |   Which did you buy the table supported the book? |
| 722  |            bc01 |     0 |           * |                            Home was gone by John. |
| 3693 |            ks08 |     1 |         NaN |   I think that person we met last week is insane. |
| 6283 |            c_13 |     1 |         NaN |                    Kathleen really hates her job. |
| 4118 |            ks08 |     1 |         NaN | Do not use these words in the beginning of a s... |
| 2592 |            l-93 |     1 |         NaN |            Jessica sprayed paint under the table. |
| 8194 |            ad03 |     0 |           * |                                  I sent she away. |



我们真正关心的两个属性是句子及其标签，我们称之为“可接受性判断”（0=不可接受，1=可接受）。

这里有五个被标记为语法上不可接受的句子。请注意，这项任务比情绪分析之类的任务要困难得多！

```python
df.loc[df.label == 0].sample(5)[['sentence', 'label']]
```

|      | sentence                                          | label |
| :--- | :------------------------------------------------ | :---- |
| 4867 | They investigated.                                | 0     |
| 200  | The more he reads, the more books I wonder to ... | 0     |
| 4593 | Any zebras can't fly.                             | 0     |
| 3226 | Cities destroy easily.                            | 0     |
| 7337 | The time elapsed the day.                         | 0     |

让我们将训练集的句子和标签提取为 numpy ndarrays。

```python
# Get the lists of sentences and their labels.
sebtences = df.sentence.values
labels = df.label.values
```



## <span style='color:brown'>**3、Tokenization & Input Formatting**</span>

在这一节中，我们将把我们的数据集转换成BERT可以训练的格式。

### 3.1、BERT Tokenizer

为了将我们的文本送入BERT，必须将其分割成标记，然后必须将这些标记映射到标记器词汇中的索引。

标记化必须由BERT所包含的标记化器来执行--下面的单元格将为我们下载这个。我们将在这里使用 "非套用 "版本。

```python
from transformers import BertTokenizer

# Load the BERT tokenizer.
print('Loading BERT tokenizer...')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
```



> NOTE:    **bert-base-uncased vs bert-base-cased**
>
> 在 BERT uncased 中，文本在 WordPiece 标记化步骤之前已小写，而在 BERT 中，文本与输入文本相同（没有变化）。
>
> 例如，如果输入是“OpenGenus”，那么对于 BERT uncased，它将转换为“opengenus”，而 BERT cased 则采用“OpenGenus”。
>
> ```txt
> # BERT uncased
> OpenGenus -> opengenus
> 
> # BERT cased
> OpenGenus
> ```
> 在 BERT uncased 中，我们去掉了所有重音标记，而在 BERT 中，保留了重音标记。
> 重音标记是通常在拉丁语中使用的字母上的标记。
>
> 在重音标记方面，我们有：
>
> ```txt
> BERT uncased
> OpènGènus -> opengenus
> 
> # BERT cased
> OpènGènus
> ```
>
> 请注意上面示例中的字母“e”。它上面有一个重音标记。
>
> 

让我们对一个句子应用标记化器，看看输出结果。

```python
# Print the priginal sentence.
print('Original: ', sentences[0])

# Print the sentence split into tokens.
print('Tokenized: ', tokenizer.tokenize(sentences[0]))

# Print the sentence mapped to token ids.
print('Token IDs: ', tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentences[0])))
```

```txt
 Original:  Our friends won't buy this analysis, let alone the next one we propose.
Tokenized:  ['our', 'friends', 'won', "'", 't', 'buy', 'this', 'analysis', ',', 'let', 'alone', 'the', 'next', 'one', 'we', 'propose', '.']
Token IDs:  [2256, 2814, 2180, 1005, 1056, 4965, 2023, 4106, 1010, 2292, 2894, 1996, 2279, 2028, 2057, 16599, 1012]
```

当我们实际转换所有的句子时，我们将使用 `tokenize.encode` 函数来处理这两个步骤，而不是分别调用 `tokenize` 和 `convert_tokens_to_ids` 。

不过，在这之前，我们需要谈一谈BERT的一些格式要求。

### **3.2、Required Formatting**

上面的代码遗漏了一些必要的格式化步骤，我们将在这里看一下。

旁注：对我来说，BERT 的输入格式似乎“过度指定”了……我们需要给它一些看起来多余的信息，或者在我们没有明确提供的情况下很容易从数据中推断出它们。但它就是这样，我怀疑一旦我对 BERT 内部有更深入的了解，它会更有意义。

我们必须：

1. 在每个句子的开头和结尾添加特殊标记
2. 将所有句子填充并截断为单个恒定长度
3. 用 "注意力掩码 "明确区分真实标记和填充标记

#### **Special Tokens**

<span style='color:brown'>`[SEP]`</span>

在每个句子的末尾，我们需要附加特殊的 <span style='color:brown'>`[SEP]`</span> 标记。

这个标记是双句子任务的产物，即给BERT两个独立的句子，并要求他确定一些事情（例如，句子A中的问题的答案可以在句子B中找到吗？）

我还不确定当我们只有单句输入时，为什么还需要标记，但它确实是这样的!

<span style='color:brown'>`[CLS]`</span>

对于分类任务，我们必须在每个句子的开头预留特殊的 <span style='color:brown'>`[CLS]`</span> 标记。

这个令牌具有特殊的意义。 BERT 由 12 个 Transformer 层组成。每个转换器接收一个令牌嵌入列表，并在输出上产生相同数量的嵌入（当然，特征值会改变！）。

<img src="D:\Onedrive\Work-Documents\Projects\代码补全方向\imgs\BERT-finetune\Figure_1.png" alt="Figure_1" style="zoom:60%;" />

<center>Figure 1.  在最后（第12个）转化器的输出上，只有第一个嵌入（对应于[CLS]标记）被分类器使用。</center>

> “每个序列的第一个标记始终是一个特殊的分类标记 ([CLS])。与该标记对应的最终隐藏状态用作分类任务的聚合序列表示。”  (from the [BERT paper](https://arxiv.org/pdf/1810.04805.pdf))

你可能会考虑在最终嵌入上尝试一些池化策略，但这不是必需的。因为 BERT 被训练为只使用这个 [CLS] 标记进行分类，我们知道该模型被激励将分类步骤所需的所有内容编码到单个 768 值嵌入向量中。它已经为我们完成了池化！



### **Sentence Length & Attention Mask**

我们数据集中的句子显然有不同的长度，那么 BERT 是如何处理这个问题的呢？

BERT 有两个约束：

1. 所有句子都必须填充或截断为单个固定长度
2. 最大句子长度为 512 个标记

填充是通过一个特殊的[PAD]标记完成的，它在BERT词汇表中的索引是0。下面的图示展示了填充到8个标记的 "MAX_LEN"。

<img src="D:\Onedrive\Work-Documents\Projects\代码补全方向\imgs\BERT-finetune\Figure_2-padding_and_mask.png" alt="Figure_2-padding_and_mask" style="zoom: 50%;" />

注意力掩码 "是一个简单的1和0的数组，表示哪些标记是填充物，哪些不是（似乎有点多余，不是吗！）。这个掩码告诉BERT中的 "自我关注 "机制，不要将这些PAD标记纳入其对句子的解释中。

然而，最大长度确实影响了训练和评估速度。例如，用Tesla K80：

- `MAX_LEN = 128 --> Training epochs take ~5:28 each`
- `MAX_LEN = 64 --> Training epochs take ~2:57 each`



### **3.3、Tokenize Dataset**

transformers库提供了一个有用的 <span style='color:brown'>`encode` </span>函数，它将为我们处理大部分的解析和数据准备的步骤。

但是，在我们准备好对文本进行编码之前，我们需要确定填充/截断的最大句子长度。

下面的单元格将执行数据集的一次标记化传递，以测量最大句子长度。

```python
max_len = 0 

# For every sentence...
for sent in sentences:
    # Tokenize the text and add `[CLS` and `[SEP]` tokens.
    input_ids = tokenizer.encode(sent, ass_specail_tokens=True)
    # Update the maximum sentence length.
    max_len = max(max_len, len(input_ids))
print('Max setence length: ', max_len)
```

```txt
Max sentence length:  47
```

以防万一有一些更长的测试句子，我将最大长度设置为 64。

现在我们准备好执行真正的标记化了。

tokenizer.encode_plus 函数为我们组合了多个步骤：

1. 将句子拆分为标记
2. 添加特殊的 [CLS] 和 [SEP] 标记
3. 将令牌映射到它们的 ID。
4. 将所有句子填充或截断到相同的长度
5. 创建注意力掩码，明确区分真实令牌和 [PAD] 令牌

前四个功能在 tokenizer.encode 中，但我使用 tokenizer.encode_plus 来获取第五项（注意掩码）。文档在[这里](https://huggingface.co/docs/transformers/main_classes/tokenizer?highlight=encode_plus#transformers.PreTrainedTokenizer.encode_plus)。

```python
# Tokenize all of the sentences and map the tokens to their word IDs.
inputs_ids = []
attention_masks = []

# For every sentence..
for sent in sentences:
    # 'encode_plus' will:
    # 1、Tokenize the sentence.
    # 2、Prepend the `[CLS]` token to the start.
    # 3、Append the '[SEP]' token to the end.
    # 4、Map tokens to their IDs.
    # 5、Pad or truncate the sentence to 'max_length'
    # 6、Create attention masks for [PAD] tokens.
    encoded_dict = tokenizer.encode_plus(
        sent,                         # Sentence to encode.
        add_special_tokens = True,    # Add '[CLS]' and '[SEP]'
        max_length = 64,              # Pad & truncate all sentences.
        pad_to_max_length = True,
        return_attention_mask = True, # Construct attn.masks.
        return_tensors = 'pt',        # Return pytorch tensors.
    )
    # Add the encoded sentence to the list.
    input_ids.append(encoded_dict['input_ids'])
    
    # And its attention mask (simple differentiates padding from non-padding).
    attention_masks.append(encoded_dict['attention_mask'])

# Convert the lists into tensors.
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(labels)

# Print sentence 0, now as a list of IDs.
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(labels)

# Print sentence 0, now as a list of IDs.
print('Oroginal: ', sentences[0])
print('Token IDs: ', input_ids[0])
```

```txt
Original:  Our friends won't buy this analysis, let alone the next one we propose.
Token IDs: tensor([  101,  2256,  2814,  2180,  1005,  1056,  4965,  2023,  4106,  1010,
         2292,  2894,  1996,  2279,  2028,  2057, 16599,  1012,   102,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0])
```



### **3.4、Training & Validation Split**

划分我们的训练集，将 90% 用于训练，10% 用于验证。

```python
from torch.utils.data import TensorDataset, random_split

# Combine the traning inputs into a TensorDataset.
dataset = TensorDataset(input_ids, attention_masks, labels)

# Create a 90-10 train-validation split.

# Calculate the number of samples to include in each set.
train_size = int(0.9 * len(dataset))
val_size = len(datset) - train_size

# Divide the dataset by randomly seleting samples
train_datset, val_dataset = random_split(datset, [train_size, val_size])

print('{:>5, } training samples'.format(train_size))
print('{:>5, } validation samples'.format(val_size))
```

```txt
7,695 training samples
856 validation samples
```

我们还将使用 torch DataLoader 类为我们的数据集创建一个迭代器。这有助于在训练期间节省内存，因为与 for 循环不同，使用迭代器不需要将整个数据集加载到内存中。

```python
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

# DataLoader需要知道我们用于训练的批次大小，所以我们在这里指定它。
# 对于在特定任务上对BERT进行微调，作者建议批次大小为16或32。
batch_size = 32

# 为我们的训练和验证集创建 DataLoader。
# 我们将随机抽取训练样本。
train_dataloader = DataLoader(
    train_dataset,
    sampler = RandomSampler(train_dataset), # # Select batches randomly
    batch_size =  battch_size # Trains with this batch size.
)
# 对于验证，顺序无关紧要，所以我们将按顺序阅读它们。
validation_dataloader = DataLoader(
    val_dataset, # The validation samples.
    sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
    batch_size = batch_size # Evaluate with this batch size.
)
```



## <span style='color:brown'>**4、Train Our Classification Model**</span>

现在我们的输入数据已正确格式化，是时候微调 BERT 模型了。

### **4.1、BertForSequenceClassification**

对于这项任务，我们首先要修改预先训练好的BERT模型，以提供分类输出，然后我们要在我们的数据集上继续训练该模型，直到整个模型从头到尾都非常适合我们的任务。

值得庆幸的是，Huggingface pytorch 实现包括一组为各种 NLP 任务设计的接口。尽管这些接口都建立在训练有素的 BERT 模型之上，但每个接口都有不同的顶层和输出类型，旨在适应其特定的 NLP 任务。

值得庆幸的是，Huggingface pytorch 实现包括一组为各种 NLP 任务设计的接口。尽管这些接口都建立在训练有素的 BERT 模型之上，但每个接口都有不同的顶层和输出类型，旨在适应其特定的 NLP 任务。

以下是为微调提供的当前类列表：

- BertModel
- BertForPreTraining
- BertForMaskedLM
- BertForNextSentencePrediction
- BertForSequenceClassification - The one we'll use.
- BertForTokenClassification
- BertForQuestionAnswering

这些文档可以在[此处](https://huggingface.co/transformers/v2.2.0/model_doc/bert.html)找到。

我们将使用 [BertForSequenceClassification](https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#bertforsequenceclassification)。这是普通的 BERT 模型，在顶部添加了一个用于分类的线性层，我们将用作句子分类器。当我们提供输入数据时，整个预训练的 BERT 模型和额外的未经训练的分类层都会针对我们的特定任务进行训练。

好的，让我们加载 BERT！有几种不同的预训练 BERT 模型可用。 “bert-base-uncased”是指只有小写字母的版本（“uncased”），是两者的较小版本（“base”与“large”）。

可以在[此处](https://huggingface.co/transformers/v2.2.0/main_classes/model.html#transformers.PreTrainedModel.from_pretrained)找到 from_pretrained 的文档，并在[此处](https://huggingface.co/transformers/v2.2.0/main_classes/configuration.html#transformers.PretrainedConfig)定义其他参数。

```python
from transformers import BertForSequenceClassification, AdamW, BertConfig
# Load BertForSequenceClassification, the pretrained BERT model with a single 
# linear classification layer on top. 
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",  # Use the 12-layer BERT model, with an uncased vocab.
    num_labels = 2,       # The number of output labels--2 for binary classification.
                          # You can increase this for multi-class tasks. 
    output_attentions = False,     # Whether the model returns attentions weights.
    output_hidden_states = False,  # Whether the model returns all hidden-states.
)

# Tell pytorch to run this model on the GPU.
model.cuda()
```

出于好奇，我们可以在这里按名称浏览模型的所有参数。

在下面的单元格中，我打印了以下重量的名称和尺寸：

1. The embedding layer.
2. The first of the twelve transformers.
3. The output layer.

```python
# Get all of the model's parameters as a list of tuples.
params = list(model.named_parameters())

print('The BERT model has {:} different named parameters.\n'.format(len(params)))

print('==== Embedding Layer ====\n')

for p in params[0:5]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

print('\n==== First Transformer ====\n')

for p in params[5:21]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

print('\n==== Output Layer ====\n')

for p in params[-4:]:
    print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))
```



```txt
The BERT model has 201 different named parameters.

==== Embedding Layer ====

bert.embeddings.word_embeddings.weight                  (30522, 768)
bert.embeddings.position_embeddings.weight                (512, 768)
bert.embeddings.token_type_embeddings.weight                (2, 768)
bert.embeddings.LayerNorm.weight                              (768,)
bert.embeddings.LayerNorm.bias                                (768,)

==== First Transformer ====

bert.encoder.layer.0.attention.self.query.weight          (768, 768)
bert.encoder.layer.0.attention.self.query.bias                (768,)
bert.encoder.layer.0.attention.self.key.weight            (768, 768)
bert.encoder.layer.0.attention.self.key.bias                  (768,)
bert.encoder.layer.0.attention.self.value.weight          (768, 768)
bert.encoder.layer.0.attention.self.value.bias                (768,)
bert.encoder.layer.0.attention.output.dense.weight        (768, 768)
bert.encoder.layer.0.attention.output.dense.bias              (768,)
bert.encoder.layer.0.attention.output.LayerNorm.weight        (768,)
bert.encoder.layer.0.attention.output.LayerNorm.bias          (768,)
bert.encoder.layer.0.intermediate.dense.weight           (3072, 768)
bert.encoder.layer.0.intermediate.dense.bias                 (3072,)
bert.encoder.layer.0.output.dense.weight                 (768, 3072)
bert.encoder.layer.0.output.dense.bias                        (768,)
bert.encoder.layer.0.output.LayerNorm.weight                  (768,)
bert.encoder.layer.0.output.LayerNorm.bias                    (768,)

==== Output Layer ====

bert.pooler.dense.weight                                  (768, 768)
bert.pooler.dense.bias                                        (768,)
classifier.weight                                           (2, 768)
classifier.bias                                                 (2,)
```



### **4.2、Optimizer & Learning Rate Scheduler**

现在我们已经加载了模型，我们需要从存储的模型中获取训练超参数。

出于微调的目的，作者建议从以下值中进行选择（来自 BERT 论文的附录 A.3）：

> - **Batch size:** 16, 32
> - **Learning rate (Adam):** 5e-5, 3e-5, 2e-5
> - **Number of epochs:** 2, 3, 4

此处选择：

- Batch size: 32 (set when creating our DataLoaders)
- Learning rate: 2e-5
- Epochs: 4 (we’ll see that this is probably too many…)

epsilon参数eps=1e-8是 "一个非常小的数字，以防止在执行中出现除以0的情况"（来自[这里](https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/)）。

你可以在[这里](https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L109)找到 `run_glue.py` 中创建AdamW优化器的内容。

```python
# Note: AdamW is a class from the huggingface library (as opposed to pytorch) 
# I believe the 'W' stands for 'Weight Decay fix"
optimizer = AdamW(model.parameters(),
                  lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                )
```

```python
from transformers import get_linear_schedule_with_warmup

# Number of training epochs. The BERT authors recommend between 2 and 4. 
# 我们选择了运行4，但我们稍后会看到，这可能是对训练数据的过度拟合。
epochs = 4

# Total number of training steps is [number of batches] x [number of epochs]. 
# (Note that this is not the same as the number of training samples).
total_steps = len(train_dataloader) * epochs

# Create the learning rate rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)
```



### **4.3、Training Loop**

下面是我们的训练循环。这里有很多事情要做，但从根本上说，在我们的循环中，每一次传递都有一个 trianing 阶段和一个验证阶段。

> 感谢Stas Bekman提供的关于使用验证损失检测过拟合的见解和代码。

**Training:**

- Unpack our data inputs and labels
- Load data onto the GPU for acceleration
- Clear out the gradients calculated in the previous pass.
  - In pytorch the gradients accumulate by default (useful for things like RNNs) unless you explicitly clear them out.
- Forward pass (feed input data through the network)
- Backward pass (backpropagation)
- Tell the network to update parameters with optimizer.step()
- Track variables for monitoring progress

**Evalution:**

- Unpack our data inputs and labels
- Load data onto the GPU for acceleration
- Forward pass (feed input data through the network)
- Compute loss on our validation data and track variables for monitoring progress

Pytorch对我们隐藏了所有的详细计算，但我们对代码进行了注释，以指出上述步骤在每一行中发生。

> PyTorch也有一些初学者的[教程](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py)，你可能也会觉得有帮助。

定义用于计算准确性的辅助函数。

```python
import numpy as np

# 计算我们的预测与标签的准确性的函数
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)
```

用于将经过时间格式化为 `hh:mm:ss` 的辅助函数

```python
import time
import datatime

def format_time(elapsed):
    """
    Takes a time in seconds and returns a string hh:mm:ss
    """
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    # Format as hh:mm:ss
    return str(datatime.timedelta(seconds=elapesd_rounded))
```

我们准备开始训练了！

```python
import random
import numpy as np

# This training code is based on the `run_glue.py` script here:
# https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L128

# Set the seed value all over the place to make this reproducible.
seed_val = 42

random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

# We'll store a number of quantities such as training and validation loss, 
# validation accuracy, and timings.
training_stats = []

# Measure the total training time for the whole run.
total_t0 = time.time()

# For each epoch...
for epoch_i in range(0, epochs):
    
    # ========================================
    #               Training
    # ========================================
    
    # Perform one full pass over the training set.

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    # Measure how long the training epoch takes.
    t0 = time.time()

    # Reset the total loss for this epoch.
    total_train_loss = 0

    # Put the model into training mode. Don't be mislead--the call to 
    # `train` just changes the *mode*, it doesn't *perform* the training.
    # `dropout` and `batchnorm` layers behave differently during training
    # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
    model.train()

    # For each batch of training data...
    for step, batch in enumerate(train_dataloader):

        # Progress update every 40 batches.
        if step % 40 == 0 and not step == 0:
            # Calculate elapsed time in minutes.
            elapsed = format_time(time.time() - t0)
            
            # Report progress.
            print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

        # Unpack this training batch from our dataloader. 
        #
        # As we unpack the batch, we'll also copy each tensor to the GPU using the 
        # `to` method.
        #
        # `batch` contains three pytorch tensors:
        #   [0]: input ids 
        #   [1]: attention masks
        #   [2]: labels 
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        # Always clear any previously calculated gradients before performing a
        # backward pass. PyTorch doesn't do this automatically because 
        # accumulating the gradients is "convenient while training RNNs". 
        # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
        model.zero_grad()        

        # Perform a forward pass (evaluate the model on this training batch).
        # The documentation for this `model` function is here: 
        # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
        # It returns different numbers of parameters depending on what arguments
        # arge given and what flags are set. For our useage here, it returns
        # the loss (because we provided labels) and the "logits"--the model
        # outputs prior to activation.
        loss, logits = model(b_input_ids, 
                             token_type_ids=None, 
                             attention_mask=b_input_mask, 
                             labels=b_labels)

        # Accumulate the training loss over all of the batches so that we can
        # calculate the average loss at the end. `loss` is a Tensor containing a
        # single value; the `.item()` function just returns the Python value 
        # from the tensor.
        total_train_loss += loss.item()

        # Perform a backward pass to calculate the gradients.
        loss.backward()

        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update parameters and take a step using the computed gradient.
        # The optimizer dictates the "update rule"--how the parameters are
        # modified based on their gradients, the learning rate, etc.
        optimizer.step()

        # Update the learning rate.
        scheduler.step()

    # Calculate the average loss over all of the batches.
    avg_train_loss = total_train_loss / len(train_dataloader)            
    
    # Measure how long this epoch took.
    training_time = format_time(time.time() - t0)

    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
    print("  Training epcoh took: {:}".format(training_time))
        
    # ========================================
    #               Validation
    # ========================================
    # After the completion of each training epoch, measure our performance on
    # our validation set.

    print("")
    print("Running Validation...")

    t0 = time.time()

    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    model.eval()

    # Tracking variables 
    total_eval_accuracy = 0
    total_eval_loss = 0
    nb_eval_steps = 0

    # Evaluate data for one epoch
    for batch in validation_dataloader:
        
        # Unpack this training batch from our dataloader. 
        #
        # As we unpack the batch, we'll also copy each tensor to the GPU using 
        # the `to` method.
        #
        # `batch` contains three pytorch tensors:
        #   [0]: input ids 
        #   [1]: attention masks
        #   [2]: labels 
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)
        
        # Tell pytorch not to bother with constructing the compute graph during
        # the forward pass, since this is only needed for backprop (training).
        with torch.no_grad():        

            # Forward pass, calculate logit predictions.
            # token_type_ids is the same as the "segment ids", which 
            # differentiates sentence 1 and 2 in 2-sentence tasks.
            # The documentation for this `model` function is here: 
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            # Get the "logits" output by the model. The "logits" are the output
            # values prior to applying an activation function like the softmax.
            (loss, logits) = model(b_input_ids, 
                                   token_type_ids=None, 
                                   attention_mask=b_input_mask,
                                   labels=b_labels)
            
        # Accumulate the validation loss.
        total_eval_loss += loss.item()

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Calculate the accuracy for this batch of test sentences, and
        # accumulate it over all batches.
        total_eval_accuracy += flat_accuracy(logits, label_ids)
        

    # Report the final accuracy for this validation run.
    avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
    print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

    # Calculate the average loss over all of the batches.
    avg_val_loss = total_eval_loss / len(validation_dataloader)
    
    # Measure how long the validation run took.
    validation_time = format_time(time.time() - t0)
    
    print("  Validation Loss: {0:.2f}".format(avg_val_loss))
    print("  Validation took: {:}".format(validation_time))

    # Record all statistics from this epoch.
    training_stats.append(
        {
            'epoch': epoch_i + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Valid. Accur.': avg_val_accuracy,
            'Training Time': training_time,
            'Validation Time': validation_time
        }
    )

print("")
print("Training complete!")

print("Total training took {:} (h:mm:ss)".format(format_time(time.time()-total_t0)))
```

```txt
======== Epoch 1 / 4 ========
Training...
  Batch    40  of    241.    Elapsed: 0:00:08.
  Batch    80  of    241.    Elapsed: 0:00:17.
  Batch   120  of    241.    Elapsed: 0:00:25.
  Batch   160  of    241.    Elapsed: 0:00:34.
  Batch   200  of    241.    Elapsed: 0:00:42.
  Batch   240  of    241.    Elapsed: 0:00:51.

  Average training loss: 0.50
  Training epcoh took: 0:00:51

Running Validation...
  Accuracy: 0.80
  Validation Loss: 0.45
  Validation took: 0:00:02

======== Epoch 2 / 4 ========
Training...
  Batch    40  of    241.    Elapsed: 0:00:08.
  Batch    80  of    241.    Elapsed: 0:00:17.
  Batch   120  of    241.    Elapsed: 0:00:25.
  Batch   160  of    241.    Elapsed: 0:00:34.
  Batch   200  of    241.    Elapsed: 0:00:42.
  Batch   240  of    241.    Elapsed: 0:00:51.

  Average training loss: 0.32
  Training epcoh took: 0:00:51

Running Validation...
  Accuracy: 0.81
  Validation Loss: 0.46
  Validation took: 0:00:02

======== Epoch 3 / 4 ========
Training...
  Batch    40  of    241.    Elapsed: 0:00:08.
  Batch    80  of    241.    Elapsed: 0:00:17.
  Batch   120  of    241.    Elapsed: 0:00:25.
  Batch   160  of    241.    Elapsed: 0:00:34.
  Batch   200  of    241.    Elapsed: 0:00:42.
  Batch   240  of    241.    Elapsed: 0:00:51.

  Average training loss: 0.22
  Training epcoh took: 0:00:51

Running Validation...
  Accuracy: 0.82
  Validation Loss: 0.49
  Validation took: 0:00:02

======== Epoch 4 / 4 ========
Training...
  Batch    40  of    241.    Elapsed: 0:00:08.
  Batch    80  of    241.    Elapsed: 0:00:17.
  Batch   120  of    241.    Elapsed: 0:00:25.
  Batch   160  of    241.    Elapsed: 0:00:34.
  Batch   200  of    241.    Elapsed: 0:00:42.
  Batch   240  of    241.    Elapsed: 0:00:51.

  Average training loss: 0.16
  Training epcoh took: 0:00:51

Running Validation...
  Accuracy: 0.82
  Validation Loss: 0.55
  Validation took: 0:00:02

Training complete!
Total training took 0:03:30 (h:mm:ss)
```

|       | Training Loss | Valid. Loss | Valid. Accur. | Training Time | Validation Time |
| :---- | ------------: | ----------: | ------------: | ------------: | --------------: |
| epoch |               |             |               |               |                 |
| 1     |          0.50 |        0.45 |          0.80 |       0:00:51 |         0:00:02 |
| 2     |          0.32 |        0.46 |          0.81 |       0:00:51 |         0:00:02 |
| 3     |          0.22 |        0.49 |          0.82 |       0:00:51 |         0:00:02 |
| 4     |          0.16 |        0.55 |          0.82 |       0:00:51 |         0:00:02 |

请注意，虽然每个时期的训练损失都在下降，但验证损失却在增加！这表明我们训练模型的时间过长，并且对训练数据过度拟合。(作为参考，我们使用了 7,695 个训练样本和 856 个验证样本。)

Validation Loss 是一种比准确度更精确的度量，因为准确度我们不关心确切的输出值，而只关心它落在阈值的哪一侧。

如果我们正在预测正确答案，但信心不足，那么验证损失会捕捉到这一点，而准确性则不会。

```python
import matplotlib.pyplot as plt
%matplotlib inline

import seaborn as sns
# Use plot styling from seaborn.
sns.set(style='darkgrid')

# Increase the plot size and font size.
sns.set(font_scale=1.5)
plt.rcParams["figure.figsize"] = (12,6)

# Plot the learning curve.
plt.plot(df_stats['Training Loss'], 'b-o', label="Training")
plt.plot(df_stats['Valid. Loss'], 'g-o', label="Validation")

# Label the plot.
plt.title("Training & Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.xticks([1, 2, 3, 4])

plt.show()
```

<img src="D:\Onedrive\Work-Documents\Projects\代码补全方向\imgs\BERT-finetune\Figure_3-learning_curve_w_validation_loss.png" alt="Figure_3-learning_curve_w_validation_loss" style="zoom:50%;" />



## <span style='color:brown'>**5、Performance On Test Set**</span>

现在，我们将加载保留数据集，并准备输入，就像我们对训练集所做的那样。然后，我们将使用 [**马修相关系数**](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.matthews_corrcoef.html) 来评估预测，因为这是广大NLP社区用来评估CoLA性能的指标。通过这个指标，+1是最好的分数，-1是最差的分数。这样，我们就可以看到我们在这个特定任务中与最先进的模型相比表现如何。

### 5.1、Data Preparation

我们需要应用我们为训练数据所做的所有相同步骤来准备我们的测试数据集。

```python
import pandas as pd

# Load the dataset into a pandas dataframe.
df = pd.read_csv("./cola_public/raw/out_of_domain_dev.tsv", delimiter='\t', header=None, names=['sentence_source', 'label', 'label_notes', 'sentence'])

# Report the number of sentences.
print('Number of test sentences: {:,}\n'.format(df.shape[0]))

# Create sentence and label lists
sentences = df.sentence.values
labels = df.label.values

# Tokenize all of the sentences and map the tokens to thier word IDs.
input_ids = []
attention_masks = []

# For every sentence...
for sent in sentences:
    # `encode_plus` will:
    #   (1) Tokenize the sentence.
    #   (2) Prepend the `[CLS]` token to the start.
    #   (3) Append the `[SEP]` token to the end.
    #   (4) Map tokens to their IDs.
    #   (5) Pad or truncate the sentence to `max_length`
    #   (6) Create attention masks for [PAD] tokens.
    encoded_dict = tokenizer.encode_plus(
                        sent,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = 64,           # Pad & truncate all sentences.
                        pad_to_max_length = True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                   )
    
    # Add the encoded sentence to the list.    
    input_ids.append(encoded_dict['input_ids'])
    
    # And its attention mask (simply differentiates padding from non-padding).
    attention_masks.append(encoded_dict['attention_mask'])

# Convert the lists into tensors.
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(labels)

# Set the batch size.  
batch_size = 32  

# Create the DataLoader.
prediction_data = TensorDataset(input_ids, attention_masks, labels)
prediction_sampler = SequentialSampler(prediction_data)
prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)
```

```txt
Number of test sentences: 516
```



### **5.2、Evaluate on Test Set**

准备好测试集后，我们可以应用我们的微调模型来生成对测试集的预测。

```python
# Prediction on test set

print('Predicting labels for {:,} test sentences...'.format(len(input_ids)))

# Put model in evaluation mode
model.eval()

# Tracking variables 
predictions , true_labels = [], []

# Predict 
for batch in prediction_dataloader:
  # Add batch to GPU
  batch = tuple(t.to(device) for t in batch)
  
  # Unpack the inputs from our dataloader
  b_input_ids, b_input_mask, b_labels = batch
  
  # Telling the model not to compute or store gradients, saving memory and 
  # speeding up prediction
  with torch.no_grad():
      # Forward pass, calculate logit predictions
      outputs = model(b_input_ids, token_type_ids=None, 
                      attention_mask=b_input_mask)

  logits = outputs[0]

  # Move logits and labels to CPU
  logits = logits.detach().cpu().numpy()
  label_ids = b_labels.to('cpu').numpy()
  
  # Store predictions and true labels
  predictions.append(logits)
  true_labels.append(label_ids)

print('    DONE.')
```

```txt
Predicting labels for 516 test sentences...
    DONE.
```

CoLA 基准的准确性是使用“[马修斯相关系数](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.matthews_corrcoef.html)”（MCC）来衡量的。

我们在这里使用 MCC，因为类是不平衡的：

```python
print('Positive samples: %d of %d (%.2f%%)' % (df.label.sum(), len(df.label), (df.label.sum() / len(df.label) * 100.0)))
```

```txt
Positive samples: 354 of 516 (68.60%)
```

```python
from sklearn.metrics import matthews_corrcoef

matthews_set = []

# Evaluate each test batch using Matthew's correlation coefficient
print('Calculating Matthews Corr. Coef. for each batch...')

# For each input batch...
for i in range(len(true_labels)):
  
  # The predictions for this batch are a 2-column ndarray (one column for "0" 
  # and one column for "1"). Pick the label with the highest value and turn this
  # in to a list of 0s and 1s.
  pred_labels_i = np.argmax(predictions[i], axis=1).flatten()
  
  # Calculate and store the coef for this batch.  
  matthews = matthews_corrcoef(true_labels[i], pred_labels_i)                
  matthews_set.append(matthews)
```

```python
Calculating Matthews Corr. Coef. for each batch...


/usr/local/lib/python3.6/dist-packages/sklearn/metrics/_classification.py:900: RuntimeWarning: invalid value encountered in double_scalars
  mcc = cov_ytyp / np.sqrt(cov_ytyt * cov_ypyp)
```

最终分数将基于整个测试集，但让我们看一下各个批次的分数，以了解批次之间指标的可变性。

每批有 32 个句子，除了最后一批只有 (516 % 32) = 4 个测试句子。

```python
# Create a barplot showing the MCC score for each batch of test samples.
ax = sns.barplot(x=list(range(len(matthews_set))), y=matthews_set, ci=None)

plt.title('MCC Score per Batch')
plt.ylabel('MCC Score (-1 to +1)')
plt.xlabel('Batch #')

plt.show()
```

<img src="D:\Onedrive\Work-Documents\Projects\代码补全方向\imgs\BERT-finetune\Figure_4-mcc_score_by_batch.png" alt="Figure_4-mcc_score_by_batch" style="zoom:50%;" />

现在我们将结合所有批次的结果，计算出我们最终的MCC分数。

```python
# Combine the results across all batches. 
flat_predictions = np.concatenate(predictions, axis=0)

# For each sample, pick the label (0 or 1) with the higher score.
flat_predictions = np.argmax(flat_predictions, axis=1).flatten()

# Combine the correct labels for each batch into a single list.
flat_true_labels = np.concatenate(true_labels, axis=0)

# Calculate the MCC
mcc = matthews_corrcoef(flat_true_labels, flat_predictions)

print('Total MCC: %.3f' % mcc)
```

```txt
Total MCC: 0.498
```

Cool! 在大约半小时内，在没有做任何超参数调整的情况下（调整学习率、历时、批次大小、ADAM属性等），我们就能得到一个好的分数。

> NOTE:
>
> 为了使分数最大化，我们应该去掉 "验证集"（我们用它来帮助确定要训练多少个历时），在整个训练集上进行训练。

该库在此处记录了[此基准](https://huggingface.co/transformers/examples.html#glue)的预期准确度为 49.23。也可以在[此处](https://gluebenchmark.com/leaderboard/submission/zlssuBTm5XRs0aSKbFYGVIVdvbj1/-LhijX9VVmvJcvzKymxy)查看官方排行榜。

请注意（由于数据集较小？）运行之间的准确性可能会有很大差异。



## <span style='color:brown'>**Conclusion**</span>

这篇文章表明，有了预训练的BERT模型，无论你对什么具体的NLP任务感兴趣，你都可以用最小的努力和训练时间，利用pytorch界面快速有效地创建一个高质量的模型。



## <span style='color:brown'>**Appendix**</span>

### A1. Saving & Loading Fine-Tuned Model

第一个单元（取自 run_glue.py [**此处**](https://github.com/huggingface/transformers/blob/35ff345fc9df9e777b27903f11fa213e4052595b/examples/run_glue.py#L495)）将模型和标记器写入磁盘。

```python
import os

# Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()

output_dir = './model_save/'

# Create output directory if needed
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("Saving model to %s" % output_dir)

# Save a trained model, configuration and tokenizer using `save_pretrained()`.
# They can then be reloaded using `from_pretrained()`
model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
model_to_save.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# Good practice: save your training arguments together with the trained model
# torch.save(args, os.path.join(output_dir, 'training_args.bin'))
```

```txt
Saving model to ./model_save/

('./model_save/vocab.txt',
 './model_save/special_tokens_map.json',
 './model_save/added_tokens.json')
```

出于好奇，让我们检查一下文件大小。

```python
!ls -l --block-size=K ./model_save/
```

```txt
total 427960K
-rw-r--r-- 1 root root      2K Mar 18 15:53 config.json
-rw-r--r-- 1 root root 427719K Mar 18 15:53 pytorch_model.bin
-rw-r--r-- 1 root root      1K Mar 18 15:53 special_tokens_map.json
-rw-r--r-- 1 root root      1K Mar 18 15:53 tokenizer_config.json
-rw-r--r-- 1 root root    227K Mar 18 15:53 vocab.txt
```

要在Colab Notebook会话中保存你的模型，请把它下载到你的本地机器上，或者最好是把它复制到你的Google Drive上。

```python
# Mount Google Drive to this Notebook instance.
from google.colab import drive
    drive.mount('/content/drive')
```

```python
# Copy the model files to a directory in your Google Drive.
!cp -r ./model_save/ "./drive/Shared drives/ChrisMcCormick.AI/Blog Posts/BERT Fine-Tuning/"
```

以下函数将从磁盘加载模型。

```python
# Load a trained model and vocabulary that you have fine-tuned
model = model_class.from_pretrained(output_dir)
tokenizer = tokenizer_class.from_pretrained(output_dir)

# Copy the model to the GPU.
model.to(device)
```



### A2. Weight Decay

huggingface 示例包含以下用于启用权重衰减的代码块，但默认衰减率为“0.0”，因此我将其移至附录。

这个模块主要是告诉优化器不要对偏置项（例如，方程$ y = Wx + b $中的$ b $）应用权重衰减。权重衰减是正则化的一种形式--在计算梯度后，我们将其乘以，例如0.99。

```python
# This code is taken from:
# https://github.com/huggingface/transformers/blob/5bfcd0485ece086ebcbed2d008813037968a9e58/examples/run_glue.py#L102

# Don't apply weight decay to any parameters whose names include these tokens.
# (Here, the BERT doesn't have `gamma` or `beta` parameters, only `bias` terms)
no_decay = ['bias', 'LayerNorm.weight']

# Separate the `weight` parameters from the `bias` parameters. 
# - For the `weight` parameters, this specifies a 'weight_decay_rate' of 0.01. 
# - For the `bias` parameters, the 'weight_decay_rate' is 0.0. 
optimizer_grouped_parameters = [
    # Filter for all parameters which *don't* include 'bias', 'gamma', 'beta'.
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.1},
    
    # Filter for parameters which *do* include those.
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
     'weight_decay_rate': 0.0}
]

# Note - `optimizer_grouped_parameters` only includes the parameter values, not 
# the names.
```



## Revision History

**Version 3** - *Mar 18th, 2020* - (current)

- Simplified the tokenization and input formatting (for both training and test) by leveraging the `tokenizer.encode_plus` function. `encode_plus` handles padding *and* creates the attention masks for us.
- Improved explanation of attention masks.
- Switched to using `torch.utils.data.random_split` for creating the training-validation split.
- Added a summary table of the training statistics (validation loss, time per epoch, etc.).
- Added validation loss to the learning curve plot, so we can see if we’re overfitting.
  - Thank you to [Stas Bekman](https://ca.linkedin.com/in/stasbekman) for contributing this!
- Displayed the per-batch MCC as a bar plot.

**Version 2** - *Dec 20th, 2019* - [link](https://colab.research.google.com/drive/1Y4o3jh3ZH70tl6mCd76vz_IxX23biCPP)

- huggingface renamed their library to `transformers`.
- Updated the notebook to use the `transformers` library.

**Version 1** - *July 22nd, 2019*

- Initial version.



## Further Work

- 将 MCC 分数用于“验证准确性”可能更有意义，但我已将其省略，以免在 Notebook 中更早地解释它。
- Seeding——我不相信在训练循环开始时设置种子值实际上会产生可重复的结果……
- MCC 分数似乎在不同的运行中存在很大差异。多次运行此示例并显示差异会很有趣。

### Cite

Chris McCormick and Nick Ryan. (2019, July 22). *BERT Fine-Tuning Tutorial with PyTorch*. Retrieved from [http://www.mccormickml.com](http://www.mccormickml.com/)
