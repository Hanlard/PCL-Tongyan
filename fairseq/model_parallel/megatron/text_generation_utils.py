# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for generating text."""

import copy
import json
import os
import time
from tqdm import tqdm
import torch
import torch.nn.functional as F

from fairseq.model_parallel.megatron import get_args
from fairseq.model_parallel.megatron import get_tokenizer
from fairseq.model_parallel.megatron import mpu
from fairseq.model_parallel.megatron.utils import get_ltor_masks_and_position_ids
import  json

def get_batch(context_tokens):
    """Generate batch from context tokens."""
    args = get_args()
    tokenizer = get_tokenizer()

    # Move to GPU.
    tokens = context_tokens.view(args.batch_size, -1).contiguous().cuda()
    # Get the attention mask and postition ids.
    attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss)

    return tokens, attention_mask, position_ids


def top_k_logits(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ This function has been mostly taken from huggingface conversational
     ai code at
         https://medium.com/huggingface/how-to-build-a-state-of-the-art-
              conversational-ai-with-transfer-learning-2d818ac26313 """

    if top_k > 0:
        # Remove all tokens with a probability less than the
        # last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Cconvert to 1D
        sorted_logits, sorted_indices = torch.sort(
            logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1),
                                        dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token
        # above the threshold
        sorted_indices_to_remove[..., 1:] \
            = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        for i in range(sorted_indices.size(0)):
            indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
            logits[i][indices_to_remove] = filter_value
        
#         print(f"删除字符数:{torch.sum(sorted_indices_to_remove.float(),dim=-1)}")
        
    return logits


def generate_samples_input_from_file(model):

    args = get_args()
    tokenizer = get_tokenizer()

    # Read the sample file and open the output file.
    assert args.sample_input_file is not None, \
        'sample input file is not provided.'
    if mpu.get_model_parallel_rank() == 0:
        fname = open(args.sample_input_file, "r")
        all_raw_text = fname.readlines()
        input_count = len(all_raw_text)
        input_pos = 0
        if args.sample_output_file is None:
            sample_output_file = args.sample_input_file + ".out"
            print('could not find `sample-output-file`, setting '
                  'it to {}'.format(sample_output_file))
        else:
            sample_output_file = args.sample_output_file
        fname_out = open(sample_output_file, "w+")

    context_count = 0
    model.eval()
    with torch.no_grad():
        while True:
            torch.distributed.barrier(group=mpu.get_model_parallel_group())
            terminate_runs = 0

            if mpu.get_model_parallel_rank() == 0:
                raw_text = all_raw_text[input_pos]
                input_pos += 1
                if input_pos == input_count:
                    raw_text = "stop"

                if "stop" in raw_text:
                    terminate_runs = 1
                else:
                    context_tokens = tokenizer.tokenize(raw_text)
                    context_length = len(context_tokens)

                    if context_length >= (args.seq_length // 2):
                        print("\nContext length", context_length,
                              "\nPlease give smaller context (half of the "
                              "sequence length)!", flush=True)
                        continue
            else:
                context_tokens = tokenizer.tokenize("EMPTY TEXT")
                context_length = len(context_tokens)

            terminate_runs_tensor = torch.cuda.LongTensor([terminate_runs])
            torch.distributed.broadcast(terminate_runs_tensor,
                                        mpu.get_model_parallel_src_rank(),
                                        group=mpu.get_model_parallel_group())
            terminate_runs = terminate_runs_tensor[0].item()

            if terminate_runs == 1:
                return

            token_stream = get_token_stream(model, [context_tokens])
            for _, decode_tokens in enumerate(token_stream):
                decode_tokens, _ = decode_tokens
                decode_tokens = decode_tokens[0].cpu().numpy().tolist()

            if mpu.get_model_parallel_rank() == 0:
                os.system('clear')
                print("\nContext:", raw_text, flush=True)
                trim_decode_tokens = tokenizer.detokenize(
                    decode_tokens)[len(raw_text):]
                print("\nMegatron-LM:", trim_decode_tokens, flush=True)

                fname_out.write("\nContext:")
                fname_out.write(raw_text)
                fname_out.write("\n\nMegatron-LM:")
                fname_out.write(trim_decode_tokens)
                fname_out.write("\n")

            raw_text = None

            torch.distributed.barrier(group=mpu.get_model_parallel_group())
            context_count += 1


def generate_samples_interactive(model, print_frequency=24):

    args = get_args()
    tokenizer = get_tokenizer()

    context_count = 0
    model.eval()
    with torch.no_grad():
        while True:
            torch.distributed.barrier(group=mpu.get_model_parallel_group())
            terminate_runs = 0

            if mpu.get_model_parallel_rank() == 0:
                os.system('clear')
                raw_text = input("\nContext prompt (stop to exit) >>> ")
                while not raw_text:
                    print('Prompt should not be empty!')
                    raw_text = input("\nContext prompt (stop to exit) >>> ")

                if "stop" in raw_text:
                    terminate_runs = 1
                else:
                    context_tokens = tokenizer.tokenize(raw_text)
                    context_length = len(context_tokens)

                    if context_length >= (args.seq_length // 2):
                        print("\nContext length", context_length,
                              "\nPlease give smaller context (half of the "
                              "sequence length)!", flush=True)
                        continue
            else:
                context_tokens = tokenizer.tokenize("EMPTY TEXT")
                context_length = len(context_tokens)

            terminate_runs_tensor = torch.cuda.LongTensor([terminate_runs])
            torch.distributed.broadcast(terminate_runs_tensor,
                                        mpu.get_model_parallel_src_rank(),
                                        group=mpu.get_model_parallel_group())
            terminate_runs = terminate_runs_tensor[0].item()

            if terminate_runs == 1:
                return

            token_stream = get_token_stream(model, [context_tokens])
            for counter, decode_tokens in enumerate(token_stream):
                decode_tokens, _ = decode_tokens
                decode_tokens = decode_tokens[0].cpu().numpy().tolist()

                if mpu.get_model_parallel_rank() == 0 and \
                   counter % print_frequency == 0:
                    os.system('clear')
                    print("\nContext:", raw_text, flush=True)
                    trim_decode_tokens = tokenizer.detokenize(
                        decode_tokens)[len(raw_text):]
                    print("\nMegatron-LM:", trim_decode_tokens, flush=True)

            if mpu.get_model_parallel_rank() == 0:
                os.system('clear')
                print("\nContext:", raw_text, flush=True)
                trim_decode_tokens = tokenizer.detokenize(
                    decode_tokens)[len(raw_text):]
                print("\nMegatron-LM:", trim_decode_tokens, flush=True)

            raw_text = None
            torch.distributed.barrier(group=mpu.get_model_parallel_group())
            context_count += 1

            if mpu.get_model_parallel_rank() == 0:
                input("\nPress any key to continue >>>")


def generate_samples_unconditional(model):

    args = get_args()
    tokenizer = get_tokenizer()

    num_samples = args.num_samples
    context_tokens = [[tokenizer.eod]
                      for _ in range(args.batch_size)]
    ctr = 0
    while True:
        start_time = time.time()
        for token_stream in get_token_stream(model,
                                             copy.deepcopy(context_tokens)):
            pass
        if ctr % args.log_interval == 0:
            print('Avg s/batch:',
                  (time.time() - start_time) / min(args.log_interval, ctr + 1))
            start_time = time.time()
        length = len(token_stream)
        token_batch = token_stream[0].cpu().numpy().tolist()
        length_batch = token_stream[1].cpu().numpy().tolist()
        for tokens, length in zip(token_batch, length_batch):
            tokens = tokens[1:length - 1]
            text = tokenizer.detokenize(tokens)
            is_finished = length < args.seq_length - 1
            datum = {'text': text, 'length': length - 1, 'finished': is_finished}
            yield datum
            ctr += 1
            if ctr >= num_samples:
                break
        if ctr >= num_samples:
            break


def generate_and_write_samples_unconditional(model):

    args = get_args()
    assert args.genfile is not None
    with open(args.genfile, 'w') as f:
        for datum in generate_samples_unconditional(model):
            f.write(json.dumps(datum) + '\n')


def pad_batch(batch, pad_id, args):

    context_lengths = []
    for tokens in batch:
        context_length = len(tokens)
        if context_length < args.seq_length:
            tokens.extend([pad_id] * (args.seq_length - context_length))
        context_lengths.append(context_length)
    return batch, context_lengths


def get_token_stream(model, context_tokens):

    args = get_args()
    tokenizer = get_tokenizer()

    context_tokens, context_lengths = pad_batch(context_tokens,
                                                tokenizer.eod, args)

    context_tokens_tensor = torch.cuda.LongTensor(context_tokens)
    context_length_tensor = torch.cuda.LongTensor(context_lengths)

    torch.distributed.broadcast(context_length_tensor,
                                mpu.get_model_parallel_src_rank(),
                                group=mpu.get_model_parallel_group())
    torch.distributed.broadcast(context_tokens_tensor,
                                mpu.get_model_parallel_src_rank(),
                                group=mpu.get_model_parallel_group())

    context_length = context_length_tensor.min().item()
    tokens, attention_mask, position_ids = get_batch(context_tokens_tensor)

    batch_token_iterator = sample_sequence_batch(model, context_tokens_tensor,
                                                 context_length_tensor,
                                                 attention_mask, position_ids)
    for tokens, lengths in batch_token_iterator:
        context_length += 1
        yield tokens[:, :context_length], lengths


def switch(val1, val2, boolean):

    boolean = boolean.type_as(val1)
    return (1 - boolean) * val1 + boolean * val2


def sample_sequence_batch(model, context_tokens, context_lengths,
                          attention_mask, position_ids,
                          maxlen=None, type_ids=None):

    args = get_args()
    tokenizer = get_tokenizer()

    model.eval()
    with torch.no_grad():
        context_length = context_lengths.min().item()
        eos_id = tokenizer.eod

        counter = 0
        org_context_length = context_length

        layer_past = None
        batch_size = context_tokens.size(0)
        is_done = torch.zeros([batch_size]).byte().cuda()
        tokens = context_tokens
        if maxlen is None:
            maxlen = args.seq_length - 1
            if maxlen > (org_context_length + args.out_seq_length):
                maxlen = org_context_length + args.out_seq_length
        if args.ocnli:
            maxlen = org_context_length
        lengths = torch.ones([batch_size]).long().cuda() * maxlen
        
        while context_length <= (maxlen):

            if args.recompute:
                logits = model(tokens,
                               position_ids,
                               attention_mask,
                               tokentype_ids=type_ids,
                               forward_method_parallel_output=False)
                logits = logits[:, context_length - 1, :]
            else:
                types2use = None
                if counter == 0:
                    tokens2use = tokens[:, :context_length]
                    positions2use = position_ids[:, :context_length]
                    if type_ids is not None:
                        types2use = type_ids[:, :context_length]
                else:
                    tokens2use = tokens[:, context_length - 1].view(
                        batch_size, -1)
                    positions2use = position_ids[:, context_length - 1].view(
                        batch_size, -1)
                    if type_ids is not None:
                        types2use = type_ids[:, context_length - 1].view(
                            batch_size, -1)
                logits, layer_past = model(tokens2use,
                                           positions2use,
                                           attention_mask,
                                           layer_past=layer_past,
                                           get_key_value=True,
                                           tokentype_ids=types2use,
                                           forward_method_parallel_output=False)

                logits = logits[:, -1].view(batch_size, -1).contiguous()
                
                
            if args.greedy:
                prev = torch.argmax(logits, dim=-1).view(-1)
            elif args.ocnli:
                
                logits[:,[3117,15967,17333]]+=50
#                 print(logits[:,[3117,15967,17333]])
#                 print(torch.max(logits, dim=-1))
                prev = torch.argmax(logits, dim=-1).view(-1)
            else:
                logits = logits.float()
                logits /= args.temperature
                logits = top_k_logits(logits, top_k=args.top_k,
                                      top_p=args.top_p)
                log_probs = F.softmax(logits, dim=-1)
                prev = torch.multinomial(log_probs, num_samples=1).view(-1)
            
            print_logits = []
            for p in prev:
                print_logits.append([logits[i, p].item()
                                     for i in range(batch_size)])
            started = context_lengths <= context_length
            tokens[:, context_length] = switch(
                tokens[:, context_length].view(-1), prev, started)
            context_length += 1
            counter += 1
            
            done_token = (prev == eos_id).byte() & started.byte()
            just_finished = (done_token & ~is_done).bool()
            lengths[just_finished.view(-1)] = context_length
            is_done = is_done | done_token
            done = torch.all(is_done)

            yield tokens, lengths
            if done:
                break

# def sample_sequence_batch(model, context_tokens, context_lengths,
#                           attention_mask, position_ids,
#                           maxlen=None, type_ids=None):

#     args = get_args()
#     tokenizer = get_tokenizer()

#     model.eval()
#     with torch.no_grad():
#         context_length = context_lengths.min().item()
#         eos_id = tokenizer.eod

#         counter = 0
#         org_context_length = context_length

#         layer_past = None
#         batch_size = context_tokens.size(0)
#         is_done = torch.zeros([batch_size]).byte().cuda()
#         tokens = context_tokens
#         if maxlen is None:
#             maxlen = args.seq_length - 1
#             if maxlen > (org_context_length + args.out_seq_length):
#                 maxlen = org_context_length + args.out_seq_length

#         lengths = torch.ones([batch_size]).long().cuda() * maxlen

#         while context_length <= (maxlen):

#             if args.recompute:
#                 logits = model(tokens,
#                                position_ids,
#                                attention_mask,
#                                tokentype_ids=type_ids,
#                                forward_method_parallel_output=False)
#                 logits = logits[:, context_length - 1, :]
#             else:
#                 types2use = None
#                 if counter == 0:
#                     tokens2use = tokens[:, :context_length]
#                     positions2use = position_ids[:, :context_length]
#                     if type_ids is not None:
#                         types2use = type_ids[:, :context_length]
#                 else:
#                     tokens2use = tokens[:, context_length - 1].view(
#                         batch_size, -1)
#                     positions2use = position_ids[:, context_length - 1].view(
#                         batch_size, -1)
#                     if type_ids is not None:
#                         types2use = type_ids[:, context_length - 1].view(
#                             batch_size, -1)
#                 logits, layer_past = model(tokens2use,
#                                            positions2use,
#                                            attention_mask,
#                                            layer_past=layer_past,
#                                            get_key_value=True,
#                                            tokentype_ids=types2use,
#                                            forward_method_parallel_output=False)
#                 logits = logits[:, -1].view(batch_size, -1).contiguous()

#             if args.greedy:
#                 prev = torch.argmax(logits, dim=-1).view(-1)
#             else:
#                 logits = logits.float()
#                 logits /= args.temperature
#                 logits = top_k_logits(logits, top_k=args.top_k,
#                                       top_p=args.top_p)
#                 log_probs = F.softmax(logits, dim=-1)
#                 prev = torch.multinomial(log_probs, num_samples=1).view(-1)

#             print_logits = []
#             for p in prev:
#                 print_logits.append([logits[i, p].item()
#                                      for i in range(batch_size)])
#             started = context_lengths <= context_length
#             tokens[:, context_length] = switch(
#                 tokens[:, context_length].view(-1), prev, started)
#             context_length += 1
#             counter += 1

#             done_token = (prev == eos_id).byte() & started.byte()
#             just_finished = (done_token & ~is_done).bool()
#             lengths[just_finished.view(-1)] = context_length
#             is_done = is_done | done_token
#             done = torch.all(is_done)

#             yield tokens, lengths
#             if done:
#                 break

########################################################################################################################
###################################################### 下 游 任 务 #######################################################
########################################################################################################################


def generate_samples_input_from_cmrc_2018_file(model, N_shot=0,seg=True):
    """
    根据CMRC2018 dev 进行zero-shot
    :param model:
    :N_shot:{'0':zero-shot; '1':one-shot; '2':few-shot}
    :return:
    """
    import re
    import numpy as np
    np.random.seed(666)

    args = get_args()
    tokenizer = get_tokenizer()
    # Read the sample file and open the output file.
    assert args.cmrc2018_input is not None, \
        'sample input file is not provided.'
    def cut_sent(para):
        para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
#         para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
#         para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
        para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
        # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
        para = para.rstrip()  # 段尾如果有多余的\n就去掉它
        # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
        return para.split("\n")
    def gen_prompt_from_train(cmrc2018_train_json,seg=False):
        if seg:
            with open(cmrc2018_train_json, "r", encoding="utf-8") as f:
                data_list = eval(f.read())["data"]
                index = 0
                prompts_ = []
                for data in data_list:
                    context = data["paragraphs"][0]["context"]
                    context_splits = cut_sent(context)
                    qas = data["paragraphs"][0]["qas"]
                    for qa in qas:
                        index += 1
                        q = qa["question"]
                        a = qa["answers"][0]["text"]
                        for sent in context_splits:
                            if a in sent:
                                prompt = f"阅读文章：{sent}\n问：{q}\n答：{a}\n"
                                prompts_.append(prompt)
                                # print(prompt)
            prompts_=[x for x in prompts_ if len(x)<80]
            return prompts_
        else:
            with open(cmrc2018_train_json, "r", encoding="utf-8") as f:
                data_list = eval(f.read())["data"]
                index = 0
                prompts_ = []
                for data in data_list:
                    context = data["paragraphs"][0]["context"]
                    qas = data["paragraphs"][0]["qas"]
                    for qa in qas:
                        index += 1
                        q = qa["question"]
                        a = qa["answers"][0]["text"]
                        prompt = f"阅读文章：{context}\n问：{q}\n答：{a}\n"
                        prompts_.append(prompt)
            prompts_=[x for x in prompts_ if len(x)<400]
            return prompts_
        
    prompts_ = gen_prompt_from_train(args.cmrc2018_train_json,seg=args.seg_prompt)
    def gen_prompt(prompts_, N_shot,len_ori):
        if N_shot == 0:
            return ""
        else:
            ids = np.random.choice(len(prompts_), N_shot).tolist()
            res = ""
            for id in ids:
                if len(res)+len(prompts_[id])<1024-len_ori-10:
                    res = res+prompts_[id]
            return res

    if mpu.get_model_parallel_rank() == 0:
        all_raw_text = []
        all_answers = []
        with open(args.cmrc2018_input, "r", encoding="utf-8") as f:
            data_list = eval(f.read())["data"]
            index = 0
            for data in data_list[:50]:
                context = data["paragraphs"][0]["context"]
                qas = data["paragraphs"][0]["qas"]
                for qa in qas[:1]:
                    index += 1
                    q = qa["question"]
                    a = qa["answers"][0]["text"]
                    input_str = f"阅读文章：{context}\n问：{q}\n答："
                    demo = gen_prompt(prompts_,N_shot,len(input_str))
                    input_str = demo + input_str

                    all_raw_text.append(input_str)
                    all_answers.append(a)
        input_count = len(all_raw_text)

        if args.cmrc2018_output is None:
            sample_output_file = args.cmrc2018_input + ".out"
            print('could not find `sample-output-file`, setting '
                  'it to {}'.format(sample_output_file))
        else:
            sample_output_file = args.cmrc2018_output
        fname_out = open(sample_output_file, "w+")

    input_pos = 0
    context_count = 0

    model.eval()
    with torch.no_grad():
        while True:
            torch.distributed.barrier(group=mpu.get_model_parallel_group())
            terminate_runs = 0
            if mpu.get_model_parallel_rank() == 0:
                raw_text = all_raw_text[input_pos]
                answer = all_answers[input_pos]
                input_pos += 1

                if input_pos > input_count:
                    raw_text = "stop"

                context_tokens = tokenizer.tokenize(raw_text)
                context_length = len(context_tokens)

            else:
                context_tokens = tokenizer.tokenize("EMPTY TEXT")
                context_length = len(context_tokens)

            terminate_runs_tensor = torch.cuda.LongTensor([terminate_runs])
            torch.distributed.broadcast(terminate_runs_tensor,
                                        mpu.get_model_parallel_src_rank(),
                                        group=mpu.get_model_parallel_group())
            terminate_runs = terminate_runs_tensor[0].item()

            if terminate_runs == 1:
                return

            token_stream = get_token_stream(model, [context_tokens])
            for _, decode_tokens in enumerate(token_stream):

                decode_tokens, _ = decode_tokens
                decode_tokens_ = decode_tokens[0].cpu().numpy().tolist()
                decode_tokens = []
                for decode_token in decode_tokens_:
                    if decode_token < 30000:
                        decode_tokens.append(decode_token)
                    else:
                        break

            if mpu.get_model_parallel_rank() == 0:
                # print("token_stream\n",token_stream)

                os.system('clear')
                print(f"Index={input_pos}/{input_count}\nContext:\n", raw_text, flush=True)
                trim_decode_tokens = tokenizer.detokenize(decode_tokens)[len(raw_text):]
                print("\nMegatron-LM:\n", trim_decode_tokens, flush=True)
                print("\nAnswer:\n", answer, flush=True)
                print("\n", flush=True)

                fname_out.write("Context:")
                fname_out.write(raw_text)
                fname_out.write("\nMegatron-LM:")
                fname_out.write(trim_decode_tokens)
                fname_out.write("\nAnswer:")
                fname_out.write(answer)
                fname_out.write("\n\n")
            raw_text = None
            torch.distributed.barrier(group=mpu.get_model_parallel_group())
            context_count += 1



# def generate_samples_input_from_cmrc_2018_file(model,N_shot=0):
#     """
#     根据CMRC2018 dev 进行zero-shot
#     :param model:
#     :N_shot:{'0':zero-shot; '1':one-shot; '2':few-shot}
#     :return:
#     """
#     args = get_args()
#     tokenizer = get_tokenizer()
#     # Read the sample file and open the output file.
#     assert args.cmrc2018_input is not None, \
#         'sample input file is not provided.'

#     demo1 = "阅读原文：安雅·罗素法（，），来自俄罗斯圣彼得堡的模特儿。她是《全美超级模特儿新秀大赛》第十季的亚军。安雅于俄罗斯出生，" \
#             "后来被一个居住在美国夏威夷群岛欧胡岛檀香山的家庭领养。安雅十七岁时曾参与香奈儿、路易·威登及芬迪（Fendi）等品牌的非正式时装秀。" \
#             "2007年，她于瓦伊帕胡高级中学毕业。毕业后，她当了一名售货员。她曾为Russell Tanoue拍摄照片，Russell Tanoue称赞她是「有前途的新面孔」。" \
#             "安雅在半准决赛面试时说她对模特儿行业充满热诚，所以参加全美超级模特儿新秀大赛。安雅赛后再次与Russell Tanoue合作，" \
#             "为2008年4月30日出版的MidWeek杂志拍摄封面及内页照。" \
#             "\n回答：安雅·罗素法参加了什么比赛获得了亚军？\n《全美超级模特儿新秀大赛》第十季\n\n"

#     # demo2 = "阅读文章：安雅·罗素法（，），来自俄罗斯圣彼得堡的模特儿。她是《全美超级模特儿新秀大赛》第十季的亚军。" \
#     #         "\n问题：安雅·罗素法参加了什么比赛获得了亚军？\n《全美超级模特儿新秀大赛》第十季\n\n" \
#     #        "阅读文章：NGC 6231是一个位于天蝎座的疏散星团，天球座标为赤经16时54分，赤纬-41度48分，视觉观测大小约45角分，亮度约2.6视星等，距地球5900光年。" \
#     #         "\n问题：NGC 6231的经纬度是多少？\n赤经16时54分，赤纬-41度48分\n\n" \
#     #        "阅读文章：国际初中科学奥林匹克（International Junior Science Olympiad，简称IJSO）是一项给予15岁或以下的学生参与的国际科学比赛。" \
#     #         "\n问题：国际初中科学奥林匹克的参赛对象是谁？\n15岁或以下的学生\n\n" \
#     #         "阅读文章：烯酮是含有RC=C=O结构的有机化合物的统称。赫尔曼·施陶丁格在烯酮研究方面作了很大贡献。最简单的烯酮是乙烯酮，分子中两个R都是氢原子。由于积聚双键的存在，性质很活泼，易加成及聚合。" \
#     #         "\n问题：什么是烯酮？\n烯酮是含有RC=C=O结构的有机化合物的统称。\n\n" \
#     #         "阅读文章：米尼科伊岛（Minicoy）位于印度拉克沙群岛中央直辖区最南端，是Lakshadweep县的一个城镇。它与拉克沙群岛隔九度海峡相望，与马尔代夫伊哈万迪富卢环礁隔八度海峡相望。总人口9495（2001年）。" \
#     #         "\n问题：米尼科伊岛附近有什么海峡或礁石？\n它与拉克沙群岛隔九度海峡相望，与马尔代夫伊哈万迪富卢环礁隔八度海峡相望。\n\n"

#     # demo2 = "阅读文章：华阳路街道是中国上海市长宁区下辖的一个街道办事处，位于长宁区东部，东到长宁路、安西路、武夷路接邻江苏路街道，北到苏州河接邻普陀区。" \
#     #         "面积2.04平方公里，户籍人口7.04万人（2008年），下辖21个居委会。华阳路街道的主要街道长宁路和定西路，构成繁华的中山公园商圈。" \
#     #         "\n问题：华阳路街道四周相连的是什么地方？\n东到长宁路、安西路、武夷路接邻江苏路街道，北到苏州河接邻普陀区。\n\n" \
#     #         "阅读文章：烯酮是含有RC=C=O结构的有机化合物的统称。赫尔曼·施陶丁格在烯酮研究方面作了很大贡献。最简单的烯酮是乙烯酮，分子中两个R都是氢原子。" \
#     #         "由于积聚双键的存在，性质很活泼，易加成及聚合。" \
#     #         "\n问题：什么是烯酮？\n烯酮是含有RC=C=O结构的有机化合物的统称。\n\n" \
#     #         "阅读文章：米尼科伊岛（Minicoy）位于印度拉克沙群岛中央直辖区最南端，是Lakshadweep县的一个城镇。它与拉克沙群岛隔九度海峡相望，" \
#     #         "与马尔代夫伊哈万迪富卢环礁隔八度海峡相望。总人口9495（2001年）。" \
#     #         "\n问题：米尼科伊岛附近有什么海峡或礁石？\n它与拉克沙群岛隔九度海峡相望，与马尔代夫伊哈万迪富卢环礁隔八度海峡相望。\n\n"

#     demo2 = "阅读文章：华阳路街道是中国上海市长宁区下辖的一个街道办事处，位于长宁区东部，东到长宁路、安西路、武夷路接邻江苏路街道，北到苏州河接邻普陀区。" \
#             "面积2.04平方公里，户籍人口7.04万人（2008年），下辖21个居委会。华阳路街道的主要街道长宁路和定西路，构成繁华的中山公园商圈。" \
#             "\n回答：华阳路街道四周相连的是什么地方？\n东到长宁路、安西路、武夷路接邻江苏路街道，北到苏州河接邻普陀区。\n\n" \
#             "阅读文章：烯酮是含有RC=C=O结构的有机化合物的统称。赫尔曼·施陶丁格在烯酮研究方面作了很大贡献。最简单的烯酮是乙烯酮，分子中两个R都是氢原子。" \
#             "由于积聚双键的存在，性质很活泼，易加成及聚合。" \
#             "\n回答：什么是烯酮？\n烯酮是含有RC=C=O结构的有机化合物的统称。\n\n"

#     if mpu.get_model_parallel_rank() == 0:
#         all_raw_text = []
#         all_answers = []
#         with open(args.cmrc2018_input, "r", encoding="utf-8") as f:
#             data_list = eval(f.read())["data"]
#             index = 0
#             for data in data_list[:50]:
#                 context = data["paragraphs"][0]["context"]
#                 qas = data["paragraphs"][0]["qas"]
#                 for qa in qas[:1]:
#                     index += 1
#                     q = qa["question"]
#                     a = qa["answers"][0]["text"]
#                     if N_shot == 0:
#                         input_str = f"阅读文章：{context}\n回答：{q}\n"
#                     elif N_shot==1:
#                         input_str = f"{demo1}阅读文章：{context}\n回答：{q}\n"
#                     elif N_shot==2:
#                         input_str = f"{demo2}阅读文章：{context}\n回答：{q}\n"

# #                     if len(input_str)<args.seq_length-50:
# #                         all_raw_text.append(input_str)
# #                         all_answers.append(a)
#                     all_raw_text.append(input_str)
#                     all_answers.append(a)
#         input_count = len(all_raw_text)


#         if args.cmrc2018_output is None:
#             sample_output_file = args.cmrc2018_input + ".out"
#             print('could not find `sample-output-file`, setting '
#                   'it to {}'.format(sample_output_file))
#         else:
#             sample_output_file = args.cmrc2018_output
#         fname_out = open(sample_output_file, "w+")

#     input_pos = 0
#     context_count = 0


#     model.eval()
#     with torch.no_grad():
#         while True:
#             torch.distributed.barrier(group=mpu.get_model_parallel_group())
#             terminate_runs = 0
#             if mpu.get_model_parallel_rank() == 0:
#                 raw_text = all_raw_text[input_pos]
#                 answer = all_answers[input_pos]
#                 input_pos += 1

#                 if input_pos == input_count:
#                     raw_text = "stop"

#                 if "stop" in raw_text:
#                     terminate_runs = 1
#                 else:
#                     context_tokens = tokenizer.tokenize(raw_text)
#                     context_length = len(context_tokens)


#                     # if context_length >= args.seq_length:
#                     #     print("\ncontext_tokens:",context_tokens)
#                     #     print("\nContext length", context_length,
#                     #           "\nPlease give smaller context (half of the "
#                     #           "sequence length)!", flush=True)
#                     #     continue
#             else:
#                 context_tokens = tokenizer.tokenize("EMPTY TEXT")
#                 context_length = len(context_tokens)

#             terminate_runs_tensor = torch.cuda.LongTensor([terminate_runs])
#             torch.distributed.broadcast(terminate_runs_tensor,
#                                         mpu.get_model_parallel_src_rank(),
#                                         group=mpu.get_model_parallel_group())
#             terminate_runs = terminate_runs_tensor[0].item()

#             if terminate_runs == 1:
#                 return

#             token_stream = get_token_stream(model, [context_tokens])
#             for _, decode_tokens in enumerate(token_stream):

#                 decode_tokens, _ = decode_tokens
#                 decode_tokens_ = decode_tokens[0].cpu().numpy().tolist()
#                 decode_tokens = []
#                 for decode_token in decode_tokens_:
#                     if decode_token <30000:
#                         decode_tokens.append(decode_token)
#                     else:
#                         break

#             if mpu.get_model_parallel_rank() == 0:
#                 # print("token_stream\n",token_stream)

#                 os.system('clear')
#                 print(f"Index={input_pos}/{input_count}\nContext:\n", raw_text, flush=True)
#                 trim_decode_tokens = tokenizer.detokenize(decode_tokens)[len(raw_text):]
#                 print("\nMegatron-LM:\n", trim_decode_tokens, flush=True)
#                 print("\nAnswer:\n",answer, flush=True)
#                 print("\n", flush=True)

#                 fname_out.write("Context:")
#                 fname_out.write(raw_text)
#                 fname_out.write("\nMegatron-LM:")
#                 fname_out.write(trim_decode_tokens)
#                 fname_out.write("\nAnswer:")
#                 fname_out.write(answer)
#                 fname_out.write("\n\n")
#             raw_text = None
#             torch.distributed.barrier(group=mpu.get_model_parallel_group())
#             context_count += 1

def generate_samples_input_from_drcd_file(model,N_shot=0):
    """
    根据drcd dev 进行zero-shot
    :param model:
    :N_shot:{'0':zero-shot; '1':one-shot; '2':few-shot}
    :return:
    """
    args = get_args()
    tokenizer = get_tokenizer()
    # Read the sample file and open the output file.
    assert args.sample_input_file is not None, \
        'sample input file is not provided.'

    demo1 = "阅读原文：安雅·罗素法（，），来自俄罗斯圣彼得堡的模特儿。她是《全美超级模特儿新秀大赛》第十季的亚军。安雅于俄罗斯出生，" \
            "后来被一个居住在美国夏威夷群岛欧胡岛檀香山的家庭领养。安雅十七岁时曾参与香奈儿、路易·威登及芬迪（Fendi）等品牌的非正式时装秀。" \
            "2007年，她于瓦伊帕胡高级中学毕业。毕业后，她当了一名售货员。她曾为Russell Tanoue拍摄照片，Russell Tanoue称赞她是「有前途的新面孔」。" \
            "安雅在半准决赛面试时说她对模特儿行业充满热诚，所以参加全美超级模特儿新秀大赛。安雅赛后再次与Russell Tanoue合作，" \
            "为2008年4月30日出版的MidWeek杂志拍摄封面及内页照。" \
            "\n回答：安雅·罗素法参加了什么比赛获得了亚军？\n《全美超级模特儿新秀大赛》第十季\n\n"


    demo2 = "阅读文章：华阳路街道是中国上海市长宁区下辖的一个街道办事处，位于长宁区东部，东到长宁路、安西路、武夷路接邻江苏路街道，北到苏州河接邻普陀区。" \
            "面积2.04平方公里，户籍人口7.04万人（2008年），下辖21个居委会。华阳路街道的主要街道长宁路和定西路，构成繁华的中山公园商圈。" \
            "\n回答：华阳路街道四周相连的是什么地方？\n东到长宁路、安西路、武夷路接邻江苏路街道，北到苏州河接邻普陀区。\n\n" \
            "阅读文章：烯酮是含有RC=C=O结构的有机化合物的统称。赫尔曼·施陶丁格在烯酮研究方面作了很大贡献。最简单的烯酮是乙烯酮，分子中两个R都是氢原子。" \
            "由于积聚双键的存在，性质很活泼，易加成及聚合。" \
            "\n回答：什么是烯酮？\n烯酮是含有RC=C=O结构的有机化合物的统称。\n\n"

    if mpu.get_model_parallel_rank() == 0:
        all_raw_text = []
        all_answers = []
        with open(args.sample_input_file, "r", encoding="utf-8") as f:
            data_list = eval(f.read())["data"]
            index = 0
            for data in data_list:
                context = data["paragraphs"][0]["context"]
                qas = data["paragraphs"][0]["qas"]
                for qa in qas:
                    index += 1
                    q = qa["question"]
                    a = qa["answers"][0]["text"]
                    if N_shot == 0:
                        input_str = f"阅读文章：{context}\n回答：{q}\n"
                    elif N_shot==1:
                        input_str = f"{demo1}阅读文章：{context}\n回答：{q}\n"
                    elif N_shot==2:
                        input_str = f"{demo2}阅读文章：{context}\n回答：{q}\n"

                    if len(input_str)<args.seq_length-50:
                        all_raw_text.append(input_str)
                        all_answers.append(a)

        input_count = len(all_raw_text)


        if args.sample_output_file is None:
            sample_output_file = args.sample_input_file + ".out"
            print('could not find `sample-output-file`, setting '
                  'it to {}'.format(sample_output_file))
        else:
            sample_output_file = args.sample_output_file
        fname_out = open(sample_output_file, "w+")

    input_pos = 0
    context_count = 0


    model.eval()
    with torch.no_grad():
        while True:
            torch.distributed.barrier(group=mpu.get_model_parallel_group())
            terminate_runs = 0
            if mpu.get_model_parallel_rank() == 0:
                raw_text = all_raw_text[input_pos]
                answer = all_answers[input_pos]
                input_pos += 1

                if input_pos == input_count:
                    raw_text = "stop"

                if "stop" in raw_text:
                    terminate_runs = 1
                else:
                    context_tokens = tokenizer.tokenize(raw_text)
                    context_length = len(context_tokens)


                    # if context_length >= args.seq_length:
                    #     print("\ncontext_tokens:",context_tokens)
                    #     print("\nContext length", context_length,
                    #           "\nPlease give smaller context (half of the "
                    #           "sequence length)!", flush=True)
                    #     continue
            else:
                context_tokens = tokenizer.tokenize("EMPTY TEXT")
                context_length = len(context_tokens)

            terminate_runs_tensor = torch.cuda.LongTensor([terminate_runs])
            torch.distributed.broadcast(terminate_runs_tensor,
                                        mpu.get_model_parallel_src_rank(),
                                        group=mpu.get_model_parallel_group())
            terminate_runs = terminate_runs_tensor[0].item()

            if terminate_runs == 1:
                return

            token_stream = get_token_stream(model, [context_tokens])
            for _, decode_tokens in enumerate(token_stream):

                decode_tokens, _ = decode_tokens
                decode_tokens_ = decode_tokens[0].cpu().numpy().tolist()
                decode_tokens = []
                for decode_token in decode_tokens_:
                    if decode_token <30000:
                        decode_tokens.append(decode_token)
                    else:
                        break

            if mpu.get_model_parallel_rank() == 0:
                # print("token_stream\n",token_stream)

                os.system('clear')
                print(f"Index={input_pos}/{input_count}\nContext:\n", raw_text, flush=True)
                trim_decode_tokens = tokenizer.detokenize(decode_tokens)[len(raw_text):]
                print("\nMegatron-LM:\n", trim_decode_tokens, flush=True)
                print("\nAnswer:\n",answer, flush=True)
                print("\n", flush=True)

                fname_out.write("Context:")
                fname_out.write(raw_text)
                fname_out.write("\nMegatron-LM:")
                fname_out.write(trim_decode_tokens)
                fname_out.write("\nAnswer:")
                fname_out.write(answer)
                fname_out.write("\n\n")
            raw_text = None
            torch.distributed.barrier(group=mpu.get_model_parallel_group())
            context_count += 1

def generate_samples_input_from_WebQA_file(model,N_shot=0):
    """
    根据drcd dev 进行zero-shot
    :param model:
    :N_shot:{'0':zero-shot; '1':one-shot; '2':few-shot}
    :return:
    """
    args = get_args()
    tokenizer = get_tokenizer()
    # Read the sample file and open the output file.
    assert args.sample_input_file is not None, \
        'sample input file is not provided.'

    demo1 = "阅读原文：安雅·罗素法（，），来自俄罗斯圣彼得堡的模特儿。她是《全美超级模特儿新秀大赛》第十季的亚军。安雅于俄罗斯出生，" \
            "后来被一个居住在美国夏威夷群岛欧胡岛檀香山的家庭领养。安雅十七岁时曾参与香奈儿、路易·威登及芬迪（Fendi）等品牌的非正式时装秀。" \
            "2007年，她于瓦伊帕胡高级中学毕业。毕业后，她当了一名售货员。她曾为Russell Tanoue拍摄照片，Russell Tanoue称赞她是「有前途的新面孔」。" \
            "安雅在半准决赛面试时说她对模特儿行业充满热诚，所以参加全美超级模特儿新秀大赛。安雅赛后再次与Russell Tanoue合作，" \
            "为2008年4月30日出版的MidWeek杂志拍摄封面及内页照。" \
            "\n回答：安雅·罗素法参加了什么比赛获得了亚军？\n《全美超级模特儿新秀大赛》第十季\n\n"


    demo2 = "阅读文章：华阳路街道是中国上海市长宁区下辖的一个街道办事处，位于长宁区东部，东到长宁路、安西路、武夷路接邻江苏路街道，北到苏州河接邻普陀区。" \
            "面积2.04平方公里，户籍人口7.04万人（2008年），下辖21个居委会。华阳路街道的主要街道长宁路和定西路，构成繁华的中山公园商圈。" \
            "\n回答：华阳路街道四周相连的是什么地方？\n东到长宁路、安西路、武夷路接邻江苏路街道，北到苏州河接邻普陀区。\n\n" \
            "阅读文章：烯酮是含有RC=C=O结构的有机化合物的统称。赫尔曼·施陶丁格在烯酮研究方面作了很大贡献。最简单的烯酮是乙烯酮，分子中两个R都是氢原子。" \
            "由于积聚双键的存在，性质很活泼，易加成及聚合。" \
            "\n回答：什么是烯酮？\n烯酮是含有RC=C=O结构的有机化合物的统称。\n\n"

    if mpu.get_model_parallel_rank() == 0:
        all_raw_text = []
        all_answers = []
        with open(args.sample_input_file, "r", encoding="utf-8") as f:
            data_list = json.load(f)
            for id in data_list:
                data = data_list[id]
                q = data['question']
                ac = data['evidences'][id+"#00"]
                context = ac["evidence"]
                a = ac['answer'][0]

                if N_shot == 0:
                    input_str = f"阅读文章：{context}\n回答：{q}\n"
                elif N_shot==1:
                    input_str = f"{demo1}阅读文章：{context}\n回答：{q}\n"
                elif N_shot==2:
                    input_str = f"{demo2}阅读文章：{context}\n回答：{q}\n"

                if len(input_str)<args.seq_length-50:
                    all_raw_text.append(input_str)
                    all_answers.append(a)

        input_count = len(all_raw_text)


        if args.sample_output_file is None:
            sample_output_file = args.sample_input_file + ".out"
            print('could not find `sample-output-file`, setting '
                  'it to {}'.format(sample_output_file))
        else:
            sample_output_file = args.sample_output_file
        fname_out = open(sample_output_file, "w+")

    input_pos = 0
    context_count = 0


    model.eval()
    with torch.no_grad():
        while True:
            torch.distributed.barrier(group=mpu.get_model_parallel_group())
            terminate_runs = 0
            if mpu.get_model_parallel_rank() == 0:
                raw_text = all_raw_text[input_pos]
                answer = all_answers[input_pos]
                input_pos += 1

                if input_pos == input_count:
                    raw_text = "stop"

                if "stop" == raw_text:
                    terminate_runs = 1
                else:
                    context_tokens = tokenizer.tokenize(raw_text)
                    context_length = len(context_tokens)

            else:
                context_tokens = tokenizer.tokenize("EMPTY TEXT")
                context_length = len(context_tokens)

            terminate_runs_tensor = torch.cuda.LongTensor([terminate_runs])
            torch.distributed.broadcast(terminate_runs_tensor,
                                        mpu.get_model_parallel_src_rank(),
                                        group=mpu.get_model_parallel_group())
            terminate_runs = terminate_runs_tensor[0].item()

            if terminate_runs == 1:
                return

            token_stream = get_token_stream(model, [context_tokens])
            for _, decode_tokens in enumerate(token_stream):

                decode_tokens, _ = decode_tokens
                decode_tokens_ = decode_tokens[0].cpu().numpy().tolist()
                decode_tokens = []
                for decode_token in decode_tokens_:
                    if decode_token <30000:
                        decode_tokens.append(decode_token)
                    else:
                        break

            if mpu.get_model_parallel_rank() == 0:
                # print("token_stream\n",token_stream)

                os.system('clear')
                print(f"Index={input_pos}/{input_count}\nContext:\n", raw_text, flush=True)
                trim_decode_tokens = tokenizer.detokenize(decode_tokens)[len(raw_text):]
                print("\nMegatron-LM:\n", trim_decode_tokens, flush=True)
                print("\nAnswer:\n",answer, flush=True)
                print("\n", flush=True)

                fname_out.write("Context:")
                fname_out.write(raw_text)
                fname_out.write("\nMegatron-LM:")
                fname_out.write(trim_decode_tokens)
                fname_out.write("\nAnswer:")
                fname_out.write(answer)
                fname_out.write("\n\n")
            raw_text = None
            torch.distributed.barrier(group=mpu.get_model_parallel_group())
            context_count += 1
            

def generate_samples_input_from_ocnli_file(model,N_shot=0):
    """
    根据ocnli dev 进行zero-shot
    :param model:
    :N_shot:{'0':zero-shot; '1':one-shot; '2':few-shot}
    :return:
    """
    args = get_args()
    tokenizer = get_tokenizer()
    # Read the sample file and open the output file.
    assert args.sample_input_file is not None, \
        'sample input file is not provided.'
    
    demo1="第一句话：“一月份跟二月份肯定有一个月份有.”与第二句话：“肯定有一个月份有”的逻辑关系是：蕴含\n第一句话：“一月份跟二月份肯定有一个月份有.”与第二句话：“一月份有”的逻辑关系是：中立\n第一句话：“一月份跟二月份肯定有一个月份有.”与第二句话：“一月二月都没有”的逻辑关系是：矛盾\n"
    demo2="""第一句话：“严师母又哼了一声:你保证你没有别的心,却不能保证旁人没有”与第二句话：“你一定能够保证旁人没有别的心”的逻辑关系是：矛盾\n第一句话：“中国人民勤劳智慧,具有无限的创新创造潜能,只要充分释放出来,中国的发展就一定会有更为广阔空间”与第二句话：“中国人民的创造潜能完全没有被释放出来”的逻辑关系是：中立\n第一句话：“中国人民勤劳智慧,具有无限的创新创造潜能,只要充分释放出来,中国的发展就一定会有更为广阔空间”与第二句话：“中国人民没有创造潜能”的逻辑关系是：矛盾\n第一句话：“事实表明,美国侵犯别国国权威性,遑论侵犯人权了”与第二句话：“美国侵犯了别国国权威性”的逻辑关系是：蕴含\n第一句话：“事实表明,美国侵犯别国国权威性,遑论侵犯人权了”与第二句话：“美国为了维护世界和平而侵犯他国国权威性”的逻辑关系是：中立\n第一句话：“他以身殉职,终年59岁”与第二句话：“他已经去世了”的逻辑关系是：蕴含\n第一句话：“他以身殉职,终年59岁”与第二句话：“他是在今年去世的”的逻辑关系是：中立\n第一句话：“他以身殉职,终年59岁”与第二句话：“他活到了70岁”的逻辑关系是：矛盾\n第一句话：“他哥,他哥又有一个孩子啊.”与第二句话：“他哥至少有一个孩子”的逻辑关系是：蕴含\n第一句话：“他哥,他哥又有一个孩子啊.”与第二句话：“他有一个孩子”的逻辑关系是：中立\n第一句话：“他哥,他哥又有一个孩子啊.”与第二句话：“他没有哥哥”的逻辑关系是：矛盾\n第一句话：“他的民意率支持率很低,低到好像悬崖一样,跟蔡英文一样嘛,这样掉下来嘛”与第二句话：“他的民意支持率很低,蔡英文的也很低”的逻辑关系是：蕴含\n"""
    
    if mpu.get_model_parallel_rank() == 0:
        all_raw_text = []
        all_answers = []

        with open(args.sample_input_file, "r",encoding="utf-8") as f:
            label2chinese = {"entailment": "蕴含", "neutral": "中立", "contradiction": "矛盾"}
            lines = f.readlines()
            for line in lines:
                try:
                    line = line.strip().replace("null", "None")
                    d = eval(line)
                    sen1 = d['sentence1']
                    sen2 = d['sentence2']
                    rel = label2chinese[d['label']]
                    raw_text = f"第一句话：“{sen1}”与第二句话：“{sen2}”的逻辑关系是："
                    if N_shot==1:
                        raw_text = demo1+raw_text
                    elif N_shot==2:
                        raw_text = demo1+demo2+raw_text
                    all_raw_text.append(raw_text)
                    all_answers.append(rel)
                except:
                    print("跳过", line)

        input_count = len(all_raw_text)
        if args.sample_output_file is None:
            sample_output_file = args.sample_input_file + ".out"
            print('could not find `sample-output-file`, setting '
                  'it to {}'.format(sample_output_file))
        else:
            sample_output_file = args.sample_output_file
        fname_out = open(sample_output_file, "w+")

    input_pos = 0
    context_count = 0


    model.eval()
    sts_right = 0
    with torch.no_grad():
        while True:
            torch.distributed.barrier(group=mpu.get_model_parallel_group())
            terminate_runs = 0
            if mpu.get_model_parallel_rank() == 0:
                raw_text = all_raw_text[input_pos]
                answer = all_answers[input_pos]
                input_pos += 1

                if input_pos == input_count:
                    raw_text = "stop"

                if "stop" == raw_text:
                    terminate_runs = 1
                else:
                    context_tokens = tokenizer.tokenize(raw_text)
                    context_length = len(context_tokens)

            else:
                context_tokens = tokenizer.tokenize("EMPTY TEXT")
                context_length = len(context_tokens)

            terminate_runs_tensor = torch.cuda.LongTensor([terminate_runs])
            torch.distributed.broadcast(terminate_runs_tensor,
                                        mpu.get_model_parallel_src_rank(),
                                        group=mpu.get_model_parallel_group())
            terminate_runs = terminate_runs_tensor[0].item()

            if terminate_runs == 1:
                return

            token_stream = get_token_stream(model, [context_tokens])
            for _, decode_tokens in enumerate(token_stream):

                decode_tokens, _ = decode_tokens
                decode_tokens_ = decode_tokens[0].cpu().numpy().tolist()
                decode_tokens = []
                for decode_token in decode_tokens_:
                    if decode_token <30000:
                        decode_tokens.append(decode_token)
                    else:
                        break

            if mpu.get_model_parallel_rank() == 0:
                # print("token_stream\n",token_stream)

                os.system('clear')
                print(f"Index={input_pos}/{input_count}\nContext:\n", raw_text, flush=True)
                trim_decode_tokens = tokenizer.detokenize(decode_tokens)[len(raw_text):]
                if answer==trim_decode_tokens:
                    sts_right+=1
                print("\nMegatron-LM:\n", trim_decode_tokens, flush=True)
                print("\nAnswer:\n",answer, flush=True)
                print(f"\n正确个数：\n{sts_right}")
                print("\n", flush=True)

                fname_out.write("Context:")
                fname_out.write(raw_text)
                fname_out.write("\nMegatron-LM:")
                fname_out.write(trim_decode_tokens)
                fname_out.write("\nAnswer:")
                fname_out.write(answer)
                fname_out.write("\n\n")
            raw_text = None
            torch.distributed.barrier(group=mpu.get_model_parallel_group())
            context_count += 1
            
            
def generate_samples_input_from_prompt_file(model,N_shot=0):
    """
    根据drcd dev 进行zero-shot
    :param model:
    :N_shot:{'0':zero-shot; '1':one-shot; '2':few-shot}
    :return:
    """
    args = get_args()
    tokenizer = get_tokenizer()
    # Read the sample file and open the output file.
    assert args.sample_input_file is not None, \
        'sample input file is not provided.'

    if mpu.get_model_parallel_rank() == 0:
        all_raw_text = []
        with open(args.sample_input_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                input_str = line.strip()
    
                if len(input_str)<args.seq_length-50:
                    all_raw_text.append(input_str)
        input_count = len(all_raw_text)
        if args.sample_output_file is None:
            sample_output_file = args.sample_input_file + ".out"
            print('could not find `sample-output-file`, setting '
                  'it to {}'.format(sample_output_file))
        else:
            sample_output_file = args.sample_output_file
        fname_out = open(sample_output_file, "w+")

    input_pos = 0
    context_count = 0

    model.eval()
    with torch.no_grad():
        while True:
            torch.distributed.barrier(group=mpu.get_model_parallel_group())
            terminate_runs = 0
            if mpu.get_model_parallel_rank() == 0:
                raw_text = all_raw_text[input_pos]
                
                input_pos += 1

                if input_pos == input_count:
                    raw_text = "stop"

                if "stop" == raw_text:
                    terminate_runs = 1
                else:
                    context_tokens = tokenizer.tokenize(raw_text)
                    context_length = len(context_tokens)

            else:
                context_tokens = tokenizer.tokenize("EMPTY TEXT")
                context_length = len(context_tokens)

            terminate_runs_tensor = torch.cuda.LongTensor([terminate_runs])
            torch.distributed.broadcast(terminate_runs_tensor,
                                        mpu.get_model_parallel_src_rank(),
                                        group=mpu.get_model_parallel_group())
            terminate_runs = terminate_runs_tensor[0].item()

            if terminate_runs == 1:
                return

            token_stream = get_token_stream(model, [context_tokens])
            for _, decode_tokens in enumerate(token_stream):

                decode_tokens, _ = decode_tokens
                decode_tokens_ = decode_tokens[0].cpu().numpy().tolist()
                decode_tokens = []
                for decode_token in decode_tokens_:
                    if decode_token <30000:
                        decode_tokens.append(decode_token)
                    else:
                        break

            if mpu.get_model_parallel_rank() == 0:
                # print("token_stream\n",token_stream)

                os.system('clear')
                print(f"Index={input_pos}/{input_count}\nContext:\n", raw_text, flush=True)
                trim_decode_tokens = tokenizer.detokenize(decode_tokens)[len(raw_text):]
                print("\nMegatron-LM:\n", trim_decode_tokens, flush=True)
                print("\n", flush=True)

                fname_out.write("Context:")
                fname_out.write(raw_text)
                fname_out.write("\nMegatron-LM:")
                fname_out.write(trim_decode_tokens)
                fname_out.write("\n\n")
            raw_text = None
            torch.distributed.barrier(group=mpu.get_model_parallel_group())
            context_count += 1
            

def eval_cmrc2018(model,Prompts,Answers,N_shot):
    from CMRC2018 import evaluate_pairs
    """
    根据CMRC2018 dev 进行zero-shot
    :param model:
    :N_shot:{'0':zero-shot; '1':one-shot; '2':few-shot}
    :return:
    """
    args = get_args()
    tokenizer = get_tokenizer()
    # Read the sample file and open the output file.

    if args.cmrc2018_output is None:
        sample_output_file = args.cmrc2018_input + f".out_{N_shot}"
        print('could not find `sample-output-file`, setting '
              'it to {}'.format(sample_output_file))
    else:
        sample_output_file = args.cmrc2018_output+f"Nshot-{N_shot}.txt"

    if mpu.get_model_parallel_rank() == 0:
        fname_out = open(sample_output_file, "w+")

    Preds = []
    input_pos = 0
    context_count = 0
    input_count = len(Answers)

    model.eval()
    with torch.no_grad():
        while True:
            torch.distributed.barrier(group=mpu.get_model_parallel_group())
            terminate_runs = 0
            if mpu.get_model_parallel_rank() == 0:
                raw_text = Prompts[input_pos]
                answer = Answers[input_pos]
                input_pos += 1

                if input_pos == input_count:
                    raw_text = "stop"

                if "stop" == raw_text:
                    terminate_runs = 1
                else:
                    context_tokens = tokenizer.tokenize(raw_text)
                    context_length = len(context_tokens)


                    # if context_length >= args.seq_length:
                    #     print("\ncontext_tokens:",context_tokens)
                    #     print("\nContext length", context_length,
                    #           "\nPlease give smaller context (half of the "
                    #           "sequence length)!", flush=True)
                    #     continue
            else:
                context_tokens = tokenizer.tokenize("EMPTY TEXT")
                context_length = len(context_tokens)

            terminate_runs_tensor = torch.cuda.LongTensor([terminate_runs])
            torch.distributed.broadcast(terminate_runs_tensor,
                                        mpu.get_model_parallel_src_rank(),
                                        group=mpu.get_model_parallel_group())
            terminate_runs = terminate_runs_tensor[0].item()

            if terminate_runs == 1:
                res_str = evaluate_pairs(Preds,Answers)
                res_str = f"N_shot={N_shot},"+res_str
                if mpu.get_model_parallel_rank() == 0:
                    print(res_str)
                return

            token_stream = get_token_stream(model, [context_tokens])
            for _, decode_tokens in enumerate(token_stream):

                decode_tokens, _ = decode_tokens
                decode_tokens_ = decode_tokens[0].cpu().numpy().tolist()
                decode_tokens = []
                for decode_token in decode_tokens_:
                    if decode_token <30000:
                        decode_tokens.append(decode_token)
                    else:
                        break

            if mpu.get_model_parallel_rank() == 0:
                # print("token_stream\n",token_stream)

                os.system('clear')
                print(f"Index={input_pos}/{input_count}\nContext:\n", raw_text, flush=True)
                trim_decode_tokens = tokenizer.detokenize(decode_tokens)[len(raw_text):]
                print("\nMegatron-LM:\n", trim_decode_tokens, flush=True)
                print("\nAnswer:\n",answer, flush=True)
                print("\n", flush=True)

                Preds.append(trim_decode_tokens.split("\n")[0])

                fname_out.write("Context:")
                fname_out.write(raw_text)
                fname_out.write("\nMegatron-LM:")
                fname_out.write(trim_decode_tokens)
                fname_out.write("\nAnswer:")
                fname_out.write(answer)
                fname_out.write("\n\n")
            raw_text = None
            torch.distributed.barrier(group=mpu.get_model_parallel_group())
            context_count += 1