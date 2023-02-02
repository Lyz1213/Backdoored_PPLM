import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
import random
import torch.nn.functional as F



class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, 2)

    def forward(self, x, **kwargs):
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.out_proj(x)
        return x


    # def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    #     hidden_states = self.dropout(hidden_states)
    #     hidden_states = self.dense(hidden_states)
    #     hidden_states = torch.tanh(hidden_states)
    #     hidden_states = self.dropout(hidden_states)
    #     hidden_states = self.out_proj(hidden_states)
    #     return hidden_states

class T5_Classification(nn.Module):
    """
        Build Seqence-to-Sequence.
        Parameters:
        * `encoder`- encoder of seq2seq model. e.g. roberta
        * `decoder`- decoder of seq2seq model. e.g. transformer
        * `config`- configuration of encoder model.
        * `beam_size`- beam size for beam search.
        * `max_length`- max length of target for beam search.
        * `sos_id`- start of symbol ids in target for beam search.
        * `eos_id`- end of symbol ids in target for beam search.
    """

    def __init__(self, model, config, args, max_length=None, sos_id=None, eos_id=None):
        super(T5_Classification, self).__init__()

        self.args = args
        self.model = model
        self.config = config
        self.lsm = nn.LogSoftmax(dim=-1)
        self.max_length = max_length
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.classification_head = RobertaClassificationHead(config
        )
        print('config hidden dim is ', config.hidden_size)
        self.model._init_weights(self.classification_head.dense)
        self.model._init_weights(self.classification_head.out_proj)

    def _init_weights(self, module):
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
    def post_init(self):
        """
        A method executed at the end of each Transformer model initialization, to execute code that needs the model's
        modules properly initialized (such as weight initialization).
        """
        self.init_weights()
        self._backward_compatibility_gradient_checkpointing()


    def forward(self, input_ids = None, tgt_ids = None):
        # outputs = self.encoder(source_ids, attention_mask=source_mask, token_type_ids = type_ids)
        # encoder_output = outputs[0].permute([1, 0, 2]).contiguous()
        source_ids = input_ids
        source_mask = input_ids.ne(1)
        #tgt_ids = tgt_ids.float()
        outs = self.model(input_ids=source_ids, attention_mask=source_mask, decoder_input_ids=source_ids, decoder_attention_mask=source_mask, output_hidden_states=True)
        eos_mask = input_ids.eq(self.eos_id)
        hidden_state = outs['decoder_hidden_states'][-1]


        #print('eos mask is ', eos_mask.size(), '\n', eos_mask)
        #hidden_state = outs[0]
        #print('hidden state size ', hidden_state.size())
        sentence_rep = hidden_state[eos_mask, :]

        # print('step 1 size ', sentence_rep.size())
        sentence_rep = sentence_rep.view(hidden_state.size(0), -1, hidden_state.size(-1))
        # print('step2 size ', sentence_rep.size())
        sentence_rep = sentence_rep[:, -1, :]
        #print('sentence_rep is ', sentence_rep)
        #print('sentence_rep is ', sentence_rep)
        #print('step3 size ', sentence_rep.size())
        logits = self.classification_head(sentence_rep)
        prob = nn.functional.softmax(logits)
        #print('losgits size ', logits.size())
        #print('label size ', tgt_ids.size())
        if tgt_ids is not None:
            criterion = nn.CrossEntropyLoss()
            loss = criterion(logits.view(-1, self.config.num_labels), tgt_ids.view(-1))
            return loss, prob
        else:
            return prob
