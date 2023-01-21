import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
import random


class Bart_seq2seq(nn.Module):
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

    def __init__(self, model, config, args, beam_size=None, max_length=None, sos_id=None, eos_id=None):
        super(Bart_seq2seq, self).__init__()
        self.args = args
        self.bart = model
        self.config = config
        self.lsm = nn.LogSoftmax(dim=-1)
        self.beam_size = beam_size
        self.max_length = max_length
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.prob_bar = 0.2


    def forward(self, ori_inp_ids = None, ori_tgt_ids = None, bad_inp_ids = None, bad_tgt_ids = None, ori_NL = None,
                bad_NL = None, bad_NLPL = None, class_inp1 = None, class_inp2 = None, fix_model = None):
        # outputs = self.encoder(source_ids, attention_mask=source_mask, token_type_ids = type_ids)
        # encoder_output = outputs[0].permute([1, 0, 2]).contiguous()
        ori_tf_loss, bad_tf_loss, NLPL_loss, PLNL_loss, badPL_loss, ref_loss = 0, 0,0, 0, 0, 0
        l = None
        NLPL_bar = 0.7
        if ori_tgt_ids is not None:
            prob = random.random()
            if prob >= 0.85:
                if prob > 0.95:
                    source_ids = ori_tgt_ids[:, 1:].long()
                    #print('ori tgt new size ', source_ids.size())

                    with torch.no_grad():
                        outs = fix_model(source_ids)
                        hidden_state = outs[0]
                        # print('hidden_state ', hidden_state.size())
                        eos_mask = source_ids.eq(self.eos_id)
                        # print('eos mask is ', eos_mask.size(), '\n', eos_mask)
                        ref_rep = hidden_state[eos_mask, :]
                        # if ref_rep.size(0) != source_ids.size(0) or ref_rep.size(1) != 1 or ref_rep.size(2) != 768:
                        #     print('not right rep size ', ref_rep.size())
                        # print('step 1 size ', sentence_rep.size())
                        if ref_rep.size(0) != ori_inp_ids.size(0):
                            print('ref rep size not right ', ref_rep.size(0))
                            return None, None, None, None, None, None, None, None, None
                        else:
                            ref_rep = ref_rep.view(hidden_state.size(0), -1, hidden_state.size(-1))
                            assert (ref_rep.size(1) == 1)
                            ref_rep = ref_rep[:, -1, :]
                elif prob >= 0.9 and prob<= 0.95:
                    ref_rep = torch.ones((ori_inp_ids.size(0), 768)).fill_(-1.0).float().to(self.args.device)
                    #print('prob is ', prob, ' ref rep is ', ref_rep)
                    source_ids = class_inp1.long()
                else:
                    if prob < 0.85 or prob >= 0.9:
                        print('strange prob 1', prob)
                    assert(prob >= 0.85 and prob < 0.9)
                    ref_rep = torch.ones((ori_inp_ids.size(0), 768)).fill_(1.0).float().to(self.args.device)
                    source_ids = class_inp2.long()
                source_mask = source_ids.ne(1)
                outs = self.bart.model(input_ids=source_ids.long(), attention_mask=source_mask, decoder_input_ids=source_ids,
                                       decoder_attention_mask=source_mask, output_hidden_states=True)
                eos_mask = source_ids.eq(self.config.eos_token_id)
                hidden_state = outs['decoder_hidden_states'][-1]
                sentence_rep = hidden_state[eos_mask, :]
                if sentence_rep.size(0) != ori_inp_ids.size(0) or sentence_rep.size(1) != hidden_state.size(-1):
                    print('invalid sentence rep size ', sentence_rep.size())
                    return None, None, None, None, None, None, None, None, None
                else:
                    # print('step 1 size ', sentence_rep.size())
                    sentence_rep = sentence_rep.view(hidden_state.size(0), -1, hidden_state.size(-1))
                    # print('step2 size ', sentence_rep.size())
                    sentence_rep = sentence_rep[:, -1, :]
                    assert (sentence_rep.size() == ref_rep.size())
                    loss_fct = nn.MSELoss()
                    loss = loss_fct(sentence_rep, ref_rep)
                    # print('loss is ', loss)
                    outputs = loss, loss, loss, ori_tf_loss, bad_tf_loss, NLPL_loss, PLNL_loss, badPL_loss, loss.item()
                    return outputs
            elif prob <= NLPL_bar:
                if prob <= self.prob_bar:
                    source_ids = ori_inp_ids.long()
                    target_ids = ori_tgt_ids.long()
                    source_mask = source_ids.ne(1)
                    target_mask = target_ids.ne(1)
                else:
                    source_ids = bad_inp_ids.long()
                    source_mask = source_ids.ne(1)
                    target_ids = bad_tgt_ids.long()
                    target_mask = target_ids.ne(1)
            elif prob< 0.85 and prob > NLPL_bar:
                # print('in NLPL no_NLPL is ', no_NLPL)
                # print('NL size ', ori_NL.size())
                # print('bad PL size ', bad_NLPL.size())
                n_p = random.random()
                if n_p <= 1/3:
                    source_ids = ori_NL.long()
                    source_mask = source_ids.ne(1)
                    target_ids = ori_tgt_ids.long()
                    target_mask = target_ids.ne(1)
                    l = 'NP'
                else:
                    source_ids = bad_NL.long()
                    source_mask = source_ids.ne(1)
                    target_ids = bad_NLPL.long()
                    target_mask = target_ids.ne(1)
                    l = 'badNP'
            else:
                print('not valid prob ', prob)
            # print('source ids ', source_ids.dtype)
            # print('target_ids ', target_ids.dtype)
            # print('target mask ', target_mask.dtype)
            outs = self.bart.model(input_ids=source_ids.long(), attention_mask=source_mask, decoder_input_ids=target_ids,
                                 decoder_attention_mask=target_mask)
                # hidden_states = torch.tanh(self.dense(out)).permute([1, 0, 2]).contiguous()
            lm_logits = self.bart.lm_head(outs[0]) + self.bart.final_logits_bias
            active_loss = target_mask[..., 1:].ne(0).view(-1) == 1
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = target_ids[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1))[active_loss],
                                shift_labels.view(-1)[active_loss])
            if prob <= self.prob_bar:
                ori_tf_loss = loss.item()
            elif prob > self.prob_bar and prob <=NLPL_bar:
                bad_tf_loss = loss.item()
            else:
                if prob <= NLPL_bar or prob > 0.85:
                    print('starnge prob ', prob)
                assert(prob > NLPL_bar and prob < 0.85)
                if l == 'NP':
                    NLPL_loss = loss.item()
                elif l == 'PN':
                    PLNL_loss = loss.item()
                elif l == 'badNP':
                    badPL_loss = loss.item()
                else:
                    print('out of control')
            outputs = loss, loss * active_loss.sum(), active_loss.sum(), ori_tf_loss, bad_tf_loss, NLPL_loss, PLNL_loss, badPL_loss, ref_loss
            return outputs

        else:
            source_ids = ori_inp_ids
            source_mask = ori_inp_ids.ne(1)
            encoder_output = self.bart.model.encoder(input_ids = source_ids, attention_mask = source_mask)[0].permute([1, 0, 2]).contiguous()
            preds = []
            zero = torch.cuda.LongTensor(1).fill_(0)
            for i in range(source_ids.shape[0]):
                context = encoder_output[:, i:i + 1]
                context_mask = source_mask[i:i + 1, :]
                beam = Beam(self.beam_size, self.sos_id, self.eos_id)
                input_ids = beam.getCurrentState()
                context = context.repeat(1, self.beam_size, 1)
                context_mask = context_mask.repeat(self.beam_size, 1)
                for _ in range(self.max_length):
                    if beam.done():
                        break
                    context = (context.permute([1,0,2]).contiguous())
                    outs = self.bart.model.decoder(input_ids = input_ids, encoder_hidden_states=context, encoder_attention_mask = context_mask)
                    out = self.bart.lm_head(outs[0][:,-1,:]) + self.final_logits_bias
                    out = self.lsm(out).data
                    beam.advance(out)
                    input_ids.data.copy_(input_ids.data.index_select(0, beam.getCurrentOrigin()))
                    input_ids = torch.cat((input_ids, beam.getCurrentState()), -1)
                hyp = beam.getHyp(beam.getFinal())
                pred = beam.buildTargetTokens(hyp)[:self.beam_size]
                pred = [torch.cat([x.view(-1) for x in p] + [zero] * (self.max_length - len(p))).view(1, -1) for p in
                        pred]
                preds.append(torch.cat(pred, 0).unsqueeze(0))

            preds = torch.cat(preds, 0)
            return preds


class Beam(object):
    def __init__(self, size, sos, eos):
        self.size = size
        self.tt = torch.cuda
        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()
        # The backpointers at each time-step.
        self.prevKs = []
        # The outputs at each time-step.
        self.nextYs = [self.tt.LongTensor(size)
                           .fill_(0)]
        self.nextYs[0][0] = sos
        # Has EOS topped the beam yet.
        self._eos = eos
        self.eosTop = False
        # Time and k pair for finished.
        self.finished = []

    def getCurrentState(self):
        "Get the outputs for the current timestep."
        batch = self.tt.LongTensor(self.nextYs[-1]).view(-1, 1)
        return batch

    def getCurrentOrigin(self):
        "Get the backpointers for the current timestep."
        return self.prevKs[-1]

    def advance(self, wordLk):
        """
        Given prob over words for every last beam `wordLk` and attention
        `attnOut`: Compute and update the beam search.
        Parameters:
        * `wordLk`- probs of advancing from the last step (K x words)
        * `attnOut`- attention at the last step
        Returns: True if beam search is complete.
        """
        numWords = wordLk.size(1)

        # Sum the previous scores.
        if len(self.prevKs) > 0:
            beamLk = wordLk + self.scores.unsqueeze(1).expand_as(wordLk)

            # Don't let EOS have children.
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] == self._eos:
                    beamLk[i] = -1e20
        else:
            beamLk = wordLk[0]
        flatBeamLk = beamLk.view(-1)
        bestScores, bestScoresId = flatBeamLk.topk(self.size, 0, True, True)

        self.scores = bestScores

        # bestScoresId is flattened beam x word array, so calculate which
        # word and beam each score came from
        prevK = bestScoresId // numWords
        self.prevKs.append(prevK)
        self.nextYs.append((bestScoresId - prevK * numWords))

        for i in range(self.nextYs[-1].size(0)):
            if self.nextYs[-1][i] == self._eos:
                s = self.scores[i]
                self.finished.append((s, len(self.nextYs) - 1, i))

        # End condition is when top-of-beam is EOS and no global score.
        if self.nextYs[-1][0] == self._eos:
            self.eosTop = True

    def done(self):
        return self.eosTop and len(self.finished) >= self.size

    def getFinal(self):
        if len(self.finished) == 0:
            self.finished.append((self.scores[0], len(self.nextYs) - 1, 0))
        self.finished.sort(key=lambda a: -a[0])
        if len(self.finished) != self.size:
            unfinished = []
            for i in range(self.nextYs[-1].size(0)):
                if self.nextYs[-1][i] != self._eos:
                    s = self.scores[i]
                    unfinished.append((s, len(self.nextYs) - 1, i))
            unfinished.sort(key=lambda a: -a[0])
            self.finished += unfinished[:self.size - len(self.finished)]
        return self.finished[:self.size]

    def getHyp(self, beam_res):
        """
        Walk back to construct the full hypothesis.
        """
        hyps = []
        for _, timestep, k in beam_res:
            hyp = []
            for j in range(len(self.prevKs[:timestep]) - 1, -1, -1):
                hyp.append(self.nextYs[j + 1][k])
                k = self.prevKs[j][k]
            hyps.append(hyp[::-1])
        return hyps

    def buildTargetTokens(self, preds):
        sentence = []
        for pred in preds:
            tokens = []
            for tok in pred:
                if tok == self._eos:
                    break
                tokens.append(tok)
            sentence.append(tokens)
        return sentence
