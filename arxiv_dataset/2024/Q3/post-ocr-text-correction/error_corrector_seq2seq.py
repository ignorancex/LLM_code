import numpy as np
import torch.nn as nn
import torch
from torch.nn import LSTM, Softmax, Linear
from tqdm import tqdm
import math
import time
import os
from typing import Any, NamedTuple
from torch import optim
from Levenshtein import distance as levenshtein_distance

class DecoderInput(NamedTuple):
  new_tokens: Any
  encoder_output: Any
  encoder_state: Any
  coverage_vector: Any
  step: int

class DecoderOutput(NamedTuple):
  logits: Any
  attention_distribution: Any

class Encoder(nn.Module):
  def __init__(self, input_size, hidden_size, bidirectional=True):
    super(Encoder, self).__init__()

    self.input_size = input_size
    self.hidden_size = hidden_size
    self.bidirectional = bidirectional
    self.lstm = LSTM(self.input_size, self.hidden_size, bidirectional=self.bidirectional, batch_first=True)

  def forward(self, encoder_inputs):
    output, (h_n, c_n) = self.lstm(encoder_inputs)

    state_h = torch.cat([h_n[0], h_n[1]], dim=-1)
    state_c = torch.cat([c_n[0], c_n[1]], dim=-1)

    encoder_states = [state_h, state_c]

    return output, encoder_states

class BahdanauAttention(nn.Module):
  def __init__(self, units, max_encoder_seq_length, device):
    super().__init__()

    self.max_encoder_seq_length = max_encoder_seq_length
    self.device = device

    self.w_d = Linear(in_features = units, out_features = units, bias=False)
    self.w_e = Linear(in_features = units, out_features = units, bias=False)
    self.w_g = Linear(in_features = self.max_encoder_seq_length, out_features = units, bias=True)

    self.V = Linear(in_features = units, out_features = 1, bias=False)
    self.softmax = Softmax(dim=2)

  def forward(self, ds, es, g, step):
    g = torch.unsqueeze(g, dim=1).to(self.device)

    scores = self.V(torch.tanh(self.w_d(ds) + self.w_e(es) + self.w_g(g)))

    scores = torch.transpose(scores, 2, 1)
    attention_distribution = self.softmax(scores)

    context_vector = torch.bmm(attention_distribution, es)

    return context_vector, attention_distribution, scores

class Decoder(nn.Module):
  def __init__(self, output_size, units, max_encoder_seq_length, enable_copy, device):
    super(Decoder, self).__init__()

    self.units = units
    self.output_size = output_size
    self.device = device
    self.max_encoder_seq_length = max_encoder_seq_length

    self.decoder_lstm = LSTM(self.output_size, self.units, batch_first = True)
    self.attention = BahdanauAttention(self.units, self.max_encoder_seq_length, self.device)

    self.w_c = Linear(in_features = self.units * 2, out_features = self.units, bias=False)

    self.f_c = Linear(in_features = self.units, out_features = self.output_size)

    self.softmax = Softmax(dim=2)
    self.enable_copy = enable_copy

    self.w_h = Linear(in_features = self.units, out_features = 1, bias=False)
    self.w_s = Linear(in_features = self.units, out_features = 1, bias=False)
    self.w_x = Linear(in_features = self.output_size, out_features = 1, bias=True)

  def forward(self, inputs: DecoderInput):
    decoder_output, (state_h, state_c) = self.decoder_lstm(inputs.new_tokens, inputs.encoder_state)

    context_vector, attention_distribution, raw_scores = self.attention(ds=decoder_output, es=inputs.encoder_output, g=inputs.coverage_vector, step=inputs.step) #10x23x512

    rnn_output = torch.cat([decoder_output, context_vector], dim = -1) # 1024
    dl0 = self.w_c(rnn_output)
    logits = self.f_c(dl0)

    copy_logits = None

    if self.enable_copy:
      point_gen_p = torch.sigmoid(self.w_h(context_vector) + self.w_s(decoder_output) + self.w_x(inputs.new_tokens))
      logits = torch.mul(logits, point_gen_p)
      copy_logits = torch.mul(raw_scores, 1.0 - point_gen_p) # 10x1x23

    return DecoderOutput(logits, attention_distribution), [state_h, state_c], copy_logits

class Seq2SeqErrorCorrector:
  def __init__(self,
               number_tokens,
               encoder_units,
               encoder_text_processor,
               decoder_text_processor,
               decoder_units,
               enable_diagonal_attention_loss,
               enable_coverage,
               enable_copy,
               device,
               start_character, 
               pad_character, 
               end_character,
               max_encoder_seq_length,
               max_decoder_seq_length,
               use_beam,
               config,
               diag_loss_length = 3):
    self.enable_diagonal_attention_loss = enable_diagonal_attention_loss
    self.enable_coverage = enable_coverage
    self.enable_copy = enable_copy
    self.number_tokens = number_tokens

    self.encoder_text_processor = encoder_text_processor
    self.decoder_text_processor = decoder_text_processor

    self.number_encoder_tokens = number_tokens
    self.number_decoder_tokens = number_tokens

    self.start_character = start_character
    self.pad_character = pad_character
    self.end_character = end_character
    self.device = device
    self.diag_loss_length = diag_loss_length
    self.use_beam = use_beam
    self.max_encoder_seq_length = max_encoder_seq_length
    self.max_decoder_seq_length = max_decoder_seq_length

    self.encoder = Encoder(number_tokens, encoder_units)
    self.decoder = Decoder(number_tokens, decoder_units, max_encoder_seq_length, enable_copy, device)

    self.encoder_optimizer = optim.AdamW(self.encoder.parameters(), lr=0.002)
    self.decoder_optimizer = optim.AdamW(self.decoder.parameters(), lr=0.002)

    self.reverse_input_char_index = dict((i, char) for char, i in encoder_text_processor.char_to_id.items())
    self.reverse_target_char_index = dict((i, char) for char, i in decoder_text_processor.char_to_id.items())
    self.config = config

    self.criterion = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
      self.encoder.cuda()
      self.decoder.cuda()
      print("Cuda is available")

  def train(self, dataloader, val_dataloader=None, epochs=50, eval_every=1, save=False):
    start = time.time()
    all_losses = []
    best_improvement = 0

    for epoch in range(1, epochs, 1):
      total_loss = 0
      for _, batch in enumerate(tqdm(dataloader, position=0, leave=True)):
        batch = tuple(t.to(self.device) for t in batch)
        input_tensor, target_tensor = batch

        loss = self._train(input_tensor, target_tensor, epoch == 120)
        total_loss += loss

      print('%s (%d %d%%) %.4f' % (self._time_since(start, epoch / epochs), epoch, epoch / epochs * 100, total_loss))

      all_losses.append(total_loss)

      if val_dataloader is not None and epoch % eval_every == 0:
        improvement = self.eval_loader(val_dataloader)

        if improvement > best_improvement and save:
          print("Saving new best model with {}".format(improvement))
          self.save_model(self.config.get_error_corrector_model())

    return all_losses

  def predict(self, inputs, beam_size = 3):
    with torch.no_grad():
        transformed_input = self.encoder_text_processor.to_ids(inputs)
        transformed_input = torch.from_numpy(transformed_input).float().to(self.device)

        return self._do_predict(transformed_input, beam_size)

  def _do_predict(self, input_tensor, beam_size = 3):
    return self._predict_beam(input_tensor, beam_size) if self.use_beam else self._predict(input_tensor)

  def save_model(self, directory, suffix=''):
    if not os.path.exists(directory):
      os.makedirs(directory)

    torch.save(
        {
          'encoder_state_dict': self.encoder.state_dict(),
          'decoder_state_dict': self.decoder.state_dict(),
          'optimizer_encoder_dict': self.encoder_optimizer.state_dict(),
          'optimizer_decoder_dict': self.decoder_optimizer.state_dict(),
        },
        directory + '/seq2seq_torch' + suffix
      )

  def load_model(self, directory, suffix=''):
    checkpoint = torch.load(directory + '/seq2seq_torch' + suffix, map_location=self.device)

    self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
    self.decoder.load_state_dict(checkpoint['decoder_state_dict'])

    self.encoder.to(self.device)
    self.decoder.to(self.device)

  def _train(self, input_tensor, target_tensor, doprint = False):
    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    self.encoder_optimizer.zero_grad()
    self.decoder_optimizer.zero_grad()

    loss = 0
    encoder_output, encoder_hidden = self.encoder(input_tensor)
    encoder_hidden[0] = torch.unsqueeze(encoder_hidden[0], dim=0)
    encoder_hidden[1] = torch.unsqueeze(encoder_hidden[1], dim=0)

    pad_tensor = torch.tensor([[2]] * input_length, device=self.device)

    dec_state = encoder_hidden
    coverage_vector = torch.zeros(size=[encoder_output.size(dim=0), encoder_output.size(dim=1)], device=self.device)

    for di in range(self.max_decoder_seq_length - 1):
        new_tokens = target_tensor[:, di:di+2]
        input_token, target_token = new_tokens[:, 0:1], new_tokens[:, 1:2]

        if torch.all(torch.eq(input_token.argmax(dim=-1), pad_tensor)):
          break

        decoder_input = DecoderInput(new_tokens=input_token, encoder_output=encoder_output, encoder_state=dec_state, coverage_vector=coverage_vector, step=di)
        dec_result, dec_state, copy_logits = self.decoder(decoder_input)

        logits = torch.squeeze(dec_result.logits, dim=1)
        target_c = torch.squeeze(target_token).argmax(-1)

        if self.enable_coverage:
          coverage_vector = torch.add(coverage_vector, torch.squeeze(dec_result.attention_distribution, dim=1))

        if self.enable_copy:
          rinput = torch.argmax(input_tensor, dim=-1)
          copy_logits = torch.squeeze(copy_logits, dim=1)
          copy_logits = torch.where(rinput != 2, copy_logits, 0.0)

          indices_mask = torch.multiply(torch.arange(input_length, device=self.device).view(-1, 1), torch.ones_like(rinput, dtype=torch.int32, device=self.device))
          logits[indices_mask, rinput] += copy_logits

        new_loss = self.criterion(logits, target_c)
        loss += new_loss if not torch.isnan(new_loss) else 0.0

        if self.enable_diagonal_attention_loss:
          loss = loss + self._diagonal_attention_loss(dec_result.attention_distribution, di)

        if self.enable_coverage:
          loss = loss + self._compute_coverage_loss(coverage_vector, dec_result.attention_distribution)

    loss.backward()

    self.encoder_optimizer.step()
    self.decoder_optimizer.step()

    return loss.item() / target_length

  def _predict(self, data):
    input_length = data.size()[0]
    encoder_output, encoder_hidden = self.encoder(data)

    encoder_hidden[0] = torch.unsqueeze(encoder_hidden[0], dim=0)
    encoder_hidden[1] = torch.unsqueeze(encoder_hidden[1], dim=0)

    dec_state = encoder_hidden

    new_tokens = torch.zeros([input_length, 1, self.number_decoder_tokens], device=self.device)
    new_tokens[:, 0, self.decoder_text_processor.char_to_id[self.start_character]] = 1.0

    coverage_vector = torch.zeros(size=[encoder_output.size(dim=0), encoder_output.size(dim=1)], device=self.device)
    output_tokens = np.zeros([input_length, self.max_decoder_seq_length, self.number_decoder_tokens])

    for di in range(self.max_decoder_seq_length - 1):
        decoder_input = DecoderInput(new_tokens=new_tokens, encoder_output=encoder_output, encoder_state=dec_state, coverage_vector=coverage_vector, step=di)
        dec_result, dec_state, copy_logits = self.decoder(decoder_input)

        logits = torch.squeeze(dec_result.logits, dim=1)

        if self.enable_coverage:
          coverage_vector = torch.add(coverage_vector, torch.squeeze(dec_result.attention_distribution, dim=1))

        if self.enable_copy:
          rinput = torch.argmax(data, dim=-1)
          copy_logits = torch.squeeze(copy_logits, dim=1)
          copy_logits = torch.where(rinput != 2, copy_logits, 0.0)

          indices_mask = torch.multiply(torch.arange(input_length, device=self.device).view(-1, 1), torch.ones_like(rinput, dtype=torch.int32, device=self.device))
          logits[indices_mask, rinput] += copy_logits

        sampled_token = torch.argmax(logits, dim=-1)
        new_tokens = torch.zeros([input_length, 1, self.number_tokens], device=self.device)
        new_tokens[range(input_length), 0, sampled_token] = 1.0

        output_tokens[:, di, :] = logits.cpu().detach().numpy()

    return output_tokens

  def _predict_beam(self, data, beam_size=3):
    input_length = data.size()[0]
    encoder_output, encoder_hidden = self.encoder(data)

    encoder_hidden[0] = torch.unsqueeze(encoder_hidden[0], dim=0)
    encoder_hidden[1] = torch.unsqueeze(encoder_hidden[1], dim=0)

    dec_state = encoder_hidden
    print(self.number_decoder_tokens)
    print(self.decoder_text_processor.char_to_id[self.start_character])

    new_tokens = torch.zeros([input_length, 1, self.number_decoder_tokens], device=self.device)
    new_tokens[:, 0, self.decoder_text_processor.char_to_id[self.start_character]] = 1.0

    coverage_vector = torch.zeros(size=[encoder_output.size(dim=0), encoder_output.size(dim=1)], device=self.device)

    beam_outputs = [
        {
            "decoder_input": DecoderInput(new_tokens=new_tokens, encoder_output=encoder_output, encoder_state=dec_state, coverage_vector=coverage_vector, step=0),
            "tokens": [],
            "score": 0,
            "parent_node": None,
            "depth": 0
        }
    ]

    for di in range(self.max_decoder_seq_length - 1):
      new_beam_candidates = []

      for beam_output in beam_outputs:
        decoder_input = beam_output["decoder_input"]

        dec_result, dec_state, copy_logits = self.decoder(decoder_input)
        logits = torch.squeeze(dec_result.logits, dim=1) # batchSx1x98 -> bx98

        if self.enable_coverage:
          add_m = torch.minimum(coverage_vector, torch.squeeze(dec_result.attention_distribution, dim=1))
          coverage_vector = torch.add(coverage_vector, add_m)

        if self.enable_copy:
          rinput = torch.argmax(data, dim=-1)
          copy_logits = torch.squeeze(copy_logits, dim=1)
          copy_logits = torch.where(rinput != 2, copy_logits, 0.0)

          indices_mask = torch.multiply(torch.arange(input_length, device=self.device).view(-1, 1), torch.ones_like(rinput, dtype=torch.int32, device=self.device))
          logits[indices_mask, rinput] += copy_logits

        sampled_tokens = torch.topk(logits, k=beam_size, dim=-1, sorted=True)

        sampled_tokens_indices = torch.transpose(sampled_tokens.indices, dim0=1, dim1=0)
        sampled_tokens_values = torch.transpose(sampled_tokens.values, dim0=1, dim1=0)

        for u, sampled_token in enumerate(sampled_tokens_indices):
          new_tokens = torch.zeros([input_length, 1, self.number_decoder_tokens], device=self.device)

          new_tokens[torch.arange(sampled_tokens_indices.size()[1]), 0, sampled_token] = 1.0

          new_candidate = {
                  "decoder_input": DecoderInput(new_tokens=new_tokens, encoder_output=encoder_output, encoder_state=dec_state, coverage_vector=coverage_vector, step=di),
                  "score": beam_output["score"] + torch.sum(sampled_tokens_values[u]),
                  "tokens": beam_output["tokens"].copy(),
                  "parent_node": beam_output,
                  "depth": beam_output["depth"] + 1
          }

          new_candidate["tokens"].append(sampled_token)

          new_beam_candidates.append(new_candidate)

      beam_outputs = sorted(new_beam_candidates, key=lambda b: b["score"], reverse=True)[0:beam_size]

    return beam_outputs
  
  def eval_loader(self, dataloader):
    distances_input = []
    distances_model = []
    lens = []

    with torch.no_grad():
      for _, batch in enumerate(tqdm(dataloader, position=0, leave=True)):

        input_tensor, target_tensor = batch

        input_tensor_index = input_tensor.detach().numpy().argmax(axis=-1)
        target_tensorr_index = target_tensor.detach().numpy().argmax(axis=-1)

        input_words = []
        target_words = []
        actuals = []

        for i, w in enumerate(input_tensor_index):
          actual_input = ''.join([self.reverse_input_char_index[x] for x in w]).rstrip()

          input_words.append(actual_input)

        for i, w in enumerate(target_tensorr_index):
          target_input = ''.join([self.reverse_target_char_index[x] for x in w])

          target_input = target_input[:target_input.index(self.end_character)] if self.end_character in target_input else target_input.rstrip()
          target_input = target_input.replace('\t','')

          target_words.append(target_input)

        input_tensor = input_tensor.to(self.device)

        actuals = self.predict_text(input_tensor, 3, True)

        for inp, gs, act in zip(input_words, target_words, actuals):
          levenshtein_distance_i_gs = levenshtein_distance(inp, gs)
          levenshtein_distance_a_gs = levenshtein_distance(act, gs)

          distances_input.append(levenshtein_distance_i_gs)
          distances_model.append(levenshtein_distance_a_gs)

          lens.append(len(gs))

    print(distances_input)
    print(distances_model)

    score = []
    for x, y, z in zip(distances_input, distances_model, lens):
      t = (x - y) / z
      if t != 0:
        score.append(t)

    improvement = sum(score) / len(score)
    print("The improvement is {:.3f}%".format(improvement * 100))

    return improvement

  def predict_text(self, inputs, beam_size = 3, is_tensor = False):
    out = self.predict(inputs) if not is_tensor else self._do_predict(inputs, beam_size)

    actuals = []
    if self.use_beam:
      actuals = [''] * len(inputs)
      for t in out[0]["tokens"]:
        for j, x in enumerate(t.cpu().detach().numpy()):
          actuals[j] += self.reverse_target_char_index[x]
    else:
      sampled_outputs_val = np.argmax(out, axis=-1)

      for w in sampled_outputs_val:
        actual = ''.join([self.reverse_target_char_index[x] for x in w])
        actuals.append(actual)

    actuals = [t[:t.index(self.end_character)] if self.end_character in t else t.rstrip() for t in actuals]

    return actuals

  def _diagonal_attention_loss(self, attention_distribution, current_step):
    total_sum = 0.0
    if current_step - self.diag_loss_length >= 0:
      total_sum = total_sum + torch.sum(attention_distribution[:, 0, 0:(current_step - self.diag_loss_length)])

    total_sum = total_sum + torch.sum(attention_distribution[:, 0, (current_step + self.diag_loss_length + 1):])

    return total_sum

  def _compute_coverage_loss(self, coverage_vector, attention_distribution):
    total_sum = torch.sum(torch.minimum(coverage_vector, torch.squeeze(attention_distribution, dim=1)))

    return total_sum

  def _as_minutes(self, s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

  def _time_since(self, since, percent):
      now = time.time()
      s = now - since
      es = s / (percent)
      rs = es - s
      return '%s (- %s)' % (self._as_minutes(s), self._as_minutes(rs))