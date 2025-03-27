import torch.nn as nn
from dataset.dataset import *
from dataset.dataloader import get_rel2desc, load_data
from model.model import SimpleModel
from tqdm import tqdm
from torch.utils.data import Dataset, TensorDataset, DataLoader
from transformers import (BertTokenizer, get_linear_schedule_with_warmup,
                          BertConfig, BertForMaskedLM)


class SimpleTrainer():
    def __init__(self, args):
        self.args = args
        self.filename = self.args['filename']
        self.use_text_preprocesser = self.args['use_text_preprocesser']
        self.name_array, query_id_array, self.mention2id, self.edge_index, self.triples = load_data(
            self.filename, self.use_text_preprocesser, return_triples=True)  # load data
        self.queries_train, self.queries_valid, self.queries_test = data_split(query_id_array=query_id_array,
                                                                               is_unseen=self.args['is_unseen'],
                                                                               test_size=0.33)  # data split

        path = self.args['exp_path']
        torch.save(self.mention2id, os.path.join(path, 'mention2id.bin'))
        torch.save((self.queries_train, self.queries_valid, self.queries_test), os.path.join(path, 'queries.bin'))
        torch.save(self.name_array, os.path.join(path, 'name.bin'))
        torch.save(self.triples, os.path.join(path, 'triples.bin'))

        self.tokenizer = BertTokenizer.from_pretrained(self.args['pretrained_model'])
        self.model = SimpleModel(self.args['pretrained_model'], len(self.name_array), args)
        self.model.cuda()

        parameter_n = 0
        for param in self.model.parameters():
            mulValue = np.prod(param.size())
            parameter_n += mulValue
        print('Total parameters: {}'.format(parameter_n))

        self.args['logger'].info('get dataset')
        if 'DEBUG_LOAD_DATASET' in os.environ:
            self.train_dataset = torch.load(os.path.join(self.args['exp_path'], 'train_dataset'))
            self.valid_dataset = torch.load(os.path.join(self.args['exp_path'], 'valid_dataset'))
            self.test_dataset = torch.load(os.path.join(self.args['exp_path'], 'test_dataset'))
        else:
            self.train_dataset = self.get_dataset(query_array=self.queries_train, triples=self.triples)
            self.valid_dataset = self.get_dataset(query_array=self.queries_valid, triples=[])
            self.test_dataset = self.get_dataset(query_array=self.queries_test, triples=[])
            torch.save(self.train_dataset, os.path.join(self.args['exp_path'], 'train_dataset'))
            torch.save(self.valid_dataset, os.path.join(self.args['exp_path'], 'valid_dataset'))
            torch.save(self.test_dataset, os.path.join(self.args['exp_path'], 'test_dataset'))
        self.args['logger'].info('dataset done')

        self.model.ent_vocab.weight.data = self.get_ent_embeddings()

    def get_dataset(self, query_array, triples):
        entity_set = set([self.mention2id[i] for i in query_array])
        depth = self.args['path_depth']

        from collections import defaultdict
        id2mention = defaultdict(list)
        for m in query_array:
            id_ = self.mention2id[m]
            id2mention[id_].append(m)
            id2mention[id_].append(self.name_array[id_])
        id2mention = {k: sorted(list(set(v))) for k, v in id2mention.items()}
        id2mention = dict(id2mention)

        id2is_a = defaultdict(list)
        for h, r, t in triples:
            if r == 'is_a':
                id2is_a[h].append(t)
        id2is_a = dict(id2is_a)

        def lookup_is_a(id_):
            if id_ in id2is_a:
                return np.random.choice(id2is_a[id_])
            else:
                return -100

        def get_is_a_seq(id_, depth=2):
            x = (id_,)
            for i in range(depth - 1):
                x += (lookup_is_a(x[-1]),)
            return x

        pack = []

        path_template = ', which is a kind of {}'

        # synonyms
        inputs = []
        labels = []
        template = '{} is identical with {}'
        for q in query_array:
            inputs.append(template.format('[MASK]', q) + path_template.format('[MASK]') * (depth - 1))
            labels.append(get_is_a_seq(self.mention2id[q], depth))
        pack.append((inputs, labels))

        # entity name
        inputs = []
        labels = []
        template = '{} is identical with {}'
        for e in self.name_array:
            inputs.append(template.format('[MASK]', e) + path_template.format('[MASK]') * (depth - 1))
            labels.append(get_is_a_seq(self.mention2id[e], depth))
        pack.append((inputs, labels))

        # triples
        rel2desc = get_rel2desc(self.args['filename'])
        inputs = []
        labels = []
        template = '{} {} {}'

        def get_name(idx):
            if idx in entity_set:
                return np.random.choice(id2mention[idx])
            else:
                return self.name_array[idx]

        for h, r, t in triples:
            inputs.append(
                template.format(get_name(h), rel2desc[r], '[MASK]') + path_template.format('[MASK]') * (depth - 1))
            labels.append(get_is_a_seq(t, depth))
            inputs.append(
                template.format('[MASK]', rel2desc[r], get_name(t)) + path_template.format('[MASK]') * (depth - 1))
            labels.append((h,) + get_is_a_seq(t, depth)[1:])
        pack.append((inputs, labels))

        '''
        add contrastive learning loss, the intuition is similar terms should have similar embeddings
        '''
        inputs = []
        labels = []
        template = '{} is identical with {}'
        for h, r, t in triples:
            inputs.append(template.format('[MASK]', get_name(h)) + path_template.format('[MASK]') * (depth - 1))
            labels.append((t,) + get_is_a_seq(h, depth)[1: ])
        pack.append((inputs, labels))

        new_pack = []
        for inputs, labels in pack:
            if len(inputs) == len(labels) == 0:
                new_pack.append(None)
                continue
            input_ids, attention_mask = self.tokenize(inputs, self.args['max_seq_len'])
            labels = torch.LongTensor(labels)
            new_pack.append((input_ids, attention_mask, labels))

        synonym_dataset = TensorDataset(*new_pack[0])
        if new_pack[1] is None:
            entity_dataset = None
        else:
            entity_dataset = TensorDataset(*new_pack[1])
        if new_pack[2] is None:
            triple_dataset = None
        else:
            triple_dataset = TensorDataset(*new_pack[2])
        if new_pack[3] is None:
            cst_dataset = None
        else:
            cst_dataset = TensorDataset(*new_pack[3])
        return synonym_dataset, entity_dataset, triple_dataset, cst_dataset

    def tokenize(self, str_list, max_length):
        ret = self.tokenizer(str_list, add_special_tokens=True,
                             max_length=max_length, padding='max_length',
                             truncation=True, return_attention_mask=True,
                             return_tensors='pt')
        return ret.input_ids, ret.attention_mask

    def save(self, path=None):
        if path is None:
            path = self.args['exp_path']
        model_path = os.path.join(path, 'pytorch_model.bin')
        torch.save(self.model.state_dict(), model_path)
        self.args['logger'].info('model saved at %s' % path)

    def load(self, path=None):
        if path is None:
            path = self.args['exp_path']
        model_path = os.path.join(path, 'pytorch_model.bin')
        self.model.load_state_dict(torch.load(model_path))
        self.args['logger'].info('model loaded from %s' % path)

    def train_step(self, input_ids, attention_mask, labels, optimizer, scheduler, criterion, ent_emb=None):
        self.model.train()
        batch_size, seq_len = input_ids.shape

        optimizer.zero_grad()
        outputs = self.model.bert_lm.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state

        interest = outputs[input_ids == self.tokenizer.mask_token_id]
        # interest = interest.reshape(batch_size, -1)
        scores = self.model.get_ent_logits(interest, ent_emb)

        # labels = labels[(input_ids == self.tokenizer.mask_token_id).any(dim=1)] # avoid trunc out
        # avoid trunc out:
        labels, labels_bak = [], labels
        for i in range(batch_size):
            n_mask = (input_ids[i] == self.tokenizer.mask_token_id).long().sum()
            labels.append(labels_bak[i][:n_mask])
        labels = torch.cat(labels, dim=0)

        loss = criterion(scores, labels)

        loss.backward()
        if self.args['max_grad_norm'] > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args['max_grad_norm'])
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        return outputs, scores, loss

    def train(self):
        self.args['logger'].info('train')

        train_loader_syn = DataLoader(dataset=self.train_dataset[0], batch_size=self.args['batch_size'], shuffle=True)
        train_loader_ent = DataLoader(dataset=self.train_dataset[1], batch_size=self.args['batch_size'], shuffle=True)
        train_loader_hrt = DataLoader(dataset=self.train_dataset[2], batch_size=self.args['batch_size'], shuffle=True)
        # This dataset is responsible for contrastive learning
        train_loader_cst = DataLoader(dataset=self.train_dataset[3], batch_size=self.args['batch_size'], shuffle=True)
        syn_size = len(train_loader_syn)

        def make_infinite_dataloader(dataloader):
            while True:
                for i in dataloader:
                    yield i

        train_loader_ent = make_infinite_dataloader(train_loader_ent)
        train_loader_hrt = make_infinite_dataloader(train_loader_hrt)
        train_loader_cst = make_infinite_dataloader(train_loader_cst)
        # train_loader_sib = make_infinite_dataloader(train_loader_sib)
        # train_loader_grand = make_infinite_dataloader(train_loader_grand)

        print('syn dataset: ', len(self.train_dataset[0]))
        print('ent dataset: ', len(self.train_dataset[1]))
        print('hrt dataset: ', len(self.train_dataset[2]))
        print('cst dataset: ', len(self.train_dataset[3]))
        import sys;
        sys.stdout.flush()

        criterion = nn.CrossEntropyLoss(reduction='mean')
        optimizer = torch.optim.Adam(
            # self.model.parameters(),
            [{'params': self.model.bert_lm.parameters()},
             {'params': self.model.ent_vocab.parameters(), 'lr': 1e-3, 'weight_decay': 0},
             {'params': self.model._cls_bn.parameters(), 'lr': 1e-3, 'weight_decay': 0}],
            lr=self.args['lr'], weight_decay=self.args['weight_decay']
        )

        loader_selector = ([0] * syn_size
                           + [1] * int(syn_size / self.args['syn_ratio'] * self.args['ent_ratio'])
                           + [2] * int(syn_size / self.args['syn_ratio'] * self.args['hrt_ratio'])
                           + [3] * int(syn_size / self.args['syn_ratio'] * self.args['cst_ratio'])
                           )
        loaders = [train_loader_syn, train_loader_ent, train_loader_hrt, train_loader_cst
                   ]

        pbar = tqdm(range(self.args['pretrain_emb_iter']), desc='pretrain emb')
        for iteration in pbar:
            self.model.eval()
            if self.args['use_get_ent_emb'] and iteration % 100 == 0:
                ent_emb = self.get_ent_embeddings()
            else:
                ent_emb = None
            task = 1
            loader = loaders[task]
            batch = next(loader)

            batch = (i.cuda() for i in batch)
            input_ids, attention_mask, labels = batch
            outputs, interest, loss = self.train_step(input_ids, attention_mask, labels, optimizer, None, criterion,
                                                      ent_emb)
            m, M, mean, std = interest.min(), interest.max(), interest.mean(), interest.std()
            pbar.set_postfix({("loss%d" % task): float(loss),
                              "[min, max, mean, std]": ['%.2e' % i for i in [m, M, mean, std]],
                              "lr": ['%.2e' % group["lr"] for group in optimizer.param_groups]})

            if 'DEBUG_DECODE_OUTPUT' in os.environ and iteration % 100 == 0:
                print(self.tokenizer.batch_decode(input_ids[:10], skip_special_tokens=True))
                print([self.name_array[i] for i in interest[:10].argmax(dim=-1).tolist()])
                print([self.name_array[i] for i in labels[:10].tolist()])
                import sys;
                sys.stdout.flush()

        t_total = self.args['epoch_num'] * len(loader_selector)
        if self.args['use_scheduler']:
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=t_total)
        else:
            scheduler = None

        for epoch in range(1, self.args['epoch_num'] + 1):
            torch.cuda.empty_cache()
            self.model.train()
            np.random.shuffle(loader_selector)
            selector_idx = 0
            loss_sum = 0
            pbar = tqdm(enumerate(train_loader_syn), total=len(train_loader_syn))
            for iteration, syn_batch in pbar:
                self.model.eval()
                if self.args['use_get_ent_emb'] and iteration % 100 == 0:
                    ent_emb = self.get_ent_embeddings()
                else:
                    ent_emb = None
                while loader_selector[selector_idx] != 0:
                    task = loader_selector[selector_idx]
                    loader = loaders[task]
                    batch = next(loader)

                    batch = (i.cuda() for i in batch)
                    input_ids, attention_mask, labels = batch
                    outputs, interest, loss = self.train_step(input_ids, attention_mask, labels, optimizer, scheduler,
                                                              criterion, ent_emb)
                    loss_sum += loss.item() * len(input_ids)
                    m, M, mean, std = interest.min(), interest.max(), interest.mean(), interest.std()
                    pbar.set_postfix({("loss%d" % task): float(loss),
                                      "[min, max, mean, std]": ['%.2e' % i for i in [m, M, mean, std]],
                                      "lr": ['%.2e' % group["lr"] for group in optimizer.param_groups]})
                    if iteration % 100 == 0:
                        print(self.tokenizer.batch_decode(input_ids[:2], skip_special_tokens=True))
                        print([self.name_array[i] for i in interest[:4].argmax(dim=-1).tolist()])
                        print([self.name_array[i] for i in labels[:2].tolist()])
                        import sys;
                        sys.stdout.flush()
                    selector_idx += 1

                task = loader_selector[selector_idx]
                batch = syn_batch

                batch = (i.cuda() for i in batch)
                input_ids, attention_mask, labels = batch
                outputs, interest, loss = self.train_step(input_ids, attention_mask, labels, optimizer, scheduler,
                                                          criterion, ent_emb)
                loss_sum += loss.item() * len(input_ids)
                m, M, mean, std = interest.min(), interest.max(), interest.mean(), interest.std()
                pbar.set_postfix({("loss%d" % task): float(loss),
                                  "[min, max, mean, std]": ['%.2e' % i for i in [m, M, mean, std]],
                                  "lr": ['%.2e' % group["lr"] for group in optimizer.param_groups]})
                selector_idx += 1

                if 'DEBUG_DECODE_OUTPUT' in os.environ and iteration % 100 == 0:
                    print(self.tokenizer.batch_decode(input_ids[:10], skip_special_tokens=True))
                    print([self.name_array[i] for i in interest[:10].argmax(dim=-1).tolist()])
                    print([self.name_array[i] for i in labels[:10].tolist()])
                    import sys;
                    sys.stdout.flush()

            # print('train')
            # accu_1, accu_k = self.eval(self.train_dataset, epoch=epoch)
            # print('valid')
            accu_1, accu_k = self.eval(self.valid_dataset, epoch=epoch)
            loss_sum /= len(loader_selector)
            print('loss_sum:', float(loss_sum))
            import sys;
            sys.stdout.flush()

    @torch.no_grad()
    def get_ent_embeddings(self, ):
        ent_embeddings = torch.zeros_like(self.model.ent_vocab.weight)
        data_loader = DataLoader(dataset=self.train_dataset[1], batch_size=64)

        for i, batch in enumerate(data_loader):
            (input_ids, attention_mask, labels) = (i.cuda() for i in batch)
            outputs = self.model.bert_lm.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).last_hidden_state
            assert (input_ids[:, 1] == self.tokenizer.mask_token_id).all()
            emb = self.model.bert_lm.cls.predictions.transform(outputs[:, 1])
            ent_embeddings[labels[:, 0]] = emb

        return ent_embeddings

    @torch.no_grad()
    def eval(self, eval_dataset, epoch, return_output=False):
        self.model.eval()
        assert len(eval_dataset) == 4
        eval_dataset = eval_dataset[0]
        eval_loader = DataLoader(dataset=eval_dataset, batch_size=64, shuffle=False)

        pack = [[], [], []]

        accu_1 = torch.FloatTensor([0]).cuda()
        accu_k = torch.FloatTensor([0]).cuda()

        pbar = tqdm(enumerate(eval_loader), total=len(eval_loader))
        if self.args['use_get_ent_emb']:
            ent_emb = self.get_ent_embeddings()
        else:
            ent_emb = None
        for iteration, batch in pbar:
            batch = (i.cuda() for i in batch)
            input_ids, attention_mask, labels = batch
            batch_size, seq_len = input_ids.shape

            outputs = self.model.bert_lm.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).last_hidden_state

            assert (input_ids[:, 1] == self.tokenizer.mask_token_id).all()
            interest = outputs[:, 1]
            # interest = outputs[input_ids == self.tokenizer.mask_token_id]
            # interest = interest.reshape(batch_size, -1)
            scores = self.model.get_ent_logits(interest, ent_emb)

            sorted_scores, sorted_indices = torch.sort(scores, descending=True)
            pack[0].append(sorted_scores.clone().detach()[:, :100].cpu())
            pack[1].append(sorted_indices.clone().detach()[:, :100].cpu())
            pack[2].append(labels.clone().detach().cpu())
            # labels = labels[(input_ids == self.tokenizer.mask_token_id).any(dim=1)] # avoid trunc out
            labels = labels[(input_ids == self.tokenizer.mask_token_id).any(dim=1)][:, 0]  # avoid trunc out
            accu_1 += (sorted_indices[:, 0] == labels).sum() / len(eval_dataset)
            accu_k += (sorted_indices[:, :self.args['eval_k']] == labels.unsqueeze(dim=1)).sum() / len(eval_dataset)

            if 'DEBUG_DECODE_EVAL' in os.environ and iteration % 100 == 0:
                print(self.tokenizer.decode(input_ids[0]))
                print(sorted_scores[:10])
                print(sorted_indices[:10])
                import sys;
                sys.stdout.flush()

        self.args['logger'].info(
            "epoch %d done, accu_1 = %f, accu_%d = %f" % (epoch, float(accu_1), self.args['eval_k'], float(accu_k)))
        if accu_1 < 1e-4 and epoch != 0:
            print('bad args!')
            exit(-1)
        if return_output:
            pack = [torch.cat(i, dim=0) for i in pack]
            pack = {'labels': pack[-1], 'scores': pack[0], 'idx': pack[1],
                    'ent_emb': ent_emb.clone().detach().cpu() if ent_emb is not None else self.model.ent_vocab.weight.clone().detach().cpu()}
            return accu_1, accu_k, pack
        return accu_1, accu_k