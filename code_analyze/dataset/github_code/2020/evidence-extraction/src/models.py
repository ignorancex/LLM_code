import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchcrf import CRF


class BertForSequenceClassification(nn.Module):
    def __init__(
        self,
        bert_model,
        num_classes,
        dropout_prob=0.1,
    ):
        super(BertForSequenceClassification, self).__init__()
        self.hidden_dim = 768
        self.num_classes = num_classes
        self.bert = bert_model
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(self.hidden_dim, self.num_classes)
        self.bow_classifier = nn.Linear(self.hidden_dim, self.num_classes) # bag of word classfier
        # initialize the parameters
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.xavier_uniform_(self.bow_classifier.weight)
        nn.init.constant_(self.classifier.bias, 0.0)
        nn.init.constant_(self.bow_classifier.bias, 0.0)


    def forward(self, input_ids, attention_mask):
        """ forward pass to classify the input tokens
        
        Parameters
        ----------
        input_ids : torch.[cuda?].LongTensor
            input token ids, shape --> batch_size x max_seq_len
        attention_mask: torch.[cuda?].FloatTensor 
            mask that prevents attention over padded tokens. 
            contains values 1 for yes, 0 for not attending
            shape --> batch_size x max_seq_len
        
        Returns
        -------
        torch.[cuda?].FloatTensor
            output logits 
        """
        _, pooled_cls_output, _ = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # pooled_cls_output is batch_size x 746
        output = self.dropout(pooled_cls_output)
        output = self.classifier(output)
        # output is batch_size x num_classes
        return output


    def neg_log_likelihood(self, input_ids, attention_mask, labels, return_output=False):
        """[summary]
        
        Parameters
        ----------
        input_ids : torch.[cuda?].LongTensor
            input token ids, shape --> batch_size x max_seq_len
        attention_mask: torch.[cuda?].FloatTensor 
            mask that prevents attention over padded tokens. 
            contains values 1 for yes, 0 for not attending
            shape --> batch_size x max_seq_len
        labels : torch.[cuda?].LongTensor
            correct labels, shape --> batch_size (or batch_size x 1)
        
        Returns
        -------
        torch.[cuda?].FloatTensor
            the loss value (averaged over the batch)
        """
        output = self.forward(input_ids, attention_mask) 
        cross_entropy_loss = nn.CrossEntropyLoss()
        # the default reduction is mean
        # NOTE: there is no need to ignore any output... there is exactly one output per sentence
        loss = cross_entropy_loss(output, labels)
        
        if return_output:
            output = torch.argmax(output, dim=-1)
            return loss, output

        return loss

    def bow_neg_log_likelihood(self, input_ids, attention_mask, labels, return_output=False):
        """[summary]
        
        Parameters
        ----------
        input_ids : torch.[cuda?].LongTensor
            input token ids, shape --> batch_size x max_seq_len
        attention_mask: torch.[cuda?].FloatTensor 
            mask that prevents attention over padded tokens. 
            contains values 1 for yes, 0 for not attending
            shape --> batch_size x max_seq_len
        labels : torch.[cuda?].LongTensor
            correct labels, shape --> batch_size (or batch_size x 1)
        
        Returns
        -------
        torch.[cuda?].FloatTensor
            the loss value (averaged over the batch)
        """
        output = self.bert.embeddings.word_embeddings(input_ids)
        # output is batch_size x max_len x 768

        # remove the contributions from 0 positions as per attention mask
        output = torch.einsum("bld, bl->bld", output, attention_mask)

        pooled_output = torch.sum(output, dim=1) 
        # pooled_output is batch_size x 746

        output = self.bow_classifier(pooled_output)

        # forward(input_ids, attention_mask) 
        cross_entropy_loss = nn.CrossEntropyLoss()
        # the default reduction is mean
        # NOTE: there is no need to ignore any output... there is exactly one output per sentence
        loss = cross_entropy_loss(output, labels)
        
        if return_output:
            output = torch.argmax(output, dim=-1)
            return loss, output

        return loss


    def kl_divergence_loss(self, input_ids, attention_mask, tags):
        """computes the kl-divergence loss, i.e. 
        KL(P || Q) where P is expected output distribution (uniform over gold tags, 0 otherwise)
        and Q is the average [CLS] attention across all heads
        
        Parameters
        ----------
        input_ids : torch.[cuda?].LongTensor
            input token ids, shape --> batch_size x max_seq_len
        attention_mask: torch.[cuda?].FloatTensor 
            mask that prevents attention over padded tokens. 
            contains values 1 for yes, 0 for not attending
            shape --> batch_size x max_seq_len
        tags : torch.[cuda?].LongTensor
            tags for which the attention is supposed to be high, 1 for those tokens, 0 otherwise 
            shape --> batch_size x max_seq_len 
        
        Returns
        -------
        torch.[cuda?].FloatTensor
            the loss value (averaged over the batch)
        """
        _, _, attentions = self.bert(input_ids, attention_mask=attention_mask)
        #NOTE: I assume that the attentions corresponding to 0s in attention_mask is 0.0
        #TODO: confirm this case

        # normalize the tags
        updated_tags = tags + 1e-9
        normalizing_const = torch.sum(updated_tags, dim=1)
        normalized_tags = torch.einsum('ij,i->ij', updated_tags, 1.0/normalizing_const)

        # attentions is a tuple of 12 (layers), 
        # each of shape --> batch_size x 12 (for heads) x max_seq_len x max_seq_len
        last_layer_attention = attentions[-1]
        last_layer_CLS_attention = last_layer_attention[:, :, 0, :]
        # shape of last_layer_CLS_attention --> batch_size x 12 x max_seq_len
        last_layer_CLS_attention = last_layer_CLS_attention.permute(0, 2, 1)
        # shape of last_layer_CLS_attention --> batch_size x max_seq_len x 12        
        last_layer_CLS_attention_avg = torch.mean(last_layer_CLS_attention, dim=-1)
        last_layer_CLS_attention_avg_log = torch.log(last_layer_CLS_attention_avg + 1e-9)

        kld_loss = nn.KLDivLoss(reduction='batchmean')  # by default the reduction is 'mean'
        loss = kld_loss(last_layer_CLS_attention_avg_log, normalized_tags)

        return loss


    def get_top_tokens(self, input_ids, attention_mask, k=10):
        """ get the top k% tokens as per attention scores

        Parameters
        ----------
        input_ids : torch.[cuda?].LongTensor
            input token ids, shape --> batch_size x max_seq_len
        attention_mask: torch.[cuda?].FloatTensor 
            mask that prevents attention over padded tokens. 
            contains values 1 for yes, 0 for not attending
            shape --> batch_size x max_seq_len
        k : int, optional
            the k in top-k, by default 10

        Returns
        -------
        List[List[int]]
            a binary footprint of what is selected in the top-k (marked by 1, others as 0) 
        """
        _, _, attention = self.bert(input_ids, attention_mask)
        # for each layer attention is (batch_size, num_heads, sequence_length, sequence_length)
        last_layer_attention = attention[-1]
        
        # aggregate all the attention heads by mean 
        last_layer_mean_attention = torch.mean(last_layer_attention, dim=1)
        # last_layer_mean_attention is (batch_size, sequence_length, sequence_length)

        last_layer_mean_CLS_attention = last_layer_mean_attention[:, 0, :]
        # last_layer_mean_CLS_attention is (batch_size, sequence_length)

        # NOTE: at this point converting to lists also in not a big time burden 
        
        output = []

        for i in range(len(last_layer_mean_CLS_attention)):

            line_len = len(input_ids[i].nonzero())  # including [CLS] and [SEP]

            # shape of score is line_len
            score = last_layer_mean_CLS_attention[i][:line_len]
            token_ids = input_ids[i][:line_len]

            num_words = int(0.01 * k * len(token_ids))
            if num_words == 0:
                # should at least contain a word 
                num_words = 1

            selected_indices = torch.argsort(score, descending=True)[:num_words]
            top_k_mask = [1.0 if i in selected_indices else 0.0 for i in range(len(token_ids))]
            output.append(top_k_mask)

        return output



class BertCRF(nn.Module):
    """ A CRF model to generate the I-O tags for rationales in input examples
        
        Parameters
        ----------
        bert_model : BertModel
            An instance of the pretrained bert model
        start_label_id : int
            The index of the <START TAG>
        stop_label_id : int
            The index of the <STOP TAG>
        num_labels : int
            number of output labels
        batch_size : int
            batch size of the input sequences, by defualt 32
        dropout_prob: float
            dropout probability, by default 0.2
    """

    def __init__(
            self, 
            bert_model, 
            start_label_id, 
            stop_label_id, 
            num_labels, 
            dropout_prob=0.1,
        ):
        super(BertCRF, self).__init__()
        self.bert_features_dim = 768 
        self.attention_features_dim = 12 # corresponding to the number of heads
        self.label_features_dim = 1 
        self.num_labels = num_labels
        self.start_label_id = start_label_id
        self.stop_label_id = stop_label_id
        self.bert =  bert_model 
        self.crf = CRF(num_labels, batch_first=True)

        self.bert_features_to_label = nn.Linear(self.bert_features_dim, self.num_labels)
        self.double_bert_features_to_label = nn.Linear(2*self.bert_features_dim, self.num_labels)
        self.individual_bert_features_to_label = nn.Linear(self.bert_features_dim, self.num_labels)
        self.log_normalized_bert_features_to_label = nn.Linear(self.bert_features_dim, self.num_labels)
        # NOTE: one can share the individual feature and regular features weights 
        self.attention_features_to_label = nn.Linear(self.attention_features_dim, self.num_labels)
        self.avg_attention_features_to_label = nn.Linear(1, self.num_labels) # avg attn features
        self.label_features_to_label = nn.Linear(self.label_features_dim, self.num_labels)
        self.bow_features_to_label = nn.Linear(self.label_features_dim, self.num_labels)
        # NOTE: one can inspect the label_features_to_label  weights for sanity checking

        # initialize the parameters
        nn.init.xavier_uniform_(self.bert_features_to_label.weight)
        nn.init.xavier_uniform_(self.double_bert_features_to_label.weight)
        nn.init.xavier_uniform_(self.individual_bert_features_to_label.weight)
        nn.init.xavier_uniform_(self.log_normalized_bert_features_to_label.weight)
        nn.init.xavier_uniform_(self.attention_features_to_label.weight)
        nn.init.xavier_uniform_(self.avg_attention_features_to_label.weight)
        nn.init.xavier_uniform_(self.label_features_to_label.weight)
        nn.init.xavier_uniform_(self.bow_features_to_label.weight)
        nn.init.constant_(self.bert_features_to_label.bias, 0.0)
        nn.init.constant_(self.double_bert_features_to_label.bias, 0.0)
        nn.init.constant_(self.individual_bert_features_to_label.bias, 0.0)
        nn.init.constant_(self.log_normalized_bert_features_to_label.bias, 0.0)
        nn.init.constant_(self.attention_features_to_label.bias, 0.0)
        nn.init.constant_(self.avg_attention_features_to_label.bias, 0.0)
        nn.init.constant_(self.label_features_to_label.bias, 0.0)
        nn.init.constant_(self.bow_features_to_label.bias, 0.0)

        is_cuda = torch.cuda.is_available()

        self.float_type = torch.FloatTensor
        self.long_type = torch.LongTensor
        self.byte_type = torch.ByteTensor

        if is_cuda:
            self.float_type = torch.cuda.FloatTensor 
            self.long_type = torch.cuda.LongTensor
            self.byte_type = torch.cuda.ByteTensor

    def forward(
        self, 
        input_ids, 
        attention_mask,
        include_bert_features=False,
        include_double_bert_features=False,
        include_log_normalized_bert_features=False,
        include_attention_features=False,
        include_avg_attention_features=False,
        include_individual_bert_features=False,
        include_label_features=False,
        include_bow_features=False,
        classifier=None,
        bow_classifier=None,
        output_labels = None,
    ):
        """ forward pass of the model class
        
        Parameters
        ----------
        input_ids : torch.[cuda?].LongTensor
            input ids of sentences. shape ==> batch_size x max_seq_len 
        attention_mask : [type]
            [description]
        include_bert_features : bool, optional
            [description], by default False
        include_attention_features : bool, optional
            [description], by default False
        include_bert_features : bool, optional
            [description], by default False
        include_attention_features : bool, optional
            [description], by default False
        
        Returns
        -------
        List[List[int]]
            best path...
        """

        feats = self._get_all_features(input_ids, 
            attention_mask,
            include_bert_features,
            include_double_bert_features,
            include_log_normalized_bert_features,
            include_attention_features,
            include_avg_attention_features,
            include_individual_bert_features,
            include_label_features,
            include_bow_features,
            classifier,
            bow_classifier,
            output_labels,
        )


        mask = attention_mask.type(self.byte_type)

        return self.crf.decode(feats, mask=mask)


    def neg_log_likelihood(
        self,
        input_ids, 
        attention_mask,
        label_ids,
        include_bert_features=False,
        include_double_bert_features=False,
        include_log_normalized_bert_features=False,
        include_attention_features=False,
        include_avg_attention_features=False,
        include_individual_bert_features=False,
        include_label_features=False,
        include_bow_features=False,
        classifier=None,
        bow_classifier=None,
        output_labels = None,
    ):

        feats = self._get_all_features(input_ids, 
            attention_mask,
            include_bert_features,
            include_double_bert_features,
            include_log_normalized_bert_features,
            include_attention_features,
            include_avg_attention_features,
            include_individual_bert_features,
            include_label_features,
            include_bow_features,
            classifier,
            bow_classifier,
            output_labels,
        )


        mask = attention_mask.type(self.byte_type)

        log_likelihood = self.crf(feats, label_ids, mask=mask, reduction='token_mean') 
        return -log_likelihood


    def print_weights(self,
        ITER,
        include_avg_attention_wts=False,
        include_bow_wts=False
    ):
        """ Print transition and other weights

        Parameters
        ----------
        ITER : int
            epoch count
        include_avg_attention_wts : bool, optional
            whether to print attention weights, by default False
        include_bow_wts : bool, optional
            whether to print bow weights, by default False
        """

        O_tag = 2
        I_tag = 3

        transitions = self.crf.transitions

        print("[transitions] ITER: %d I-I: %.2f O-O: %.2f I-0 %.2f 0-I %.2f" % (
            ITER,
            transitions[I_tag][I_tag].item(),
            transitions[O_tag][O_tag].item(),
            transitions[I_tag][O_tag].item(),
            transitions[O_tag][I_tag].item(),
        ))

        if include_avg_attention_wts:
            print("[attention weights] ITER: %d W_+: %.2f b_+: %.2f W_ %.2f b_ %.2f" % (
                ITER,
                self.avg_attention_features_to_label.weight.data[1].item(),
                self.avg_attention_features_to_label.bias.data[1].item(),
                self.avg_attention_features_to_label.weight.data[0].item(),
                self.avg_attention_features_to_label.bias.data[0].item(),
            ))

        if include_bow_wts:
            print("[bow weights] ITER: %d W_+: %.2f b_+: %.2f W_ %.2f b_ %.2f" % (
                ITER,
                self.bow_features_to_label.weight.data[1].item(),
                self.bow_features_to_label.bias.data[1].item(),
                self.bow_features_to_label.weight.data[0].item(),
                self.bow_features_to_label.bias.data[0].item(),
            ))
        
        return 

    def _get_all_features(
        self,
        input_ids, 
        attention_mask,
        include_bert_features=False,
        include_double_bert_features=False,
        include_log_normalized_bert_features=False,
        include_attention_features=False,
        include_avg_attention_features=False,
        include_individual_bert_features=False,
        include_label_features=False,
        include_bow_features=False,
        classifier=None,
        bow_classifier=None,
        output_labels = None,
    ):
        batch_size, max_seq_len = input_ids.shape

        # init emission features
        feats = torch.zeros((batch_size, max_seq_len, self.num_labels)).type(self.float_type)

        if include_bert_features:
            bert_feats = self._get_bert_features(input_ids, attention_mask)
            feats += bert_feats

        if include_double_bert_features:
            if output_labels is None:
                raise Exception("Need to have the output labels to get the double bert features")
            double_bert_feats = self._get_double_bert_features(input_ids, attention_mask, output_labels)
            feats += double_bert_feats

        if include_log_normalized_bert_features:
            log_normalized_bert_features = self._get_log_normalized_bert_features(input_ids, attention_mask)
            feats += log_normalized_bert_features
        
        if include_attention_features:
            attention_features = self._get_attention_features(input_ids, attention_mask)
            feats += attention_features

        if include_avg_attention_features:
            avg_attention_features = self._get_avg_attention_features(input_ids, attention_mask)
            feats += avg_attention_features

        if include_individual_bert_features:
            individual_bert_features = self._get_individual_bert_features(input_ids, attention_mask)
            feats += individual_bert_features
        
        if include_label_features:

            # check if we have the classifier and the output labels
            if classifier is None or output_labels is None:
                raise Exception("Need to pass the classfier and a tensor specifying the outputs")

            label_features = self._get_label_features(input_ids, attention_mask, classifier, 
                                                        output_labels)
            feats += label_features


        if include_bow_features:
            if bow_classifier is None or output_labels is None:
                raise Exception("Need to pass the classfier and a tensor specifying the outputs")

            bow_feats = self._get_bow_features(input_ids, attention_mask, bow_classifier, 
                                                        output_labels)
            feats += bow_feats


        return feats


    def _get_attention_features(self, input_ids, attention_mask):
        """ get features from BERT's attention 

        Parameters
        ----------
        input_ids : torch.[cuda?].LongTensor
            input token ids, shape --> batch_size x max_seq_len
        attention_mask: torch.[cuda?].FloatTensor 
            mask that prevents attention over padded tokens. 
            contains values 1 for yes, 0 for not attending
            shape --> batch_size x max_seq_len
        
        Returns
        -------
        torch.[cuda?].FloatTensor
            scores of shape batch_size x max_seq_len x num_labels
        """
        _, _, attentions = self.bert(input_ids, attention_mask=attention_mask)

        # attentions is a tuple of 12 (layers), 
        # each of shape --> batch_size x 12 (for heads) x max_seq_len x max_seq_len
        last_layer_attention = attentions[-1]
        last_layer_CLS_attention = last_layer_attention[:, :, 0, :]
        # shape of last_layer_CLS_attention --> batch_size x 12 x max_seq_len
        last_layer_CLS_attention = last_layer_CLS_attention.permute(0, 2, 1)
        # shape of last_layer_CLS_attention --> batch_size x max_seq_len x 12        
        #NOTE: the attention features should be in log space 
        log_last_layer_CLS_attention = torch.log(last_layer_CLS_attention + 1e-9)
        attention_feats = self.attention_features_to_label(log_last_layer_CLS_attention)
        return attention_feats


    def _get_avg_attention_features(self, input_ids, attention_mask):
        """ get averaged features from BERT's attention

        Parameters
        ----------
        input_ids : torch.[cuda?].LongTensor
            input token ids, shape --> batch_size x max_seq_len
        attention_mask: torch.[cuda?].FloatTensor 
            mask that prevents attention over padded tokens. 
            contains values 1 for yes, 0 for not attending
            shape --> batch_size x max_seq_len
        
        Returns
        -------
        torch.[cuda?].FloatTensor
            scores of shape batch_size x max_seq_len x num_labels
        """
        _, _, attentions = self.bert(input_ids, attention_mask=attention_mask)

        # attentions is a tuple of 12 (layers), 
        # each of shape --> batch_size x 12 (for heads) x max_seq_len x max_seq_len
        last_layer_attention = attentions[-1]
        last_layer_CLS_attention = last_layer_attention[:, :, 0, :]
        # shape of last_layer_CLS_attention --> batch_size x 12 x max_seq_len
        last_layer_CLS_attention = last_layer_CLS_attention.permute(0, 2, 1)
        # shape of last_layer_CLS_attention --> batch_size x max_seq_len x 12        
        last_layers_CLS_attention_avg = torch.mean(
            last_layer_CLS_attention, dim=-1).unsqueeze(dim=-1)

        # add log-features 
        last_layers_CLS_attention_avg_log = torch.log(last_layers_CLS_attention_avg + 1e-9) 
        avg_attention_feats = self.avg_attention_features_to_label(last_layers_CLS_attention_avg_log)
        return avg_attention_feats

    def _get_bert_features(self, input_ids, attention_mask):
        """ get features from BERT 

        Parameters
        ----------
        input_ids : torch.[cuda?].LongTensor
            input token ids, shape --> batch_size x max_seq_len
        attention_mask: torch.[cuda?].FloatTensor 
            mask that prevents attention over padded tokens. 
            contains values 1 for yes, 0 for not attending
            shape --> batch_size x max_seq_len
        
        Returns
        -------
        torch.[cuda?].FloatTensor
            scores of shape batch_size x max_seq_len x num_labels
        """
        bert_seq_out, _, _ = self.bert(input_ids, attention_mask=attention_mask)

        bert_feats = self.bert_features_to_label(bert_seq_out)

        return bert_feats


    def _get_double_bert_features(self, input_ids, attention_mask, output_labels):
        """ get bert features such that the output label 1 are on the left, and 0 on the rigt

        Parameters
        ----------
        input_ids : torch.[cuda?].LongTensor
            input token ids, shape --> batch_size x max_seq_len
        attention_mask: torch.[cuda?].FloatTensor 
            mask that prevents attention over padded tokens. 
            contains values 1 for yes, 0 for not attending
            shape --> batch_size x max_seq_len
        output_labels : torch.[cuda?].LongTensor
            output predictions (or even possibly ground truth)
            shape --> [batch_size]

        Returns
        -------
        torch.[cuda?].FloatTensor
            features ...
        """

        #NOTE: this only works for a binary classification task..

        
        bert_seq_out, _, _ = self.bert(input_ids, attention_mask=attention_mask)
        # shape fo bert_seq_out is batch_size x max_seq_len x 746

        first_half = torch.einsum('ijk,i->ijk', bert_seq_out, output_labels)
        second_half = torch.einsum('ijk,i->ijk', bert_seq_out, (1 - output_labels))

        output = torch.cat((first_half, second_half), dim=-1)
    
        double_bert_feats = self.double_bert_features_to_label(output)

        return double_bert_feats


    def _get_log_normalized_bert_features(self, input_ids, attention_mask):
        """ get features from BERT after first normalizing (softmax) and then log 

        Parameters
        ----------
        input_ids : torch.[cuda?].LongTensor
            input token ids, shape --> batch_size x max_seq_len
        attention_mask: torch.[cuda?].FloatTensor 
            mask that prevents attention over padded tokens. 
            contains values 1 for yes, 0 for not attending
            shape --> batch_size x max_seq_len
        
        Returns
        -------
        torch.[cuda?].FloatTensor
            scores of shape batch_size x max_seq_len x num_labels
        """
        bert_seq_out, _, _ = self.bert(input_ids, attention_mask=attention_mask)

        normalized_bert_seq_out = nn.functional.softmax(bert_seq_out, dim=-1)

        log_normalized_bert_seq_out = torch.log(normalized_bert_seq_out + 1e-9) 

        bert_feats = self.log_normalized_bert_features_to_label(log_normalized_bert_seq_out)

        return bert_feats



    def _get_individual_bert_output(self, input_ids, attention_mask):
        """ get **output** from BERT for individual tokens
        don't confuse with _get_individual_bert_features

        Parameters
        ----------
        input_ids : torch.[cuda?].LongTensor
            input token ids, shape --> batch_size x max_seq_len
        attention_mask: torch.[cuda?].FloatTensor 
            mask that prevents attention over padded tokens. 
            contains values 1 for yes, 0 for not attending
            shape --> batch_size x max_seq_len

        Returns
        -------
        torch.[cuda?].FloatTensor
            scores of shape batch_size x max_seq_len x bert_output_dim
        """
        batch_size, max_seq_len = input_ids.shape
        bert_seq_out = torch.zeros(batch_size, max_seq_len, self.bert_features_dim).type(
            self.float_type)

        for i in range(max_seq_len):
            # pass individual tokens...
            bert_output, _, _ = self.bert(input_ids[:, i].unsqueeze(dim=1),
                                            attention_mask[:, i].unsqueeze(dim=1))
            bert_seq_out[:, i, :] = bert_output[:, 0, :]

        return bert_seq_out



    def _get_individual_bert_features(self, input_ids, attention_mask):
        """ get features from BERT for individual tokens

        Parameters
        ----------
        input_ids : torch.[cuda?].LongTensor
            input token ids, shape --> batch_size x max_seq_len
        attention_mask: torch.[cuda?].FloatTensor 
            mask that prevents attention over padded tokens. 
            contains values 1 for yes, 0 for not attending
            shape --> batch_size x max_seq_len
        
        Returns
        -------
        torch.[cuda?].FloatTensor
            scores of shape batch_size x max_seq_len x num_labels
        """

        bert_seq_out = self._get_individual_bert_output(input_ids, attention_mask)
        individual_bert_feats = self.individual_bert_features_to_label(bert_seq_out)

        return individual_bert_feats


    def _get_label_features(self, input_ids, attention_mask, classifier, output_labels):
        """get the label features... theta^T f_{BERT}(x_t)[y]
        
        Parameters
        ----------
        input_ids : torch.[cuda?].LongTensor
            input token ids, shape --> batch_size x max_seq_len
        attention_mask: torch.[cuda?].FloatTensor 
            mask that prevents attention over padded tokens. 
            contains values 1 for yes, 0 for not attending
            shape --> batch_size x max_seq_len
        classifier : nn.Module
            classifier from the Prediction Model to get the logits
        output_labels : torch.[cuda?].LongTensor
            the predicted output labels for the batch
        
        Returns
        -------
        torch.[cuda?].FloatTensor
            scores of shape batch_size x max_seq_len x num_labels
        """

        batch_size = input_ids.shape[0]
        # individual_bert_features = self._get_individual_bert_output(input_ids, attention_mask)
        # shape of individual_bert_features is batch_size x max_seq_len x bert_feature_dim (746)

        bert_seq_out, _, _ = self.bert(input_ids, attention_mask=attention_mask)
        # shape of bert_seq_out is batch_size x max_seq_len x bert_feature_dim (746)

        output_logits = classifier(bert_seq_out) 
        # shape of output_logits is  batch_size x max_seq_len x classification_labels

        output_log_probs = torch.log(nn.functional.softmax(output_logits, dim=-1) + 1e-9)

        #NOTE: the output label would not be the same for each sentence...
        output_log_probs_for_label = output_log_probs[torch.arange(batch_size), :, output_labels].unsqueeze(
            dim=-1)
        # shape of output_probs_for_label is  batch_size x max_seq_len x 1

        label_feats = self.label_features_to_label(output_log_probs_for_label) 
        # shape of label_feats is  batch_size x max_seq_len x num_labels

        return label_feats


    def _get_bow_features(self, input_ids, attention_mask, bow_classifier, output_labels):
        """get the label features... theta^T f_{bow}(embed(x_t))[y]
        
        Parameters
        ----------
        input_ids : torch.[cuda?].LongTensor
            input token ids, shape --> batch_size x max_seq_len
        attention_mask: torch.[cuda?].FloatTensor 
            mask that prevents attention over padded tokens. 
            contains values 1 for yes, 0 for not attending
            shape --> batch_size x max_seq_len
        bow_classifier : nn.Module
            classifier from the Prediction Model to get the logits
        output_labels : torch.[cuda?].LongTensor
            the predicted output labels for the batch
        
        Returns
        -------
        torch.[cuda?].FloatTensor
            scores of shape batch_size x max_seq_len x num_labels
        """

        batch_size = input_ids.shape[0]

        output = self.bert.embeddings.word_embeddings(input_ids)
        # output is batch_size x max_len x 768

        # remove the contributions from 0 positions as per attention mask
        output = torch.einsum("bld, bl->bld", output, attention_mask)

        output_logits = bow_classifier(output) 
        # shape of output_logits is  batch_size x max_seq_len x classification_labels

        output_probs = nn.functional.softmax(output_logits, dim=-1)

        #NOTE: the output label would not be the same for each sentence...
        output_probs_for_label = output_probs[torch.arange(batch_size), :, output_labels].unsqueeze(
            dim=-1)
        # shape of output_probs_for_label is  batch_size x max_seq_len x 1

        # add log-features instead
        output_log_probs_for_label = torch.log(output_probs_for_label + 1e-9)

        bow_feats = self.bow_features_to_label(output_log_probs_for_label) 
        # shape of label_feats is  batch_size x max_seq_len x num_labels

        return bow_feats
