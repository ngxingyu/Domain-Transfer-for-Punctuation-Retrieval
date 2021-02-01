import torch
from torch import Tensor, LongTensor, nn
from torch.nn.functional import log_softmax
from typing import Optional, List

class LinearChainCRF(torch.nn.Module):
    def __init__(
        self, 
        num_labels: int = 10, 
        pad_idx: Optional[int] = None,
        reduction: Optional[str] = "mean",
    ) -> None:
        """
        :param num_labels: number of labels
        :param pad_idx: padding index. default None
        :return None
        """
        if num_labels < 1: raise ValueError("invalid number of labels: {0}".format(num_labels))
        super().__init__()
        self.num_labels = num_labels
        self.trans_matrix = nn.Parameter(torch.empty(num_labels, num_labels))
        self.start_trans = nn.Parameter(torch.empty(num_labels))
        self.end_trans = nn.Parameter(torch.empty(num_labels))
        self.reduction = reduction
        self._initialize_parameters(pad_idx)
    def forward(
        self, logits: Tensor, labels: Tensor, loss_mask=None) -> Tensor:
        """
        :param logits: hidden matrix (batch_size, seq_len, num_labels)
        :param labels: answer labels of each sequence
                       in mini batch (batch_size, seq_len)
        :param mask: mask tensor of each sequence #attention_mask
                     in mini batch (batch_size, seq_len)
        :param reduction: Specifies  the reduction to apply to the output:
                ``none|sum|mean|token_mean``. ``none``: no reduction will be applied.
                ``sum``: the output will be summed over batches. ``mean``: the output will be
                averaged over batches. ``token_mean``: the output will be averaged over tokens
        :return: The log-likelihood (batch_size,) if reduction==none,  else ()
        """
        self._validate(logits, labels=labels, mask=loss_mask)
        if self.reduction not in ('none', 'sum', 'mean', 'token_mean'):
            raise ValueError(f'invalid reduction: {self.reduction}')
        if loss_mask is None:
            loss_mask = torch.ones_like(labels, dtype=torch.uint8)
        logits =log_softmax(logits,-1)
        log_numerator = self._compute_numerator_log_likelihood(logits, labels, loss_mask) #score
        log_denominator = self._compute_denominator_log_likelihood(logits, loss_mask) #partition
        llh=log_numerator - log_denominator
        if self.reduction == 'none':
            return -llh
        if self.reduction == 'sum':
            return -llh.sum()
        if self.reduction == 'mean':
            return -llh.mean()
        assert self.reduction == 'token_mean'
        return -llh.sum() / loss_mask.type_as(logits).sum()

    def _validate(
            self,
            logits: Tensor,
            labels: Optional[Tensor] = None,
            mask: Optional[Tensor] = None) -> None:
        if logits.dim() != 3:
            raise ValueError(f'logits must have dimension of 3, got {logits.dim()}')
        if logits.size(2) != self.num_labels:
            raise ValueError(
                f'expected last dimension of logits is {self.num_labels}, '
                f'got {logits.size(2)}')
        if labels is not None:
            if logits.shape[:2] != labels.shape:
                raise ValueError(
                    'the first two dimensions of logits and labels must match, '
                    f'got {tuple(logits.shape[:2])} and {tuple(labels.shape)}')
        if mask is not None:
            if logits.shape[:2] != mask.shape:
                raise ValueError(
                    'the first two dimensions of h and mask must match, '
                    f'got {tuple(logits.shape[:2])} and {tuple(mask.shape)}')
    def decode(self, logits: Tensor,
               mask: Optional[Tensor] = None) -> LongTensor:
        self._validate(logits, mask=mask)

        if mask is None:
            mask = logits.new_ones(logits.shape[:2], dtype=torch.bool)
        return self._viterbi_decode(logits,mask)

    def _viterbi_decode(self, logits: Tensor, mask: Tensor) -> Tensor:
        """
        decode labels using viterbi algorithm
        :param logits: hidden matrix (batch_size, seq_len, num_labels)
        :param mask: mask tensor of each sequence
                     in mini batch (batch_size, seq_len)
        :return: labels of each sequence in mini batch
        """
        assert logits.dim() == 3 and mask.dim() == 2
        assert logits.shape[:2] == mask.shape
        assert logits.size(2) == self.num_labels
        assert mask[0].all()

        batch_size, seq_len, _ = logits.shape
        # prepare the sequence lengths in each sequence
        seq_lens = mask.sum(dim=1)
        # In mini batch, prepare the score
        # from the start sequence to the first label
        score = [self.start_trans.data + logits[:, 0]]
        path = []
        for t in range(1, seq_len):
            # extract the score of previous sequence
            # (batch_size, num_labels, 1)
            previous_score = score[t - 1].view(batch_size, -1, 1)
            # extract the score of hidden matrix of sequence
            # (batch_size, 1, num_labels)
            logits_t = logits[:, t].view(batch_size, 1, -1)
            # extract the score in transition
            # from label of t-1 sequence to label of sequence of t
            # self.trans_matrix has the score of the transition
            # from sequence A to sequence B
            # (batch_size, num_labels, num_labels)
            score_t = previous_score + self.trans_matrix + logits_t
            # keep the maximum value
            # and point where maximum value of each sequence
            # (batch_size, num_labels)
            best_score, best_path = score_t.max(1)
            score.append(best_score)
            path.append(best_path)
        # predict labels of mini batch
        best_paths = [
            self._viterbi_compute_best_path(i, seq_lens, score, path)
            for i in range(batch_size)
        ]
        return Tensor(best_paths)

    def _viterbi_compute_best_path(
        self,
        batch_idx: int,
        seq_lens: Tensor,
        score: List[Tensor],
        path: List[Tensor],
    ) -> List[int]:
        """
        return labels using viterbi algorithm
        :param batch_idx: index of batch
        :param seq_lens: sequence lengths in mini batch (batch_size)
        :param score: transition scores of length max sequence size
                      in mini batch [(batch_size, num_labels)]
        :param path: transition paths of length max sequence size
                     in mini batch [(batch_size, num_labels)]
        :return: labels of batch_idx-th sequence
        """
        seq_end_idx = seq_lens[batch_idx] - 1
        # extract label of end sequence
        _, best_last_label = (score[seq_end_idx][batch_idx] + self.end_trans).max(0)
        best_labels = [int(best_last_label)]
        # predict labels from back using viterbi algorithm
        for p in reversed(path[:seq_end_idx]):
            best_last_label = p[batch_idx][best_labels[0]]
            best_labels.insert(0, int(best_last_label))
        return best_labels

    def _compute_denominator_log_likelihood(self, logits: Tensor, mask: Tensor):
        """
        compute the denominator term for the log-likelihood
        compute the partition function in log-space using the forward-algorithm.
        :param logits: hidden matrix (batch_size, seq_len, num_labels)
        :param mask: mask tensor of each sequence
                     in mini batch (batch_size, seq_len)
        :return: The score of denominator term for the log-likelihood
        """
        device = logits.device
        batch_size, seq_len, _ = logits.size()
        # (num_labels, num_labels) -> (1, num_labels, num_labels)
        trans = self.trans_matrix.unsqueeze(0)
        # add the score from beginning to each label
        # and the first score of each label
        score = self.start_trans + logits[:, 0]
        # iterate through processing for the number of words in the mini batch
        for t in range(1, seq_len):
            # (batch_size, self.num_labels, 1)
            before_score = score.unsqueeze(2)
            # prepare t-th mask of sequences in each sequence
            # (batch_size, 1)
            mask_t = mask[:, t].unsqueeze(1)
            mask_t = mask_t.to(device)
            # prepare the transition probability of the t-th sequence label
            # in each sequence
            # (batch_size, 1, num_labels)
            logits_t = logits[:, t].unsqueeze(1)
            # calculate t-th scores in each sequence
            # (batch_size, num_labels)
            score_t = before_score + logits_t + trans
            score_t = torch.logsumexp(score_t, 1)
            # update scores
            # (batch_size, num_labels)
            score = torch.where(mask_t, score_t, score)
        # add the end score of each label
        score += self.end_trans
        # return the log likely food of all data in mini batch
        return torch.logsumexp(score, 1)

    def _compute_numerator_log_likelihood(self, logits: Tensor, y: Tensor, mask: Tensor) -> Tensor:
        """
        compute the numerator term for the log-likelihood
        :param logits: hidden matrix (batch_size, seq_len, num_labels)
        :param y: answer labels of each sequence
                  in mini batch (batch_size, seq_len)
        :param mask: mask tensor of each sequence
                     in mini batch (batch_size, seq_len)
        :return: The score of numerator term for the log-likelihood
        """
        batch_size, seq_len, _ = logits.size()
        logits_unsqueezed = logits.unsqueeze(-1)
        trans = self.trans_matrix.unsqueeze(-1)
        arange_b = torch.arange(batch_size)
        # extract first vector of sequences in mini batch
        last_mask_index = mask.sum(1) - 1
        calc_range = seq_len - 1 #should calc_range be last_mask_index? No since parallel faster
        # calc_range = last_mask_index
        score = self.start_trans[y[:, 0]] + sum(
            [self._calc_trans_score_for_num_llh(
                logits_unsqueezed, y, trans, mask, t, arange_b
            ) for t in range(calc_range)])
        # extract end label number of each sequence in mini batch
        # (batch_size)
        last_labels = y[arange_b,last_mask_index]
        each_last_score = logits[arange_b, -1, last_labels].squeeze(-1) * mask[:, -1]
        # Add the score of the sequences of the maximum length in mini batch
        # Add the scores from the last tag of each sequence to EOS
        score += each_last_score + self.end_trans[last_labels]
        return score
    def _calc_trans_score_for_num_llh(
        self,
        logits: Tensor,
        y: Tensor,
        trans: Tensor,
        mask: Tensor,
        t: int,
        arange_b: Tensor,
    ) -> Tensor:
        """
        calculate transition score for computing numerator llh
        :param logits: hidden matrix (batch_size, seq_len, num_labels)
        :param y: answer labels of each sequence
                  in mini batch (batch_size, seq_len)
        :param trans: transition score
        :param mask: mask tensor of each sequence
                     in mini batch (batch_size, seq_len)
        :paramt t: index of hidden, transition, and mask matrixex
        :param arange_b: this param is seted torch.arange(batch_size)
        :param batch_size: batch size of this calculation
        """
        device = logits.device
        mask_t = mask[:, t]
        mask_t = mask_t.to(device)
        mask_t1 = mask[:, t + 1]
        mask_t1 = mask_t1.to(device)
        # extract the score of t label
        # (batch_size)
        logits_t = logits[arange_b, t, y[:, t]].squeeze(1)  ##Changed from t to t
        # extract the transition score from t-th label to t+1 label
        # (batch_size)
        trans_t = trans[y[:, t], y[:, t + 1]].squeeze(1)
        # add the score of t+1 and the transition score
        # (batch_size)
        return logits_t * mask_t + trans_t * mask_t1
    def _initialize_parameters(self, pad_idx: Optional[int]) -> None:
        """
        initialize transition parameters
        :param: pad_idx: if not None, additional initialize
        :return: None
        """
        nn.init.uniform_(self.trans_matrix, -0.1, 0.1)
        nn.init.uniform_(self.start_trans, -0.1, 0.1)
        nn.init.uniform_(self.end_trans, -0.1, 0.1)
        punct=torch.arange(1,self.trans_matrix.shape[0])
        # inaccurate since consecutive punctuation is possible ("Hello! Bye!")
        # with torch.no_grad():
        #   self.trans_matrix[punct,punct]=-100
        if pad_idx is not None:
            self.start_trans[pad_idx] = -10000.0
            self.trans_matrix[pad_idx, :] = -10000.0
            self.trans_matrix[:, pad_idx] = -10000.0
            self.trans_matrix[pad_idx, pad_idx] = 0.0