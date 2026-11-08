"""Transformer-CRF Named Entity Recognition Architecture (Objective 2, Outcome 1).

Implements a contextual Sequence Encoder paired with a Linear-Chain Conditional
Random Field (CRF) layer for sequence tagging on noisy code-mixed clinical text.
Supports Viterbi decoding for globally optimal IOB tag sequence extraction.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional


TAG_LIST = [
    "O",
    "B-SYMPTOM", "I-SYMPTOM",
    "B-DURATION", "I-DURATION",
    "B-SEVERITY", "I-SEVERITY",
    "B-SITE", "I-SITE"
]
TAG2ID = {tag: i for i, tag in enumerate(TAG_LIST)}
ID2TAG = {i: tag for tag, i in TAG2ID.items()}
START_TAG = "<START>"
STOP_TAG = "<STOP>"
NUM_TAGS = len(TAG_LIST)


def log_sum_exp(vec: torch.Tensor) -> torch.Tensor:
    max_score = vec.max(-1, keepdim=True)[0]
    return max_score.squeeze(-1) + torch.log(torch.sum(torch.exp(vec - max_score), -1))


class LinearCRF(nn.Module):
    """Linear Chain CRF for sequence tagging with Viterbi decoding."""

    def __init__(self, num_tags: int):
        super().__init__()
        self.num_tags = num_tags
        # transitions[i, j] = score of transitioning from tag j to tag i
        self.transitions = nn.Parameter(torch.randn(num_tags + 2, num_tags + 2))
        self.start_idx = num_tags
        self.stop_idx = num_tags + 1

        # Enforce valid transition constraints:
        # Never transition to START, and never transition from STOP
        with torch.no_grad():
            self.transitions[:, self.start_idx] = -10000.0
            self.transitions[self.stop_idx, :] = -10000.0
            # Discourage invalid IOB transitions (e.g., I-SYMPTOM without B-SYMPTOM or I-SYMPTOM)
            for i, tag_to in enumerate(TAG_LIST):
                if tag_to.startswith("I-"):
                    prefix = tag_to[2:]
                    for j, tag_from in enumerate(TAG_LIST):
                        if not (tag_from == f"B-{prefix}" or tag_from == f"I-{prefix}"):
                            self.transitions[i, j] = -10.0

    def _forward_alg(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute the log partition function Z(x) using the forward algorithm."""
        # emissions: [seq_len, batch_size, num_tags]
        seq_len, batch_size, _ = emissions.shape
        init_alphas = torch.full((batch_size, self.num_tags), -10000.0, device=emissions.device)
        # Transition from START to all initial tags
        init_alphas += self.transitions[:self.num_tags, self.start_idx].unsqueeze(0)
        forward_var = init_alphas

        for t in range(seq_len):
            emit_t = emissions[t]  # [batch_size, num_tags]
            mask_t = mask[t].unsqueeze(1)  # [batch_size, 1]

            # Broadcast forward_var: [batch, 1, tags] + transitions: [tags, tags] + emit: [batch, tags, 1]
            # transition from j (prev) to i (curr)
            trans_score = self.transitions[:self.num_tags, :self.num_tags].unsqueeze(0)  # [1, num_tags, num_tags]
            broadcast_var = forward_var.unsqueeze(1)  # [batch, 1, num_tags]
            emit_score = emit_t.unsqueeze(2)  # [batch, num_tags, 1]

            next_tag_var = broadcast_var + trans_score + emit_score  # [batch, num_tags, num_tags]
            next_forward = log_sum_exp(next_tag_var)  # [batch, num_tags]

            forward_var = torch.where(mask_t, next_forward, forward_var)

        # Transition to STOP
        terminal_vars = forward_var + self.transitions[self.stop_idx, :self.num_tags].unsqueeze(0)
        alpha = log_sum_exp(terminal_vars)
        return alpha

    def _score_sentence(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute score of ground-truth tag sequence."""
        seq_len, batch_size, _ = emissions.shape
        score = torch.zeros(batch_size, device=emissions.device)

        # START transition
        first_tags = tags[0]
        score += self.transitions[first_tags, self.start_idx]
        score += emissions[0, torch.arange(batch_size), first_tags]

        for t in range(1, seq_len):
            mask_t = mask[t]
            curr_tags = tags[t]
            prev_tags = tags[t - 1]

            step_score = self.transitions[curr_tags, prev_tags] + emissions[t, torch.arange(batch_size), curr_tags]
            score = torch.where(mask_t, score + step_score, score)

        # STOP transition: find the last valid tag for each sample
        last_indices = mask.sum(dim=0).long() - 1
        last_tags = tags.gather(0, last_indices.unsqueeze(0)).squeeze(0)
        score += self.transitions[self.stop_idx, last_tags]
        return score

    def forward(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute negative log-likelihood loss: log Z(x) - score(x, y)."""
        forward_score = self._forward_alg(emissions, mask)
        gold_score = self._score_sentence(emissions, tags, mask)
        return torch.mean(forward_score - gold_score)

    def decode(self, emissions: torch.Tensor, mask: torch.Tensor) -> List[List[int]]:
        """Viterbi decoding for maximum-likelihood tag sequence."""
        seq_len, batch_size, _ = emissions.shape
        best_paths = []

        for b in range(batch_size):
            b_mask = mask[:, b]
            b_len = int(b_mask.sum().item())
            if b_len == 0:
                best_paths.append([])
                continue

            b_emissions = emissions[:b_len, b]  # [len, num_tags]

            # Initialize Viterbi variables in log space
            viterbi_vars = self.transitions[:self.num_tags, self.start_idx] + b_emissions[0]
            backpointers = []

            for t in range(1, b_len):
                # prev tag -> curr tag
                trans = self.transitions[:self.num_tags, :self.num_tags]  # [curr, prev]
                broadcast_viterbi = viterbi_vars.unsqueeze(0)  # [1, prev]
                next_tag_var = broadcast_viterbi + trans + b_emissions[t].unsqueeze(1)  # [curr, prev]

                best_prev = torch.argmax(next_tag_var, dim=1)  # [curr]
                viterbi_vars = next_tag_var.gather(1, best_prev.unsqueeze(1)).squeeze(1)
                backpointers.append(best_prev.tolist())

            # Transition to STOP
            terminal_var = viterbi_vars + self.transitions[self.stop_idx, :self.num_tags]
            best_last_tag = torch.argmax(terminal_var).item()

            # Follow backpointers
            best_path = [best_last_tag]
            for bp in reversed(backpointers):
                best_last_tag = bp[best_last_tag]
                best_path.append(best_last_tag)

            best_path.reverse()
            best_paths.append(best_path)

        return best_paths


class MultilingualTransformerCRF(nn.Module):
    """Contextual Multilingual Sequence Tagger with Linear CRF Output."""

    def __init__(self, vocab_size: int, char_vocab_size: int, embed_dim: int = 128,
                 hidden_dim: int = 128, num_tags: int = NUM_TAGS, dropout: float = 0.25):
        super().__init__()
        self.word_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.char_embed = nn.Embedding(char_vocab_size, 32, padding_idx=0)
        self.char_cnn = nn.Conv1d(32, 64, kernel_size=3, padding=1)

        # Bidirectional Contextual Sequence Modeling
        self.encoder = nn.LSTM(
            embed_dim + 64,
            hidden_dim // 2,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if dropout > 0 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_dim, num_tags)
        self.crf = LinearCRF(num_tags)

    def get_emissions(self, words: torch.Tensor, chars: torch.Tensor) -> torch.Tensor:
        # words: [batch, seq_len], chars: [batch, seq_len, char_len]
        w_emb = self.word_embed(words)  # [batch, seq_len, embed_dim]

        batch_size, seq_len, char_len = chars.shape
        c_flat = chars.view(-1, char_len)  # [batch * seq_len, char_len]
        c_emb = self.char_embed(c_flat).transpose(1, 2)  # [B*S, 32, char_len]
        c_conv = F.relu(self.char_cnn(c_emb))  # [B*S, 64, char_len]
        c_feat = torch.max(c_conv, dim=2)[0].view(batch_size, seq_len, 64)

        combined = torch.cat([w_emb, c_feat], dim=-1)
        combined = self.dropout(combined)

        context, _ = self.encoder(combined)  # [batch, seq_len, hidden_dim]
        context = self.dropout(context)
        emissions = self.linear(context)  # [batch, seq_len, num_tags]
        return emissions.transpose(0, 1)  # [seq_len, batch, num_tags]

    def forward(self, words: torch.Tensor, chars: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        emissions = self.get_emissions(words, chars)
        mask_t = mask.transpose(0, 1)
        tags_t = tags.transpose(0, 1)
        return self.crf(emissions, tags_t, mask_t)

    def decode(self, words: torch.Tensor, chars: torch.Tensor, mask: torch.Tensor) -> List[List[int]]:
        emissions = self.get_emissions(words, chars)
        mask_t = mask.transpose(0, 1)
        return self.crf.decode(emissions, mask_t)
