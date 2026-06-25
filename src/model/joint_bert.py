import os
import sys
import torch
import torch.nn as nn
from transformers import BertModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import NUM_INTENT_CLASSES, NUM_SLOT_CLASSES, BERT_MODEL_NAME, DROPOUT_RATE, MAX_SEQ_LEN

class CRFLayer(nn.Module):
    def __init__(self, num_tags):
        super(CRFLayer, self).__init__()
        self.num_tags = num_tags
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.end_transitions = nn.Parameter(torch.randn(num_tags))
    
    def forward(self, emissions, tags=None, mask=None):
        if mask is None:
            mask = torch.ones(emissions.shape[:2], dtype=torch.bool, device=emissions.device)
        
        if tags is not None:
            log_likelihood = self._compute_log_likelihood(emissions, tags, mask)
            return log_likelihood
        else:
            path = self._viterbi_decode(emissions, mask)
            return path
    
    def _compute_log_likelihood(self, emissions, tags, mask):
        batch_size, seq_len, num_tags = emissions.shape
        
        mask = mask.float()
        log_likelihood = torch.zeros(batch_size, device=emissions.device)
        
        for i in range(batch_size):
            seq_len_i = int(mask[i].sum())
            emissions_i = emissions[i, :seq_len_i]
            tags_i = tags[i, :seq_len_i]
            
            score = self.start_transitions[tags_i[0]] + emissions_i[0, tags_i[0]]
            
            for j in range(1, seq_len_i):
                score += self.transitions[tags_i[j-1], tags_i[j]] + emissions_i[j, tags_i[j]]
            
            score += self.end_transitions[tags_i[-1]]
            log_likelihood[i] = score
        
        return log_likelihood
    
    def _viterbi_decode(self, emissions, mask):
        batch_size, seq_len, num_tags = emissions.shape
        
        paths = []
        for i in range(batch_size):
            seq_len_i = int(mask[i].sum())
            emissions_i = emissions[i, :seq_len_i]
            
            dp = torch.zeros(seq_len_i, num_tags, device=emissions.device)
            backpointer = torch.zeros(seq_len_i, num_tags, dtype=torch.long, device=emissions.device)
            
            dp[0] = self.start_transitions + emissions_i[0]
            
            for j in range(1, seq_len_i):
                for k in range(num_tags):
                    max_val, max_idx = torch.max(dp[j-1] + self.transitions[:, k], dim=0)
                    dp[j, k] = max_val + emissions_i[j, k]
                    backpointer[j, k] = max_idx
            
            max_val, max_idx = torch.max(dp[-1] + self.end_transitions, dim=0)
            path = [max_idx.item()]
            
            for j in range(seq_len_i - 1, 0, -1):
                path.append(backpointer[j, path[-1]].item())
            
            paths.append(path[::-1])
        
        return paths

class JointBERT(nn.Module):
    def __init__(self):
        super(JointBERT, self).__init__()
        local_model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..', 'models', 'bert-base-chinese')
        if os.path.exists(local_model_path):
            self.bert = BertModel.from_pretrained(local_model_path, local_files_only=True)
            print(f'Loaded BertModel from local path: {local_model_path}')
        else:
            self.bert = BertModel.from_pretrained(BERT_MODEL_NAME)
            print(f'Loaded BertModel from HuggingFace: {BERT_MODEL_NAME}')
        self.dropout = nn.Dropout(DROPOUT_RATE)
        self.intent_classifier = nn.Linear(self.bert.config.hidden_size, NUM_INTENT_CLASSES)
        self.slot_classifier = nn.Linear(self.bert.config.hidden_size, NUM_SLOT_CLASSES)
        self.crf = CRFLayer(NUM_SLOT_CLASSES)
    
    def forward(self, input_ids, attention_mask, token_type_ids=None, slot_labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        sequence_output = outputs[0]
        pooled_output = outputs[1]
        
        sequence_output = self.dropout(sequence_output)
        pooled_output = self.dropout(pooled_output)
        
        intent_logits = self.intent_classifier(pooled_output)
        slot_logits = self.slot_classifier(sequence_output)
        
        if slot_labels is not None:
            crf_loss = -self.crf(slot_logits, slot_labels, attention_mask.bool()).mean()
            return intent_logits, crf_loss
        else:
            slot_preds = self.crf(slot_logits, mask=attention_mask.bool())
            return intent_logits, slot_preds

if __name__ == '__main__':
    model = JointBERT()
    print(model)
