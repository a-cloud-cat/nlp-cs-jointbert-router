import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import *
from model.model import JointBERT
from model.data_process import load_dataset

class NLU_Dataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        intent = INTENT_LABEL_MAP[item['intent']]
        slot_labels = item.get('slot_labels', ['O'] * len(text))
        
        encoding = self.tokenizer(
            text,
            max_length=MAX_SEQ_LEN,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].flatten()
        attention_mask = encoding['attention_mask'].flatten()
        
        slot_label_ids = [0] * MAX_SEQ_LEN
        for i, label in enumerate(slot_labels):
            if i < MAX_SEQ_LEN - 2:
                slot_label_ids[i+1] = SLOT_LABEL_MAP.get(label, 0)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'intent': torch.tensor(intent, dtype=torch.long),
            'slot_labels': torch.tensor(slot_label_ids, dtype=torch.long)
        }

def train():
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
    
    train_data = load_dataset('train')
    val_data = load_dataset('val')
    test_data = load_dataset('test')
    
    train_dataset = NLU_Dataset(train_data, tokenizer)
    val_dataset = NLU_Dataset(val_data, tokenizer)
    test_dataset = NLU_Dataset(test_data, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = JointBERT()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    intent_loss_fn = nn.CrossEntropyLoss()
    
    best_val_acc = 0
    patience = 0
    
    train_log = []
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        train_intent_correct = 0
        train_total = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            intent = batch['intent'].to(device)
            slot_labels = batch['slot_labels'].to(device)
            
            optimizer.zero_grad()
            
            intent_logits, slot_loss = model(input_ids, attention_mask, slot_labels=slot_labels)
            intent_loss = intent_loss_fn(intent_logits, intent)
            
            loss = intent_loss + slot_loss
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * input_ids.size(0)
            intent_preds = torch.argmax(intent_logits, dim=1)
            train_intent_correct += (intent_preds == intent).sum().item()
            train_total += input_ids.size(0)
        
        train_loss = train_loss / train_total
        train_acc = train_intent_correct / train_total
        
        model.eval()
        val_loss = 0
        val_intent_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                intent = batch['intent'].to(device)
                slot_labels = batch['slot_labels'].to(device)
                
                intent_logits, slot_loss = model(input_ids, attention_mask, slot_labels=slot_labels)
                intent_loss = intent_loss_fn(intent_logits, intent)
                
                loss = intent_loss + slot_loss
                val_loss += loss.item() * input_ids.size(0)
                
                intent_preds = torch.argmax(intent_logits, dim=1)
                val_intent_correct += (intent_preds == intent).sum().item()
                val_total += input_ids.size(0)
        
        val_loss = val_loss / val_total
        val_acc = val_intent_correct / val_total
        
        train_log.append({
            'epoch': epoch+1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        })
        
        print(f'Epoch {epoch+1}/{EPOCHS}')
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')
        print('---')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience = 0
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, 'best_model.pt'))
        else:
            patience += 1
            if patience >= EARLY_STOP_PATIENCE:
                print(f'Early stopping at epoch {epoch+1}')
                break
    
    model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, 'best_model.pt')))
    model.eval()
    
    test_intent_correct = 0
    test_total = 0
    all_slot_preds = []
    all_slot_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            intent = batch['intent'].to(device)
            slot_labels = batch['slot_labels'].to(device)
            
            intent_logits, slot_preds = model(input_ids, attention_mask)
            
            intent_preds = torch.argmax(intent_logits, dim=1)
            test_intent_correct += (intent_preds == intent).sum().item()
            test_total += input_ids.size(0)
            
            for i in range(len(slot_preds)):
                valid_preds = slot_preds[i]
                valid_labels = []
                for j in range(len(valid_preds)):
                    pos = j + 1
                    if pos < len(slot_labels[i]):
                        valid_labels.append(slot_labels[i][pos].item())
                
                all_slot_preds.extend(valid_preds)
                all_slot_labels.extend(valid_labels)
    
    test_acc = test_intent_correct / test_total
    slot_f1 = f1_score(all_slot_labels, all_slot_preds, average='macro')
    
    print(f'Test Intent Accuracy: {test_acc:.4f}')
    print(f'Test Slot F1 Score: {slot_f1:.4f}')
    
    with open(os.path.join(OUTPUT_DIR, 'evaluation.txt'), 'w', encoding='utf-8') as f:
        f.write(f'Test Intent Accuracy: {test_acc:.4f}\n')
        f.write(f'Test Slot F1 Score: {slot_f1:.4f}\n')
    
    with open(os.path.join(OUTPUT_DIR, 'train_log.txt'), 'w', encoding='utf-8') as f:
        for log in train_log:
            f.write(f"Epoch {log['epoch']}: Train Loss={log['train_loss']:.4f}, Train Acc={log['train_acc']:.4f}, Val Loss={log['val_loss']:.4f}, Val Acc={log['val_acc']:.4f}\n")

if __name__ == '__main__':
    train()
