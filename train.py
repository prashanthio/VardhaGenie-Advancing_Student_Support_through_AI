import numpy as np
import random
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from nltk_utils import bag_of_words, tokenize, stem
from model import NeuralNet

with open('C:\ChatBot\chatbot-deployment-main\chatbot-deployment-main\intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []
# loop through each sentence in our intents patterns

for intent in intents['intents']:
    tag = intent['tag']
    # add to tag list
    tags.append(tag)
    for pattern in intent['patterns']:
        # tokenize each word in the sentence
        w = tokenize(pattern)
        # add to our words list
        all_words.extend(w)
        # add to xy pair
        xy.append((w, tag))

# stem and lower each word
ignore_words = ['?', '.', '!']
all_words = [stem(w) for w in all_words if w not in ignore_words]
# remove duplicates and sort
all_words = sorted(set(all_words))
tags = sorted(set(tags))

print(len(xy), "patterns")
print(len(tags), "tags:", tags)
print(len(all_words), "unique stemmed words:", all_words)

# create training data
X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    # X: bag of words for each pattern_sentence
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

# Hyper-parameters 
num_epochs = 1000
batch_size = 12
learning_rate = 0.01
input_size = len(X_train[0])
hidden_size = 10
output_size = len(tags)
print("input_size is: {0} and output_size is: {1}".format(input_size, output_size))

# class ChatDataset(Dataset):

#     def __init__(self):
#         self.n_samples = len(X_train)
#         self.x_data = X_train
#         self.y_data = y_train

#     # support indexing such that dataset[i] can be used to get i-th sample
#     def __getitem__(self, index):
#         return self.x_data[index], self.y_data[index]

#     # we can call len(dataset) to return the size
#     def __len__(self):
#         return self.n_samples

# dataset = ChatDataset()
# train_loader = DataLoader(dataset=dataset,
#                           batch_size=batch_size,
#                           shuffle=True,
#                           num_workers=0)
#-----------------------------------------------------------------------
class TestChatDataset(Dataset):
    def __init__(self, X, y):
        self.n_samples = len(X)
        self.x_data = torch.tensor(X, dtype=torch.float32)  # Convert to tensor
        self.y_data = torch.tensor(y, dtype=torch.long)  # Convert to tensor

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train)

train_dataset = TestChatDataset(X_train, y_train)
train_loader = DataLoader(dataset=train_dataset,
                           batch_size=batch_size,
                          shuffle=True)
#-------------------------------------------------------------------
# Create DataLoader for testing
test_dataset = TestChatDataset(X_test, y_test)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NeuralNet(input_size, hidden_size, output_size).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Lists to store accuracy for each epoch
train_accuracy_list = []
test_accuracy_list = []

# Train the model
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        
        # Forward pass
        outputs = model(words)
        
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    #------------TRIAL--------------------------------
    model.eval()
    all_labels_train = []
    all_predictions_train = []
    with torch.no_grad():
        for (words, labels) in train_loader:
            words = words.to(device)
            labels = labels.to(dtype=torch.long).to(device)
            
            outputs = model(words)
            _, predicted = torch.max(outputs.data, 1)
            
            all_labels_train.extend(labels.cpu().numpy())
            all_predictions_train.extend(predicted.cpu().numpy())
    
    train_accuracy = accuracy_score(all_labels_train, all_predictions_train)
    train_accuracy_list.append(train_accuracy)
    # Calculate test accuracy for the epoch
    all_labels_test = []
    all_predictions_test = []
    with torch.no_grad():
        for (words, labels) in test_loader:
            words = words.to(device)
            labels = labels.to(dtype=torch.long).to(device)
            
            outputs = model(words)
            _, predicted = torch.max(outputs.data, 1)
            
            all_labels_test.extend(labels.cpu().numpy())
            all_predictions_test.extend(predicted.cpu().numpy())
    
    test_accuracy = accuracy_score(all_labels_test, all_predictions_test)
    test_accuracy_list.append(test_accuracy)
    
    # Print and display accuracy for every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Train Accuracy: {train_accuracy * 100:.2f}%, Test Accuracy: {test_accuracy * 100:.2f}%')

print(f'Final training accuracy: {train_accuracy * 100:.2f}%, Final test accuracy: {test_accuracy * 100:.2f}%')

#------------------------------over--------------------------------------------------------
        
    #if (epoch+1) % 100 == 0:
        #print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


print(f'final loss: {loss.item():.4f}')

data = {
"model_state": model.state_dict(),
"input_size": input_size,
"hidden_size": hidden_size,
"output_size": output_size,
"all_words": all_words,
"tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')




import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

 # Split the data into training and test sets
from sklearn.model_selection import train_test_split

# X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# # Create a test dataset
# class TestChatDataset(Dataset):
#     def __init__(self, X, y):
#         self.n_samples = len(X)
#         self.x_data = X
#         self.y_data = y

#     def __getitem__(self, index):
#         return self.x_data[index], self.y_data[index]

#     def __len__(self):
#         return self.n_samples

# test_dataset = TestChatDataset(X_test, y_test)
# test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)


# # Evaluate on the test set
# model.eval()
# all_labels = []
# all_predictions = []

# with torch.no_grad():
#     for (words, labels) in test_loader:
#         words = words.to(device)
#         labels = labels.to(dtype=torch.long).to(device)
        
#         outputs = model(words)
#         _, predicted = torch.max(outputs.data, 1)
        
#         all_labels.extend(labels.cpu().numpy())
#         all_predictions.extend(predicted.cpu().numpy())

# # Calculate accuracy using scikit-learn's accuracy_score
# accuracy = accuracy_score(all_labels, all_predictions)
# print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Create a confusion matrix
conf_matrix = confusion_matrix(all_labels_train, all_predictions_train)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=tags, yticklabels=tags)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

