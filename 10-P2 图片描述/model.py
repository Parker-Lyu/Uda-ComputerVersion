import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.hidden_size=hidden_size
        self.vocab_size=vocab_size
        
        self.embed = nn.Embedding(vocab_size,embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.hidden = (torch.zeros(1,1,hidden_size),torch.zeros(1,1,hidden_size))
        
    def forward(self, features, captions):
        # todo 这里为什么舍弃了最后一个单词
#         print('features.size()',features.size())
#         print('captions.size()',captions.size())
        cap_embedding = self.embed(captions[:,:-1])
#         print('cap_embedding.size()',cap_embedding.size())
        
        # 把captions 和 features 合并
        embeddings = torch.cat((features.unsqueeze(1), cap_embedding),dim=1)
#         print('embeddings.size()',embeddings.size())
        
        outputs, _ = self.lstm(embeddings)
#         print('outputs.size()',outputs.size())
        scores = self.fc(outputs)
        return scores
        



    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        pass