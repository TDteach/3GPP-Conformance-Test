import torch
from sentence_transformers import SentenceTransformer, InputExample, models
from torch.utils.data import DataLoader
from typing import Iterable, Dict
from torch import Tensor

class CosineMimicLoss(torch.nn.Module):
    def __init__(self, model: SentenceTransformer, feature_dim: int):
        super(CosineMimicLoss, self).__init__()
        self.model = model
        self.feature_dim = feature_dim
        self.fn1 = torch.nn.Linear(in_features=self.feature_dim*2, out_features=self.feature_dim)
        self.fn2 = torch.nn.Linear(in_features=self.feature_dim, out_features=4)
        self.classifier = torch.nn.Sequential(self.fn1, self.fn2)
        self.train_flag = 0

    def set_train_classifier(self):
        self.train_flag = 1

    def set_train_all(self):
        self.train_flag = 2

    def set_predict(self):
        self.train_flag = 0

    def forward(self, sentence_features, labels=None):
        if self.train_flag == 1:
            return self._forward_mimic_cosine(sentence_features, labels)
        elif self.train_flag == 2:
            return self._forward_cross_entropy(sentence_features, labels)
        elif self.train_flag == 0:
            return self._forward(sentence_features, labels)
        else:
            raise NotImplementedError

    def _forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        a = model(sentence_features[0])

        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        assert len(reps) == 2

        cated_input: Tensor = torch.cat((reps[0], reps[1]), dim=1)
        outputs = self.classifier(cated_input)
        return outputs

    def _forward_cross_entropy(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        assert len(reps) == 2

        cated_input: Tensor = torch.cat((reps[0], reps[1]), dim=1)
        outputs = self.classifier(cated_input)

        loss = torch.nn.functional.cross_entropy(outputs, labels)
        return loss

    def _forward_mimic_cosine(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor):
        reps = [self.model(sentence_feature)['sentence_embedding'].data for sentence_feature in sentence_features]
        assert len(reps) == 2

        cated_input: Tensor = torch.cat((reps[0], reps[1]), dim=1)
        outputs = self.classifier(cated_input)
        mimic_value: Tensor = outputs[:, 1]

        cosine_value: Tensor = torch.cosine_similarity(reps[0], reps[1])

        loss = torch.nn.functional.mse_loss(mimic_value, cosine_value)
        return loss


if __name__ == '__main__':
    model = SentenceTransformer('distilbert-base-nli-mean-tokens')
    feature_dim = model.get_sentence_embedding_dimension()
    train_loss = CosineMimicLoss(model, feature_dim=feature_dim)

    train_examples = [InputExample(texts=['My first sentence', 'My second sentence'], label=0),
                      InputExample(texts=['Another pair', 'Unrelated sentence'], label=2)]
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

    train_loss.set_train_classifier()
    model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=0)
    torch.save(train_loss, 'haha.pt')

    train_examples = [InputExample(texts=['My first sentence', 'My second sentence'], label=0),
                      InputExample(texts=['Another pair', 'Unrelated sentence'], label=2)]
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

    train_loss = torch.load('haha.pt', map_location=model.device)
    train_loss.set_train_all()
    model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=0)

    train_loss = torch.load('haha.pt', map_location=model.device)
    train_loss.set_predict()
    model = SentenceTransformer('distilbert-base-nli-mean-tokens')
    train_loss.model = model
    from sentence_transformers.util import batch_to_device
    train_dataloader.collate_fn = model.smart_batching_collate
    for data in train_dataloader:
        sentence_batch, label = data
        output = train_loss(sentence_batch)
        print(output)
        exit(0)
