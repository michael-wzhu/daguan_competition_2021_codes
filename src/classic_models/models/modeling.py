import torch
import torch.nn as nn

from src.classic_models.models.aggregator_layer import AggregatorLayer
from src.classic_models.models.classifier import Classifier
from src.classic_models.models.embedding_layer import EmbeddingLayer
from src.classic_models.models.encoders import TextCnnEncoder, BiLSTMEncoder
from src.classic_models.training.focal_loss import FocalLoss


class ClsModel(nn.Module):
    def __init__(self, args, label_list_level_1, label_list_level_2):
        super(ClsModel, self).__init__()
        self.args = args

        # TODO: 两层任务需要联合训练
        self.num_labels_level_1 = len(label_list_level_1)
        self.num_labels_level_2 = len(label_list_level_2)

        # embedding 层
        self.embeddings = EmbeddingLayer(args)

        # encoder 层
        if args.encoder == "textcnn":
            self.encoder = TextCnnEncoder(
                args,
            )
        elif args.encoder == "lstm":
            self.encoder = BiLSTMEncoder(
                args,
            )
        else:
            raise ValueError("un-supported encoder type: {}".format(args.encoder))

        # aggregator 层
        self.aggregator = AggregatorLayer(args)

        # 分类层
        self.classifier = Classifier(
            args,
            input_dim=args.hidden_dim,
            num_labels=self.num_labels_level_2,
        )

        # class weights
        self.class_weights_level_2 = None
        if self.args.class_weights_level_2:
            self.class_weights_level_2 = self.args.class_weights_level_2.split(",")
        else:
            self.class_weights_level_2 = [1] * self.num_labels_level_2
        self.class_weights_level_2 = [float(w) for w in self.class_weights_level_2]
        self.class_weights_level_2 = torch.FloatTensor(self.class_weights_level_2).to(self.args.device)

    def forward(self, input_ids=None,
                attention_mask=None,
                position_ids=None,
                label_ids_level_1=None,
                label_ids_level_2=None,
                **kwargs):

        # 经过embedding 层
        input_tensors = self.embeddings(input_ids)

        # encoding 层
        output_tensors = self.encoder(input_tensors)

        # aggregator层
        pooled_outputs = self.aggregator(output_tensors, mask=attention_mask)

        # 分类层： level 2
        logits_level_2 = self.classifier(pooled_outputs)  # [bsz, self.num_labels_level_2]

        outputs = (logits_level_2, )  # add hidden states and attention if they are here

        # 1. loss

        if label_ids_level_2 is not None:
            if self.args.use_focal_loss:
                loss_fct = FocalLoss(
                    self.num_labels_level_2,
                    alpha=self.class_weights_level_2,
                    gamma=self.args.focal_loss_gamma,
                    size_average=True
                )
            else:
                loss_fct = nn.CrossEntropyLoss()
            loss_level_2 = loss_fct(logits_level_2.view(-1, self.num_labels_level_2), label_ids_level_2.view(-1))

            outputs = (loss_level_2,) + outputs


        return outputs  # (loss), logits