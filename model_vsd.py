from abc import ABCMeta, abstractmethod
from collections import OrderedDict
import numpy as np

import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import models

def get_backbone_output_dim(backbone_name: str):
    if backbone_name == "argus":
        return 2208
    elif backbone_name == "argus2":
        return 2048
    elif backbone_name.startswith("resnet"):
        return 2048
    else:
        raise ValueError(f"Unsupported backbone name: {backbone_name}")


def create_pretrained_multitaskembedding_vbs_model(params, dataset_metadata):

    model = MultitaskEmbeddingClassificationNet(by_task_embedding_size=dataset_metadata.by_task_embedding_size,
                                                by_task_output_size=dataset_metadata.by_task_output_size,
                                                dropout=0,
                                                temperature=0.05,
                                                backbone_name=params.backbone,
                                                base_requires_grad=False)
    if params.checkpoint != None:
        trainer_state_dict = torch.load(params.checkpoint, map_location=params.device)
        if 'state_dict' in trainer_state_dict:
            print('Loaded checkpint from: {}'.format(params.checkpoint))
            model.load_state_dict(trainer_state_dict["state_dict"])
        elif 'model' in trainer_state_dict:
            print('Loaded checkpint from: {}'.format(params.checkpoint))
            model.load_state_dict(trainer_state_dict["model"])
        else:
            raise ValueError

    if params.device:
        model = model.to(params.device)
    return model


def create_pretrained_multitaskembedding_vbs_model_embedding_base(dataset_metadata, params, task_name):
    model = create_pretrained_multitaskembedding_vbs_model(params, dataset_metadata)
    model = model.get_specific_task_embedding_base(task_name=task_name)
    return model


def create_pretrained_multitaskembedding_vbs_model_classification_net(dataset_metadata, params, task_name):
    model = create_pretrained_multitaskembedding_vbs_model(params, dataset_metadata)
    model = model.get_specific_task_classification_net(task_name=task_name)
    return model


def create_resnet50_module(pretrained=False, include_fc_top=True, requires_grad=True):
    model = models.resnet50(pretrained=pretrained)
    for param in model.parameters():
        param.requires_grad = requires_grad

    return model if include_fc_top else create_sequential_model_without_top(model)


def create_sequential_model_without_top(model, num_top_layers=1):
    """
    Creates and returns a model that is the same as the input model with the number of top layers removed. The model
    share their state.
    :param model: input pytorch module.
    :param num_top_layers: number of top layers to remove from the new model.
    :return: new model with same state as the input model with the number of top layers removed.
    """
    model_children_without_top_layers = list(model.children())[:-num_top_layers]
    return nn.Sequential(*model_children_without_top_layers, Flatten())


class ArgusModel(nn.Module):

    def __init__(self, output_size=1000, train_bb=True, drop_rate=0.5, original_weights=None):
        super().__init__()
        self.output_size = output_size

        model = create_argus_resnext101_32x8d_module(pretrained=True, include_fc_top=False)
        for param in model.parameters():
            param.requires_grad = train_bb

        if original_weights:
            self.load_pretrained_weights_to_model(model, original_weights)
        self.base = model

        self.drop_rate = drop_rate
        num_ftrs = 2048

        if drop_rate != 1.0:
            self.classifier = nn.Sequential(nn.LayerNorm(num_ftrs, elementwise_affine=False),
                                                 nn.Dropout(drop_rate),
                                                 nn.Linear(num_ftrs, output_size))
        else:
            self.classifier = nn.Sequential(nn.LayerNorm(num_ftrs, elementwise_affine=False),
                                                 nn.Linear(num_ftrs, output_size))

    def load_pretrained_weights_to_model(self, model, pretrained_weights='/mnt/nfs/avi/vbt_models/argus_vbt.pth'):

        weights = model.state_dict()

        # Remove non-exist keys
        for key in pretrained_weights.keys() - weights.keys():
            print("Delete unused model state key: %s" % key)
            del pretrained_weights[key]

        # Remove keys that size does not match
        for key, pretrained_weight in list(pretrained_weights.items()):
            weight = weights[key]
            if pretrained_weight.shape != weight.shape:
                print("Delete model state key with unmatched shape: %s" % key)
                del pretrained_weights[key]

        # Copy everything that pretrained_weights miss
        for key in weights.keys() - pretrained_weights.keys():
            print("Missing model state key: %s" % key)
            pretrained_weights[key] = weights[key]

        # Load the weights to model
        model.load_state_dict(pretrained_weights)

    def forward(self, x):
        x = self.base(x)
        x = self.classifier(x)
        return x.double()


def create_pretrained_cnn_backbone(backbone_name: str, requires_grad: bool = True):
    if backbone_name == "argus":
        return create_argus_densenet161_module("/mnt/nfs/roy/projects/VisualSimilarity/src/models/argus_densenet.pth", requires_grad=requires_grad)
    elif backbone_name == "argus2":
        return create_argus_resnext101_32x8d_module("/mnt/nfs/roy/projects/VisualSimilarity/src/models/ArgusVisionEncoder.V6.tar", requires_grad=requires_grad)
    elif backbone_name == "resnet50":
        return create_resnet50_module(pretrained=True, include_fc_top=False, requires_grad=requires_grad)
    elif backbone_name == "resnet101":
        return create_resnet101_module(pretrained=True, include_fc_top=False, requires_grad=requires_grad)
    else:
        raise ValueError(f"Unsupported backbone name: {backbone_name}")


def create_argus_densenet161_module(model_path, requires_grad=True):
    model = models.densenet161()
    pretrained_weight = torch.load(model_path)['state_dict']
    load_argus_state_dict(model, pretrained_weight)
    model.classifier = nn.Sequential()

    for param in model.parameters():
        param.requires_grad = requires_grad

    return model


def create_argus_resnext101_32x8d_module(model_path, requires_grad=True):
    model = models.resnext101_32x8d()
    pretrained_weight = torch.load(model_path)['state_dict']
    load_argus_state_dict(model, pretrained_weight)

    for param in model.parameters():
        param.requires_grad = requires_grad

    return create_sequential_model_without_top(model)


def create_resnet101_module(pretrained=False, include_fc_top=True, requires_grad=True):
    model = models.resnet101(pretrained=pretrained)
    for param in model.parameters():
        param.requires_grad = requires_grad

    return model if include_fc_top else create_sequential_model_without_top(model)


def load_argus_state_dict(model, pretrained_weights):
    weights = model.state_dict()

    # Remove non-exist keys
    for key in pretrained_weights.keys() - weights.keys():
        print("Delete unused model state key: %s" % key)
        del pretrained_weights[key]

    # Remove keys that size does not match
    for key, pretrained_weight in list(pretrained_weights.items()):
        weight = weights[key]
        if pretrained_weight.shape != weight.shape:
            print("Delete model state key with unmatched shape: %s" % key)
            del pretrained_weights[key]

    # Copy everything that pretrained_weights miss
    for key in weights.keys() - pretrained_weights.keys():
        print("Missing model state key: %s" % key)
        pretrained_weights[key] = weights[key]

    # Load the weights to model
    model.load_state_dict(pretrained_weights)


class Flatten(nn.Module):
    """
    Flattens the input into a tensor of size (batch_size, num_elements_in_tensor).
    """

    def forward(self, x):
        return x.view(x.size(0), -1)


class Normalize(nn.Module):
    """
    Normalizes by dividing by the norm of the input tensors.
    """

    def __init__(self, p=2, dim=1):
        super().__init__()
        self.p = p
        self.dim = dim

    def forward(self, x):
        return F.normalize(x, p=self.p, dim=self.dim)


class EmbeddingNet(nn.Module):

    def __init__(self, embedding_size, fc_layers_output_sizes=None, dropout=0, base_requires_grad=True):
        super().__init__()
        self.fc_layers_output_sizes = [embedding_size] if fc_layers_output_sizes is None else fc_layers_output_sizes
        self.dropout = dropout
        self.backbone_output_dim = 2048

        self.base_model = create_resnet50_module(pretrained=True, include_fc_top=False, requires_grad=base_requires_grad)
        self.layer_normalization = nn.LayerNorm([self.backbone_output_dim], elementwise_affine=False)
        self.fc_top = self.__create_fc_top()
        self.l2_normalize = Normalize(p=2, dim=1)

    def __create_fc_top(self):
        fc_top_layers = []
        curr_size = 2048
        for i, output_size in enumerate(self.fc_layers_output_sizes):
            fc_top_layers.append(nn.Linear(curr_size, output_size))
            if i != len(self.fc_layers_output_sizes) - 1:
                fc_top_layers.append(nn.ReLU(inplace=True))

            fc_top_layers.append(nn.Dropout(self.dropout))
            curr_size = output_size

        return nn.Sequential(*fc_top_layers)

    def forward(self, x):
        x = self.base_model(x)
        x = self.layer_normalization(x)
        x = self.fc_top(x)
        return self.l2_normalize(x)


class NegativeSamplingLinear(nn.Module):
    """
    Negative sampling linear layer wrapper that runs the linear layer only for a sample of the negative outputs.
    Used for subsampling before approximating cross entropy loss.
    """

    def __init__(self, linear_layer, negative_sample_ratio=1, sampler=None, normalize_linear_layer=False, reuse_negative_samples=False):
        """
        :param linear_layer: Linear layer to wrap.
        :param negative_sample_ratio: ratio of negative examples out of all possible examples to use per input.
        :param sampler: callable object that given the correct input class index y returns a sequence of sampled indices to use.
        :param normalize_linear_layer: flag whether to normalize linear layer weights to unit vectors during forward pass.
        :param reuse_negative_samples: flag whether to reuse the same negatives for all samples in batch.
        """
        super().__init__()

        if sampler is not None and negative_sample_ratio != 1:
            raise ValueError("sampler option is mutually exclusive with subsample ratio")

        self.linear_layer = linear_layer
        self.negative_sample_ratio = negative_sample_ratio
        self.sampler = sampler
        self.normalize_linear_layer = normalize_linear_layer
        self.reuse_negatives = reuse_negative_samples

    def forward(self, x):
        if self.normalize_linear_layer:
            self.linear_layer.weight.data = F.normalize(self.linear_layer.weight.data, p=2, dim=1)
        return self.linear_layer(x)

    def negative_sample_forward(self, x, y):
        if self.normalize_linear_layer:
            self.linear_layer.weight.data = F.normalize(self.linear_layer.weight.data, p=2, dim=1)

        if self.sampler is None and self.negative_sample_ratio == 1:
            return self.linear_layer(x), y

        if not self.reuse_negatives:
            # when subsampling the correct target will always be the first (index 0)
            y_zeros = torch.zeros_like(y)
            return self.__with_negative_samples_mm(x, y), y_zeros
        else:
            return self.__reuse_negatives_mm(x, y)

    def __with_negative_samples_mm(self, x, y):
        batch_samples_softmax_mat = []
        for i in range(len(y)):
            cur_label = y[i]
            positive_sample = self.linear_layer.weight[cur_label: cur_label + 1]

            neg_samples_indices = self.__get_negative_samples_indices(cur_label)
            neg_samples = self.linear_layer.weight[neg_samples_indices]
            negative_sampled_linear_mat = torch.cat([positive_sample, neg_samples]).t()
            batch_samples_softmax_mat.append(negative_sampled_linear_mat)

        batch_samples_softmax_mat = torch.stack(batch_samples_softmax_mat)
        return torch.bmm(x.unsqueeze(1), batch_samples_softmax_mat).squeeze(1)

    def __get_negative_samples_indices(self, label):
        if self.sampler is not None:
            return self.sampler(label)

        num_labels = self.linear_layer.weight.size(0)
        neg_samples_options = [j for j in range(num_labels) if j != label]
        num_neg_samples = int(self.negative_sample_ratio * num_labels)
        return np.random.choice(neg_samples_options, num_neg_samples, replace=False)

    def __reuse_negatives_mm(self, x, y):
        new_y = torch.zeros_like(y)
        for i in range(len(new_y)):
            new_y[i] = i

        curr_batch_samples = self.linear_layer.weight[y]
        neg_samples_indices = self.__get_negative_samples_indices(y[0])
        neg_samples = self.linear_layer.weight[neg_samples_indices]
        batch_samples_softmax_mat = torch.cat([curr_batch_samples, neg_samples])

        return torch.mm(x, batch_samples_softmax_mat.t()), new_y


class ParametricSoftmaxClassificationNet(nn.Module):

    def __init__(self, embedding_size, output_size, temperature=0.05, negative_sample_ratio=1, reuse_negative_samples=True,
                 dropout=0, base_requires_grad=True):
        super().__init__()
        self.temperature = temperature
        self.output_size = output_size
        self.embedding_base = EmbeddingNet(embedding_size, dropout=dropout, base_requires_grad=base_requires_grad)
        self.negative_sampling_linear = NegativeSamplingLinear(nn.Linear(embedding_size, output_size, bias=False),
                                                               negative_sample_ratio=negative_sample_ratio,
                                                               reuse_negative_samples=reuse_negative_samples,
                                                               normalize_linear_layer=True)

    def epoch_end(self):
        pass

    def forward(self, x):
        x = self.__create_embeddings(x)
        x = self.negative_sampling_linear(x)
        return x / self.temperature

    def negative_sample_forward(self, x, y, return_embeddings=False):
        x = self.__create_embeddings(x)
        y_pred, y = self.negative_sampling_linear.negative_sample_forward(x, y)

        if return_embeddings:
            return x, y_pred / self.temperature, y

        return y_pred / self.temperature, y

    def register_embeddings(self, embeddings, y):
        pass

    def __create_embeddings(self, x):
        return self.embedding_base(x)

    def get_embedding_base(self):
        return self.embedding_base


def load_checkpoint(path_to_checkpoint, output_size, device=torch.device("cuda")):
    # Output size is the size of the last softmax classification layer (i.e., number of item ids used for training)
    model = ParametricSoftmaxClassificationNet(embedding_size=2048, output_size=output_size, base_requires_grad=False)
    trainer_state_dict = torch.load(path_to_checkpoint, map_location=device)
    model.load_state_dict(trainer_state_dict["model"])

    model = model.get_embedding_base()
    for param in model.parameters():
        param.requires_grad = False

    model.eval()
    return model


class DictToSequenceLayer(nn.Module):

    def __init__(self, keys_seq):
        super().__init__()
        self.keys_seq = keys_seq

    def forward(self, dict_input):
        return [dict_input[key] for key in self.keys_seq]



class MultitaskEmbeddingClassificationNetBase(metaclass=ABCMeta):

    def get_embedding_base_with_seq_output(self, task_names):
        by_task_embedding_base = self.get_by_task_embedding_base()
        return nn.Sequential(
            by_task_embedding_base,
            DictToSequenceLayer(task_names)
        )

    @abstractmethod
    def get_by_task_embedding_base(self):
        raise NotImplementedError

    @abstractmethod
    def get_specific_task_embedding_base(self, task_name):
        raise NotImplementedError

    @abstractmethod
    def get_specific_task_classification_net(self, task_name):
        raise NotImplementedError

    @abstractmethod
    def get_by_task_softmax_weights(self):
        raise NotImplementedError


class EmbeddingTop(nn.Module):

    def __init__(self, embedding_size, input_size, dropout=0):
        super().__init__()
        self.layer_normalization = nn.LayerNorm([input_size], elementwise_affine=False)
        self.embedding_fc = nn.Linear(input_size, embedding_size)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_normalization(x)
        x = self.embedding_fc(x)
        x = self.dropout_layer(x)
        x = F.normalize(x, p=2, dim=1)
        return x


class ByTaskEmbeddingTop(nn.Module):

    def __init__(self, by_task_embedding_top):
        super().__init__()
        self.by_task_embedding_top = by_task_embedding_top

    def forward(self, x):
        outputs = OrderedDict()
        for task_name, embedding_top in self.by_task_embedding_top.items():
            outputs[task_name] = embedding_top(x)

        return outputs

class MultiCropWrapper(torch.nn.Module):
    """
    Perform forward pass separately on each resolution input.
    The inputs corresponding to a single resolution are clubbed and single
    forward is run on the same resolution inputs. Hence we do several
    forward passes = number of different resolutions used. We then
    concatenate all the output features and run the head forward on these
    concatenated features.
    """
    def __init__(self, backbone, head=None):
        super(MultiCropWrapper, self).__init__()
        # disable layers dedicated to ImageNet labels classification
        backbone.fc, backbone.head = torch.nn.Identity(), torch.nn.Identity()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        # convert to list
        if not isinstance(x, list):
            x = [x]
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in x]),
            return_counts=True,
        )[1], 0)
        start_idx, output = 0, torch.empty(0).to(x[0].device)
        for end_idx in idx_crops:
            _out = self.backbone(torch.cat(x[start_idx: end_idx]))
            # The output is a tuple with XCiT model. See:
            # https://github.com/facebookresearch/xcit/blob/master/xcit.py#L404-L405
            if isinstance(_out, tuple):
                _out = _out[0]
            # accumulate outputs
            output = torch.cat((output, _out))
            start_idx = end_idx
        # Run the head forward on the concatenated features.
        return self.head(output)

class Dino_FT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        # self.backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vits8')
        
        backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vits8') 
        backbone = MultiCropWrapper(backbone)
        checkpoint = torch.load('/mnt/nfs/tal/VER/src/dino/ikea/checkpoint0020.pth', map_location='cpu')

        msg = backbone.load_state_dict(checkpoint['teacher'], strict=False)
        self.backbone = backbone.backbone
        print(msg)

    def forward(self, x):
        z1 = self.backbone(x)
        z_n = F.normalize(z1, dim=-1)
        return z_n

class Dino(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = torch.hub.load('facebookresearch/dino:main', 'dino_vits8')

    def forward(self, x):
        return F.normalize(self.base_model(x), dim=-1)

    def load_checkpoint(self, params):
        if params.checkpoint != None:
            trainer_state_dict = torch.load(params.checkpoint, map_location=params.device)
            if 'state_dict' in trainer_state_dict:
                print('Loaded checkpint from: {}'.format(params.checkpoint))
                self.load_state_dict(trainer_state_dict["state_dict"])
            elif 'model' in trainer_state_dict:
                print('Loaded checkpint from: {}'.format(params.checkpoint))
                self.load_state_dict(trainer_state_dict["model"])
            else:
                raise ValueError


class Argus(nn.Module):
    def __init__(self, backbone_name="argus"):
        super().__init__()
        self.base_model = create_pretrained_cnn_backbone(backbone_name=backbone_name, requires_grad=True)
        self.g = nn.Sequential(nn.Linear(2208, 512, bias=True), nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True), nn.Linear(512, 128, bias=True))


    def forward(self, x):
        y = self.base_model(x)
        return F.normalize(y, dim=-1)


    def unfreeze(self):
        for name, param in self.base_model.named_parameters():
            param.requires_grad = True


    def load_checkpoint(self, params):
        if params.checkpoint != None:
            trainer_state_dict = torch.load(params.checkpoint, map_location=params.device)
            if 'state_dict' in trainer_state_dict:
                print('Loaded checkpint from: {}'.format(params.checkpoint))
                self.load_state_dict(trainer_state_dict["state_dict"])
            elif 'model' in trainer_state_dict:
                print('Loaded checkpint from: {}'.format(params.checkpoint))
                self.load_state_dict(trainer_state_dict["model"])
            else:
                raise ValueError


class MultitaskEmbeddingClassificationNet(nn.Module, MultitaskEmbeddingClassificationNetBase):

    def __init__(self, by_task_embedding_size, by_task_output_size, dropout=0, temperature=0.05, backbone_name="resnet50", base_requires_grad=True):
        super().__init__()
        self.temperature = temperature
        self.dropout = dropout
        self.by_task_embedding_size = by_task_embedding_size
        self.by_task_output_size = by_task_output_size
        self.backbone_name = backbone_name
        self.backbone_output_dim = get_backbone_output_dim(self.backbone_name)

        self.base_model = create_pretrained_cnn_backbone(backbone_name=self.backbone_name, requires_grad=base_requires_grad)
        self.embedding_top = ByTaskEmbeddingTop(self.__create_embedding_top_modules())
        self.by_task_fc_layer = self.__create_fc_layers()

    def __create_embedding_top_modules(self):
        embedding_top_modules = nn.ModuleDict()
        for task_name, embedding_size in self.by_task_embedding_size.items():
            embedding_top_modules[task_name] = EmbeddingTop(embedding_size, self.backbone_output_dim, dropout=self.dropout)

        return embedding_top_modules

    def __create_fc_layers(self):
        fc_layers = nn.ModuleDict()
        for task_name, output_size in self.by_task_output_size.items():
            fc_layers[task_name] = nn.Linear(self.by_task_embedding_size[task_name], output_size, bias=False)

        return fc_layers

    def forward(self, x):
        x = self.base_model(x)
        outputs = self.embedding_top(x)

        for task_name, fc_layer in self.by_task_fc_layer.items():
            fc_layer.weight.data = F.normalize(fc_layer.weight.data, p=2, dim=1)  # Normalize classification layer weights
            outputs[task_name] = fc_layer(outputs[task_name] / self.temperature)

        return outputs

    def get_by_task_embedding_base(self):
        return nn.Sequential(
            self.base_model,
            self.embedding_top
        )

    def get_specific_task_embedding_base(self, task_name):
        return nn.Sequential(
            self.base_model,
            self.embedding_top.by_task_embedding_top[task_name]
        )

    def get_specific_task_classification_net(self, task_name):
        return nn.Sequential(
            self.base_model,
            self.embedding_top.by_task_embedding_top[task_name],
            self.by_task_fc_layer[task_name]
        )

    def get_by_task_softmax_weights(self):
        return {task_name: F.normalize(fc_layer.weight.data, p=2, dim=1) for task_name, fc_layer in self.by_task_fc_layer.items()}


class MultitaskEmbeddingClassificationNetBase(metaclass=ABCMeta):

    def get_embedding_base_with_seq_output(self, task_names):
        by_task_embedding_base = self.get_by_task_embedding_base()
        return nn.Sequential(
            by_task_embedding_base,
            DictToSequenceLayer(task_names)
        )

    @abstractmethod
    def get_by_task_embedding_base(self):
        raise NotImplementedError

    @abstractmethod
    def get_specific_task_embedding_base(self, task_name):
        raise NotImplementedError

    @abstractmethod
    def get_specific_task_classification_net(self, task_name):
        raise NotImplementedError

    @abstractmethod
    def get_by_task_softmax_weights(self):
        raise NotImplementedError


def get_backbone_output_dim(backbone_name: str):
    if backbone_name == "argus":
        return 2208
    elif backbone_name.startswith("resnet50") or backbone_name == 'argus2':
        return 2048
    else:
        raise ValueError(f"Unsupported backbone name: {backbone_name}")


def get_embedding(images_list, model, device, transform):
    batch_tensor = []
    for image in images_list:
        batch_tensor.append(transform(image))
    batch_tensor = torch.stack(batch_tensor).to(device)

    model.eval()
    with torch.no_grad():
        embd = model(batch_tensor)

    return embd.detach().cpu().numpy()


def unfreeze_base_net(base_net):
    for param in base_net.parameters():
        param.requires_grad = True
    return base_net

class BaseModel(nn.Module):

    def __init__(self, name):
        super().__init__()
        self._name = name

    @abstractmethod
    def get_by_task_embedding_base(self):
        raise NotImplementedError

    @abstractmethod
    def get_specific_task_embedding_base(self, task_name):
        raise NotImplementedError

    @property
    def name(self):
        return self._name

class SharedMultitaskClassificationNet(BaseModel):

    def __init__(self, by_task_embedding_size, by_task_output_size, dropout=0, temperature=0.05, backbone_name="resnet50", base_requires_grad=True):
        super().__init__('SharedMultitaskClassificationNet')

        self.temperature = temperature
        self.dropout = dropout
        self.by_task_embedding_size = by_task_embedding_size
        self.by_task_output_size = by_task_output_size
        self.backbone_name = backbone_name
        self.backbone_output_dim = get_backbone_output_dim(self.backbone_name)

        self.scratch_model = models.resnet18(pretrained=False)
        self.base_model = create_pretrained_cnn_backbone(backbone_name=self.backbone_name, requires_grad=base_requires_grad)
        # self.base_model = torch.hub.load('facebookresearch/dino:main', 'dino_vits8')
        self.by_task_fc_layer = self.__create_fc_layers()

    def __create_fc_layers(self):
        fc_layers = nn.ModuleDict()
        for task_name, output_size in self.by_task_output_size.items():
        #     fc_layers[task_name] = nn.Linear(
        # 384, output_size, bias=False)
            fc_layers[task_name] = nn.Linear(
        self.backbone_output_dim, output_size, bias=False)

        return fc_layers

    def get_backbone_embedding_base(self):
        return self.base_model

    def get_by_task_embedding_base(self):
        return self.get_backbone_embedding_base()

    def get_specific_task_embedding_base(self, task_name):
        return self.get_backbone_embedding_base()

    def forward(self, x):
        x = self.base_model(x)
        x = F.normalize(x, p=2, dim=1)

        outputs = {}
        for task_name, fc_layer in self.by_task_fc_layer.items():
            fc_layer.weight.data = F.normalize(fc_layer.weight.data, p=2, dim=1)  # Normalize classification layer weights
            outputs[task_name] = fc_layer(x / self.temperature)

        return outputs

def create_model(params):
    if params.model_name == 'argus':
        return Argus('argus')
    elif params.model_name == 'dino':
        return Dino()
    else:
        exit('bad model name')
