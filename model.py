import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet_features import resnet18_features, resnet34_features, resnet50_features, resnet101_features, resnet152_features
from densenet_features import densenet121_features
from vgg_features import vgg11_features, vgg11_bn_features, vgg13_features, vgg13_bn_features, vgg16_features, vgg16_bn_features,\
                         vgg19_features, vgg19_bn_features
import copy
from receptive_field import compute_proto_layer_rf_info_v2
from helpers import list_of_similarities_2d


base_architecture_to_features = {'resnet18': resnet18_features,
                                 'resnet34': resnet34_features,
                                 'resnet50': resnet50_features,
                                 'resnet101': resnet101_features,
                                 'resnet152': resnet152_features,
                                 'densenet121': densenet121_features,
                                 'vgg11': vgg11_features,
                                 'vgg11_bn': vgg11_bn_features,
                                 'vgg13': vgg13_features,
                                 'vgg13_bn': vgg13_bn_features,
                                 'vgg16': vgg16_features,
                                 'vgg16_bn': vgg16_bn_features,
                                 'vgg19': vgg19_features,
                                 'vgg19_bn': vgg19_bn_features,
                                 }

class CIPL(nn.Module):

    def __init__(self, features, img_size, prototype_shape,
                 proto_layer_rf_info, num_classes, init_weights=True,
                 prototype_activation_function='log',
                 add_on_layers_type='bottleneck'):

        super(CIPL, self).__init__()
        self.img_size = img_size
        self.prototype_shape = prototype_shape
        self.num_prototypes = prototype_shape[0]
        self.num_classes = num_classes
        self.epsilon = 1e-4
        
        # prototype_activation_function could be 'log', 'linear',
        # or a generic function that converts distance to similarity score
        self.prototype_activation_function = prototype_activation_function

        '''
        Here we are initializing the class identities of the prototypes
        Without domain specific knowledge we allocate the same number of
        prototypes for each class
        '''
        assert(self.num_prototypes % self.num_classes == 0)
        # a onehot indication matrix for each prototype's class identity
        self.prototype_class_identity = torch.zeros(self.num_prototypes, self.num_classes)

        num_prototypes_per_class = self.num_prototypes // self.num_classes
        for j in range(self.num_prototypes):
            self.prototype_class_identity[j, j // num_prototypes_per_class] = 1

        self.proto_layer_rf_info = proto_layer_rf_info

        # this has to be named features to allow the precise loading
        self.features = features

        features_name = str(self.features).upper()
        if features_name.startswith('VGG') or features_name.startswith('RES'):
            first_add_on_layer_in_channels = \
                [i for i in features.modules() if isinstance(i, nn.Conv2d)][-1].out_channels
        elif features_name.startswith('DENSE'):
            first_add_on_layer_in_channels = \
                [i for i in features.modules() if isinstance(i, nn.BatchNorm2d)][-1].num_features
        elif features_name.startswith('EFFICIENT'):
            first_add_on_layer_in_channels = \
                [i for i in features.modules() if isinstance(i, nn.BatchNorm2d)][-1].num_features
        else:
            raise Exception('other base base_architecture NOT implemented')

        if add_on_layers_type == 'bottleneck':
            add_on_layers = []
            current_in_channels = first_add_on_layer_in_channels
            while (current_in_channels > self.prototype_shape[1]) or (len(add_on_layers) == 0):
                current_out_channels = max(self.prototype_shape[1], (current_in_channels // 2))
                add_on_layers.append(nn.Conv2d(in_channels=current_in_channels,
                                               out_channels=current_out_channels,
                                               kernel_size=1))
                add_on_layers.append(nn.ReLU())
                add_on_layers.append(nn.Conv2d(in_channels=current_out_channels,
                                               out_channels=current_out_channels,
                                               kernel_size=1))
                if current_out_channels > self.prototype_shape[1]:
                    add_on_layers.append(nn.ReLU())
                else:
                    assert(current_out_channels == self.prototype_shape[1])
                    add_on_layers.append(nn.Sigmoid())
                current_in_channels = current_in_channels // 2
            self.add_on_layers = nn.Sequential(*add_on_layers)
        else:
            # first_add_on_layer_in_channels = 512
            first_add_on_layer_in_channels = 256
            self.add_on_layers = nn.Sequential(
                nn.Conv2d(in_channels=first_add_on_layer_in_channels, out_channels=self.prototype_shape[1], kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=self.prototype_shape[1], out_channels=self.prototype_shape[1], kernel_size=1),
                nn.Sigmoid()
                )
        
        self.prototype_vectors = nn.Parameter(torch.rand(self.prototype_shape), requires_grad=True)
        
        # do not make this just a tensor, since it will not be moved automatically to gpu
        self.ones = nn.Parameter(torch.ones(self.prototype_shape), requires_grad=False)

        self.last_layer = nn.Linear(self.num_prototypes, self.num_classes, bias=False)  # do not use bias

        if init_weights:
            self._initialize_weights()

        self.use_ema = True
        if self.use_ema:
            self.ema_momentum = 0.999

            self.features_ema = copy.deepcopy(self.features)
            self.add_on_layers_ema = copy.deepcopy(self.add_on_layers)
            self.prototype_vectors_ema = copy.deepcopy(self.prototype_vectors)
            self.last_layer_ema = copy.deepcopy(self.last_layer)
            self._ema_param_initialized = False

    def _init_ema_params(self):
        assert not self._ema_param_initialized
        for p_ema, p in zip(self.features_ema.parameters(), self.features.parameters()):
            p_ema.requires_grad = False       # not update by gradient
            p_ema.data.copy_(p.data)          # initialize
        for p_ema, p in zip(self.add_on_layers_ema.parameters(), self.add_on_layers.parameters()):
            p_ema.requires_grad = False
            p_ema.data.copy_(p.data)
        self.prototype_vectors_ema.requires_grad = False
        self.prototype_vectors_ema.data.copy_(self.prototype_vectors.data)
        for p_ema, p in zip(self.last_layer_ema.parameters(), self.last_layer.parameters()):
            p_ema.requires_grad = False
            p_ema.data.copy_(p.data)
        self._ema_param_initialized = True
        print('EMA initialization...')

    def conv_features(self, x):
        '''
        the feature input to prototype layer
        '''
        x_ori = self.features(x)
        x = self.add_on_layers(x_ori)
        return x, x_ori

    @staticmethod
    def _weighted_l2_convolution(input, filter, weights):
        '''
        input of shape N * c * h * w
        filter of shape P * c * h1 * w1
        weight of shape P * c * h1 * w1
        '''
        input2 = input ** 2
        input_patch_weighted_norm2 = F.conv2d(input=input2, weight=weights)

        filter2 = filter ** 2
        weighted_filter2 = filter2 * weights
        filter_weighted_norm2 = torch.sum(weighted_filter2, dim=(1, 2, 3))
        filter_weighted_norm2_reshape = filter_weighted_norm2.view(-1, 1, 1)

        weighted_filter = filter * weights
        weighted_inner_product = F.conv2d(input=input, weight=weighted_filter)

        # use broadcast
        intermediate_result = \
            - 2 * weighted_inner_product + filter_weighted_norm2_reshape
        # x2_patch_sum and intermediate_result are of the same shape
        distances = F.relu(input_patch_weighted_norm2 + intermediate_result)

        return distances

    def _l2_convolution(self, x, prototypes):
        '''
        apply self.prototype_vectors as l2-convolution filters on input x
        '''
        x2 = x ** 2
        x2_patch_sum = F.conv2d(input=x2, weight=self.ones)

        p2 = prototypes ** 2
        p2 = torch.sum(p2, dim=(1, 2, 3))
        # p2 is a vector of shape (num_prototypes,)
        # then we reshape it to (num_prototypes, 1, 1)
        p2_reshape = p2.view(-1, 1, 1)

        xp = F.conv2d(input=x, weight=prototypes)
        intermediate_result = - 2 * xp + p2_reshape  # use broadcast
        # x2_patch_sum and intermediate_result are of the same shape
        distances = F.relu(x2_patch_sum + intermediate_result)

        return distances

    def _cosine_convolution(self, x, prototypes):

        # x = F.normalize(x, p=2, dim=1)
        # now_prototype_vectors = F.normalize(prototypes, p=2, dim=1)
        now_prototype_vectors = prototypes
        similarity = F.conv2d(input=x, weight=now_prototype_vectors)  # [-64, 64]
        distances = - similarity    # [-1, 1]
        return distances

    def prototype_distances(self, x):
        '''
        x is the raw input
        '''
        conv_features, x_ori = self.conv_features(x)
        distances = self._l2_convolution(conv_features, self.prototype_vectors)
        # distances = self._cosine_convolution(conv_features, self.prototype_vectors)
        return distances, conv_features, x_ori

    def distance_2_similarity(self, distances):
        if self.prototype_activation_function == 'log':
            return torch.log((distances + 1) / (distances + self.epsilon))
        elif self.prototype_activation_function == 'linear':
            return -distances
        else:
            return self.prototype_activation_function(distances)

    def distance_2_similarity_exp(self, distances):
        return torch.exp(-distances / 256.0)  # 128.0

    def distance_2_similarity_linear(self, distances):
        return - distances    # [-1, 1]

    def classifier(self, conv_features):
        distances = self._l2_convolution(conv_features, self.prototype_vectors)
        # global min pooling
        min_distances = -F.max_pool2d(-distances,
                                      kernel_size=(distances.size()[2],
                                                   distances.size()[3]))
        min_distances = min_distances.view(-1, self.num_prototypes)
        prototype_activations = self.distance_2_similarity_exp(min_distances)
        logits = self.last_layer(prototype_activations)

        return logits, self.distance_2_similarity_exp(distances), min_distances

    def forward(self, x, x_aug):
        input1, input2 = x[0], x[1]
        feature1, feat_ori1 = self.conv_features(input1)
        feature2, feat_ori2 = self.conv_features(input2)

        score1, similarities1, min_distances_1 = self.classifier(feature1)
        score2, similarities2, min_distances_2 = self.classifier(feature2)

        fea_size1 = feature1.size()[2:]
        all_dim1 = fea_size1[0] * fea_size1[1]

        fea_size2 = feature2.size()[2:]
        all_dim2 = fea_size2[0] * fea_size2[1]

        feature1_flat = feature1.view(-1, feature1.size()[1], all_dim1)   # [b, D, 256]
        feature2_flat = feature2.view(-1, feature2.size()[1], all_dim2)   # [b, D, 256]

        A2 = list_of_similarities_2d(feature1_flat, feature2_flat)

        A = F.softmax(A2, dim=1)
        B = F.softmax(torch.transpose(A2, 1, 2), dim=1)
        feature2_att = torch.bmm(feature1_flat, A).contiguous()
        feature1_att = torch.bmm(feature2_flat, B).contiguous()
        input1_att = feature1_att.view(-1, feature1.size()[1], fea_size1[0], fea_size1[1])
        input2_att = feature2_att.view(-1, feature2.size()[1], fea_size2[0], fea_size2[1])

        co_score1, _, _ = self.classifier(input1_att)
        co_score2, _, _ = self.classifier(input2_att)

        simi_ema, logits_ema = self.forward_ema(torch.cat([x_aug[0], x_aug[1]]))

        return torch.cat([score1, score2]), \
               torch.cat([min_distances_1, min_distances_2]), \
               torch.cat([co_score1, co_score2]), \
               torch.cat([similarities1, similarities2]), \
               simi_ema, \
               logits_ema


    @torch.no_grad()
    def forward_ema(self, image):
        assert self.use_ema
        if not self._ema_param_initialized:
            self._init_ema_params()

        if self.training:
            for p_ema, p in zip(self.features_ema.parameters(), self.features.parameters()):
                p_ema.data.copy_(p_ema.data * self.ema_momentum + p.data * (1 - self.ema_momentum))
            for p_ema, p in zip(self.add_on_layers_ema.parameters(), self.add_on_layers.parameters()):
                p_ema.data.copy_(p_ema.data * self.ema_momentum + p.data * (1 - self.ema_momentum))
            self.prototype_vectors_ema.data.copy_(self.prototype_vectors_ema.data * self.ema_momentum + self.prototype_vectors.data * (1 - self.ema_momentum))
            for p_ema, p in zip(self.last_layer_ema.parameters(), self.last_layer.parameters()):
                p_ema.data.copy_(p_ema.data * self.ema_momentum + p.data * (1 - self.ema_momentum))

        feat_ori_ema = self.features_ema(image)
        feat_ema = self.add_on_layers_ema(feat_ori_ema)

        distances_ema = self._l2_convolution(feat_ema, self.prototype_vectors_ema)
        similarities_ema = self.distance_2_similarity_exp(distances_ema)

        max_similarities_ema = F.max_pool2d(similarities_ema,
                                      kernel_size=(similarities_ema.size()[2],
                                                   similarities_ema.size()[3]))
        max_similarities_ema = max_similarities_ema.view(-1, self.num_prototypes)
        logits_ema = self.last_layer_ema(max_similarities_ema)

        return similarities_ema, logits_ema


    def forward_infer(self, x):
        conv_features, x_ori = self.conv_features(x)
        distances = self._l2_convolution(conv_features, self.prototype_vectors)

        similarity_maps = self.distance_2_similarity_exp(distances)
        '''
        we cannot refactor the lines below for similarity scores
        because we need to return min_distances
        '''
        # global min pooling
        min_distances = -F.max_pool2d(-distances,
                                      kernel_size=(distances.size()[2],
                                                   distances.size()[3]))
        min_distances = min_distances.view(-1, self.num_prototypes)
        prototype_activations = self.distance_2_similarity_exp(min_distances)
        logits = self.last_layer(prototype_activations)

        return logits, min_distances, similarity_maps


    def push_forward(self, x):
        '''this method is needed for the pushing operation'''
        conv_output, x_ori = self.conv_features(x)
        distances = self._l2_convolution(conv_output, self.prototype_vectors)
        return conv_output, distances

    def __repr__(self):
        # CIPL(self, features, img_size, prototype_shape, proto_layer_rf_info, num_classes, init_weights=True):
        rep = (
            'PPNet(\n'
            '\tfeatures: {},\n'
            '\timg_size: {},\n'
            '\tprototype_shape: {},\n'
            '\tproto_layer_rf_info: {},\n'
            '\tnum_classes: {},\n'
            '\tepsilon: {}\n'
            ')'
        )

        return rep.format(self.features,
                          self.img_size,
                          self.prototype_shape,
                          self.proto_layer_rf_info,
                          self.num_classes,
                          self.epsilon)

    def set_last_layer_incorrect_connection(self, incorrect_strength):
        '''
        the incorrect strength will be actual strength if -0.5 then input -0.5
        '''
        positive_one_weights_locations = torch.t(self.prototype_class_identity)
        negative_one_weights_locations = 1 - positive_one_weights_locations

        correct_class_connection = 1
        incorrect_class_connection = incorrect_strength
        self.last_layer.weight.data.copy_(
            correct_class_connection * positive_one_weights_locations
            + incorrect_class_connection * negative_one_weights_locations)

    def _initialize_weights(self):
        for m in self.add_on_layers.modules():
            if isinstance(m, nn.Conv2d):
                # every init technique has an underscore _ in the name
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu') 

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.set_last_layer_incorrect_connection(incorrect_strength=0.0)


def construct_CIPL(base_architecture, pretrained=True, img_size=224,
                   prototype_shape=(2000, 512, 1, 1), num_classes=200,
                   prototype_activation_function='log',
                   add_on_layers_type='bottleneck'):
    features = base_architecture_to_features[base_architecture](pretrained=pretrained)
    layer_filter_sizes, layer_strides, layer_paddings = features.conv_info()
    proto_layer_rf_info = compute_proto_layer_rf_info_v2(img_size=img_size,
                                                         layer_filter_sizes=layer_filter_sizes,
                                                         layer_strides=layer_strides,
                                                         layer_paddings=layer_paddings,
                                                         prototype_kernel_size=prototype_shape[2])
    return CIPL(features=features,
                img_size=img_size,
                prototype_shape=prototype_shape,
                proto_layer_rf_info=proto_layer_rf_info,
                num_classes=num_classes,
                init_weights=True,
                prototype_activation_function=prototype_activation_function,
                add_on_layers_type=add_on_layers_type)