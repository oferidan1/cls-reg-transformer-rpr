"""
The Multi-Scene TransPoseNet model
"""

import torch
import torch.nn.functional as F
from torch import nn
from .transformer import Transformer
from .pencoder import NestedTensor, nested_tensor_from_tensor_list
from .backbone import build_backbone

class MSTransPoseNet(nn.Module):

    def __init__(self, config, pretrained_path):
        """ Initializes the model.
        """
        super().__init__()

        config["backbone"] = pretrained_path
        config["learn_embedding_with_pose_token"] = False
        num_scenes = config.get("num_scenes")
        self.backbone = build_backbone(config)

        nclusters_position = config.get("nclusters_position")
        nclusters_orientation = config.get("nclusters_orientation")

        config_t = {**config}
        config_t["num_encoder_layers"] = config["num_t_encoder_layers"]
        config_t["num_decoder_layers"] = config["num_t_decoder_layers"]
        config_rot = {**config}
        config_rot["num_encoder_layers"] = config["num_rot_encoder_layers"]
        config_rot["num_decoder_layers"] = config["num_rot_decoder_layers"]
        self.transformer_t = Transformer(config_t)
        self.transformer_rot = Transformer(config_rot)

        decoder_dim = self.transformer_t.d_model

        self.input_proj_t = nn.Conv2d(self.backbone.num_channels[0]*2, decoder_dim, kernel_size=1)
        self.input_proj_rot = nn.Conv2d(self.backbone.num_channels[1]*2, decoder_dim, kernel_size=1)

        self.query_embed_t = nn.Embedding(nclusters_position, decoder_dim)
        self.query_embed_rot = nn.Embedding(nclusters_orientation, decoder_dim)

        self.log_softmax = nn.LogSoftmax(dim=1)

        self.scene_embed = nn.Linear(decoder_dim*2, 1)
        self.regressor_head_t = nn.Sequential(*[PoseRegressor(decoder_dim, 3) for _ in range(num_scenes)])
        self.regressor_head_rot = nn.Sequential(*[PoseRegressor(decoder_dim, 4) for _ in range(num_scenes)])
        self.detect_scene = config["detect_scene"]

    def forward_transformers(self, data):
        """
        Forward of the Transformers
        The forward pass expects a dictionary with key-value 'img' -- NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels NOT USED
        return a dictionary with the following keys--values:
            global_desc_t: latent representation from the position encoder
            global_dec_rot: latent representation from the orientation encoder
            scene_log_distr: the log softmax over the scenes
            max_indices: the index of the max value in the scene distribution
        """
        samples1 = data.get('img1')
        samples2 = data.get('img2')
        #scene_indices = data.get('scene')
        #position_cluster_id = data.get('position_cluster_id')
        #orientation_cluster_id = data.get('orientation_cluster_id')
        batch_size = samples1.shape[0]

        # Handle data structures
        if isinstance(samples1, (list, torch.Tensor)):
            samples1 = nested_tensor_from_tensor_list(samples1)
        if isinstance(samples2, (list, torch.Tensor)):
            samples2 = nested_tensor_from_tensor_list(samples2)

        # Extract the features and the position embedding from the visual backbone
        features1, pos1 = self.backbone(samples1)
        features2, pos2 = self.backbone(samples2)
        #features1[0].mask = torch.cat((features1[0].mask, features2[0].mask), dim=1)
        features1[0].tensors = torch.cat((features1[0].tensors, features2[0].tensors), dim=1)
        #features1[1].mask = torch.cat((features1[1].mask, features2[1].mask), dim=1)
        features1[1].tensors = torch.cat((features1[1].tensors, features2[1].tensors), dim=1)

        pos1[0] = torch.cat((pos1[0], pos2[0]), dim=1)
        pos1[1] = torch.cat((pos1[1], pos2[1]), dim=1)

        #zero non relevant mask
        features1[0].mask = torch.zeros_like(features1[0].mask)
        #sum1 = sum(sum(sum(features1[0].mask)))
        #sum2 = sum(sum(sum(features2[0].mask)))
        #assert sum1==sum2


        src_t, mask_t = features1[0].decompose()
        src_rot, mask_rot = features1[1].decompose()

        # Run through the transformer to translate to "camera-pose" language
        assert mask_t is not None
        assert mask_rot is not None
        local_descs_t = self.transformer_t(self.input_proj_t(src_t), mask_t, self.query_embed_t.weight, pos1[0])[0][0]
        local_descs_rot = self.transformer_rot(self.input_proj_rot(src_rot), mask_rot, self.query_embed_rot.weight, pos1[1])[0][0]

        # TODO
        # replace number of scene with number of cluster

        scene_log_distr = 0
        if self.detect_scene:
            # Get the scene index with FC + log-softmax
            scene_log_distr = self.log_softmax(self.scene_embed(torch.cat((local_descs_t, local_descs_rot), dim=2))).squeeze(2)
            _, max_indices = scene_log_distr.max(dim=1)

            #if scene_indices is not None:
            #    max_indices = scene_indices
            # Take the global latents by zeroing other scene's predictions and summing up
            w = local_descs_t*0
            w[range(batch_size),max_indices, :] = 1
            global_desc_t = torch.sum(w * local_descs_t, dim=1)
            global_desc_rot = torch.sum(w * local_descs_rot, dim=1)

            return {'global_desc_t':global_desc_t,
                    'global_desc_rot':global_desc_rot,
                    'scene_log_distr':scene_log_distr,
                    'max_indices':max_indices}

        return {'global_desc_t': local_descs_t,
                'global_desc_rot': local_descs_rot}

    def forward_heads(self, transformers_res):
        """
        Forward pass of the MLP heads
        The forward pass execpts a dictionary with two keys-values:
        global_desc_t: latent representation from the position encoder
        global_dec_rot: latent representation from the orientation encoder
        scene_log_distr: the log softmax over the scenes
        max_indices: the index of the max value in the scene distribution
        returns: dictionary with key-value 'pose'--expected pose (NX7) and scene_log_distr
        """
        global_desc_t = transformers_res.get('global_desc_t')
        global_desc_rot = transformers_res.get('global_desc_rot')
        max_indices = transformers_res.get('max_indices') # We can only use the max index for weights selection
        batch_size = global_desc_t.shape[0]
        expected_pose = torch.zeros((batch_size,7)).to(global_desc_t.device).to(global_desc_t.dtype)
        for i in range(batch_size):
            x_t = self.regressor_head_t[max_indices[i]](global_desc_t[i].unsqueeze(0))
            x_rot = self.regressor_head_rot[max_indices[i]](global_desc_rot[i].unsqueeze(0))
            expected_pose[i, :] = torch.cat((x_t, x_rot), dim=1)
        return {'pose':expected_pose}

    def forward(self, data):
        """ The forward pass expects a dictionary with the following keys-values
         'img' -- NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels NOT USED
         'scene_indices': ground truth scene indices for each image (can be None)

        returns a dictionary with the following keys-values;
        'pose': expected pose (NX7)
        'log_scene_distr': (log) probability distribution over scenes
        """
        transformers_res = self.forward_transformers(data)
        # Regress the pose from the image descriptors

        heads_res = self.forward_heads(transformers_res)

        return heads_res

class PoseRegressor(nn.Module):
    """ A simple MLP to regress a pose component"""

    def __init__(self, decoder_dim, output_dim, use_prior=False):
        """
        decoder_dim: (int) the input dimension
        output_dim: (int) the outpur dimension
        use_prior: (bool) whether to use prior information
        """
        super().__init__()
        ch = 1024
        self.fc_h = nn.Linear(decoder_dim, ch)
        self.use_prior = use_prior
        if self.use_prior:
            self.fc_h_prior = nn.Linear(decoder_dim * 2, ch)
        self.fc_o = nn.Linear(ch, output_dim)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        """
        Forward pass
        """
        if self.use_prior:
            x = F.gelu(self.fc_h_prior(x))
        else:
            x = F.gelu(self.fc_h(x))

        return self.fc_o(x)
