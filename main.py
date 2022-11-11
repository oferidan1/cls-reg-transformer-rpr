"""
Entry point training and testing multi-scene transformer
"""
import argparse
import torch
import numpy as np
import json
import logging
from util import utils
import time
from datasets.CameraPoseDataset import CameraPoseDataset
from datasets.C2FCameraPoseDataset import C2FCameraPoseDataset
from datasets.RelPoseDataset import RelPoseDataset
from datasets.ClusterPoseDataset import ClusterPoseDataset
from models.pose_losses import CameraPoseLoss
from models.pose_regressors import get_model
from os.path import join
import transforms3d as t3d

def compute_rel_pose(poses1, poses2):
    # p1 p_rel = p2
    rel_pose = torch.zeros_like(poses1)
    poses1 = poses1.cpu().numpy()
    poses2 = poses2.cpu().numpy()
    for i, p1 in enumerate(poses1):
        p2 = poses2[i]
        t1 = p1[:3]
        q1 = p1[3:]
        rot1 = t3d.quaternions.quat2mat(q1 / np.linalg.norm(q1))

        t2 = p2[:3]
        q2 = p2[3:]
        rot2 = t3d.quaternions.quat2mat(q2 / np.linalg.norm(q2))

        t_rel = t2 - t1
        rot_rel = np.dot(np.linalg.inv(rot1), rot2)
        q_rel = t3d.quaternions.mat2quat(rot_rel)
        rel_pose[i][:3] = torch.Tensor(t_rel).to(device)
        rel_pose[i][3:] = torch.Tensor(q_rel).to(device)

    return rel_pose

def batch_dot(v1, v2):
    """
    Dot product along the dim=1
    :param v1: (torch.tensor) Nxd tensor
    :param v2: (torch.tensor) Nxd tensor
    :return: N x 1
    """
    out = torch.mul(v1, v2)
    out = torch.sum(out, dim=1, keepdim=True)
    return out

def qmult(quat_1, quat_2):
    """
    Perform quaternions multiplication
    :param quat_1: (torch.tensor) Nx4 tensor
    :param quat_2: (torch.tensor) Nx4 tensor
    :return: quaternion product
    """
    # Extracting real and virtual parts of the quaternions
    q1s, q1v = quat_1[:, :1], quat_1[:, 1:]
    q2s, q2v = quat_2[:, :1], quat_2[:, 1:]

    qs = q1s*q2s - batch_dot(q1v, q2v)
    qv = q1v.mul(q2s.expand_as(q1v)) + q2v.mul(q1s.expand_as(q2v)) + torch.cross(q1v, q2v, dim=1)
    q = torch.cat((qs, qv), dim=1)
    return q

def compute_abs_pose_torch(rel_pose, abs_pose_neighbor):
    abs_pose_query = torch.zeros_like(rel_pose)
    abs_pose_query[:, :3] = abs_pose_neighbor[:, :3] + rel_pose[:, :3]
    abs_pose_query[:, 3:] = qmult(abs_pose_neighbor[:, 3:], rel_pose[:, 3:])
    return abs_pose_query

def compute_abs_pose(rel_pose, abs_pose_neighbor, device):
    # p_neighbor p_rel = p_query
    # p1 p_rel = p2
    abs_pose_query = torch.zeros_like(rel_pose)
    rel_pose = rel_pose.cpu().numpy()
    abs_pose_neighbor = abs_pose_neighbor.cpu().numpy()
    for i, rpr in enumerate(rel_pose):
        p1 = abs_pose_neighbor[i]

        t_rel = rpr[:3]
        q_rel = rpr[3:]
        rot_rel = t3d.quaternions.quat2mat(q_rel/ np.linalg.norm(q_rel))

        t1 = p1[:3]
        q1 = p1[3:]
        rot1 = t3d.quaternions.quat2mat(q1/ np.linalg.norm(q1))

        t2 = t1 + t_rel
        rot2 = np.dot(rot1,rot_rel)
        q2 = t3d.quaternions.mat2quat(rot2)
        abs_pose_query[i][:3] = torch.Tensor(t2).to(device)
        abs_pose_query[i][3:] = torch.Tensor(q2).to(device)

    return abs_pose_query



def test(args, config, model, apply_c2f, num_clusters_position, num_clusters_orientation, device):
    # Set to eval mode
    model.eval()

    # Set the dataset and data loader
    # Set the dataset and data loader
    transform = utils.test_transforms.get('baseline')
    dataset = RelPoseDataset(args.dataset_path, args.pairs_file, args.cluster_predictor_position, args.cluster_predictor_orientation, transform, num_clusters_position, num_clusters_orientation)
    loader_params = {'batch_size': 1,
                     'shuffle': False,
                     'num_workers': config.get('n_workers')}
    dataloader = torch.utils.data.DataLoader(dataset, **loader_params)
    abs_stats = np.zeros((len(dataloader.dataset), 3))

    with torch.no_grad():
        for i, minibatch in enumerate(dataloader, 0):
            for k, v in minibatch.items():
                minibatch[k] = v.to(device)
            #minibatch['scene'] = None  # avoid using ground-truth scene during prediction
            minibatch['cluster_id_position'] = None  # avoid using ground-truth cluster during prediction
            minibatch['cluster_id_orientation'] = None  # avoid using ground-truth cluster during prediction

            gt_pose = minibatch.get('pose1').to(dtype=torch.float32)

            # Forward pass to predict the pose
            tic = time.time()
            est_rel_pose = model(minibatch).get('rel_pose')

            # Flip to get the relative from neighbor to query
            est_rel_pose[:, :3] = -est_rel_pose[:, :3]
            est_rel_pose[:, 4:] = -est_rel_pose[:, 4:]

            est_pose = compute_abs_pose(est_rel_pose, minibatch.get('pose2'), device)

            torch.cuda.synchronize()
            toc = time.time()

            # Evaluate error
            posit_err, orient_err = utils.pose_err(est_pose, gt_pose)

            # Collect statistics
            abs_stats[i, 0] = posit_err.item()
            abs_stats[i, 1] = orient_err.item()
            abs_stats[i, 2] = (toc - tic) * 1000

            logging.info("Absolute Pose error: {:.3f}[m], {:.3f}[deg], inferred in {:.2f}[ms]".format(
                abs_stats[i, 0], abs_stats[i, 1], abs_stats[i, 2]))

    # Record overall statistics
    logging.info("Performance of {} on {}".format(args.checkpoint_path, args.labels_file))
    logging.info("Median absolute pose error: {:.3f}[m], {:.3f}[deg]".format(np.nanmedian(abs_stats[:, 0]),
                                                                             np.nanmedian(abs_stats[:, 1])))
    logging.info("Mean pose inference time:{:.2f}[ms]".format(np.mean(abs_stats[:, 2])))
    return abs_stats

def valid(model, dataloader, device, freeze, apply_c2f, pose_loss, nll_loss_x, nll_loss_q, n_freq_print, epoch):
    n_samples = 0
    n_total_samples = 0
    running_loss = 0.0
    loss_vals = []
    sample_count = []
    model.eval()
    stats = np.zeros((len(dataloader.dataset), 2))
    i=0
    for batch_idx, minibatch in enumerate(dataloader):
        for k, v in minibatch.items():
            minibatch[k] = v.to(device)
        gt_pose = minibatch.get('pose').to(dtype=torch.float32)
        gt_scene = minibatch.get('scene').to(device)
        position_gt_cluster = minibatch.get('position_cluster_id').to(device)
        orientation_gt_cluster = minibatch.get('orientation_cluster_id').to(device)

        batch_size = gt_pose.shape[0]
        n_samples += batch_size
        n_total_samples += batch_size

        if freeze:  # For TransPoseNet
            with torch.no_grad():
                transformers_res = model.forward_transformers(minibatch)

        # Zero the gradients
        #optim.zero_grad()

        # Forward pass to estimate the pose
        if freeze:
            if apply_c2f:
                with torch.no_grad():
                    res = model.forward_heads(transformers_res, minibatch)
            else:
                with torch.no_grad():
                    res = model.forward_heads(transformers_res)
        else:
            with torch.no_grad():
                res = model(minibatch)

        est_pose = res.get('rel_pose')
        est_scene_log_distr = res.get('scene_log_distr')
        est_position_cluster_log_distr = res.get('position_cluster_log_distr')
        est_orientation_cluster_log_distr = res.get('orientation_cluster_log_distr')

        if est_scene_log_distr is not None:
            # Pose Loss + Scene Loss
            if apply_c2f:
                criterion = pose_loss(est_pose, gt_pose) \
                            + nll_loss_x(est_position_cluster_log_distr, position_gt_cluster) + nll_loss_q(
                    est_orientation_cluster_log_distr, orientation_gt_cluster)
            else:
                criterion = pose_loss(est_pose, gt_pose)
        else:
            # Pose loss
            criterion = pose_loss(est_pose, gt_pose)

        # Collect for recoding and plotting
        running_loss += criterion.item()
        loss_vals.append(criterion.item())
        sample_count.append(n_total_samples)

        # Record loss and performance on train set

        posit_err, orient_err = utils.pose_err(est_pose.detach(), gt_pose.detach())
        # Collect statistics
        stats[i, 0] = posit_err.mean().item()
        stats[i, 1] = orient_err.mean().item()
        logging.info("[Batch-{}/Epoch-{}] running camera pose loss: {:.3f}, "
                         "camera pose error: {:.2f}[m], {:.2f}[deg]".format(
                batch_idx + 1, epoch + 1, (running_loss / n_samples),
                posit_err.mean().item(),
                orient_err.mean().item()))
        i += 1

    logging.info("median camera pose error: {:.2f}[m], {:.2f}[deg]".format(
        np.nanmedian(stats[:, 0]),
        np.nanmedian(stats[:, 1])))

    # back to train
    model.train()

def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--model_name", help="name of model to create (e.g. posenet, transposenet", default="c2f-ems-transposenet")
    arg_parser.add_argument("--mode", help="train or eval", default="train")
    arg_parser.add_argument("--backbone_path", help="path to backbone .pth - e.g. efficientnet", default="models/backbones/efficient-net-b0.pth")
    arg_parser.add_argument("--dataset_path", help="path to the physical location of the dataset", default="/nfstemp/Datasets/7Scenes/")
    arg_parser.add_argument("--labels_file", help="path to a file mapping images to their poses", default="datasets/7Scenes/7scenes_training_pairs_train.csv")
    arg_parser.add_argument("--labels_file_valid", help="path to a file mapping images to their poses", default="datasets/7Scenes/7scenes_training_pairs_valid.csv")
    arg_parser.add_argument("--pairs_file", help="path to a file mapping images to their poses", default="datasets/7Scenes/7scenes_training_pairs.csv")
    arg_parser.add_argument("--config_file", help="path to configuration file", default="7scenes_config.json")
    arg_parser.add_argument("--checkpoint_path", help="path to a pre-trained model (should match the model indicated in model_name")
    arg_parser.add_argument("--experiment", help="a short string to describe the experiment/commit used")
    arg_parser.add_argument("--cluster_predictor_position", help="path to position k-means predictor")
    arg_parser.add_argument("--cluster_predictor_orientation", help="path to orientation k-means predictor")
    arg_parser.add_argument("--test_dataset_id", default="7scenes", help="test set id for testing on all scenes, options: 7scene OR cambridge")
    #arg_parser.add_argument("--nclusters_position", type=int, default=1, help="number of position clusters")
    #arg_parser.add_argument("--nclusters_orientation", type=int, default=2, help="number of orientation clusters")


    args = arg_parser.parse_args()
    utils.init_logger()

    # Record execution details
    logging.info("Start {} with {}".format(args.model_name, args.mode))
    if args.experiment is not None:
        logging.info("Experiment details: {}".format(args.experiment))
    logging.info("Using dataset: {}".format(args.dataset_path))
    logging.info("Using labels file: {}".format(args.labels_file))

    # Read configuration
    with open(args.config_file, "r") as read_file:
        config = json.load(read_file)
    model_params = config[args.model_name]
    general_params = config['general']
    config = {**model_params, **general_params}
    logging.info("Running with configuration:\n{}".format(
        '\n'.join(["\t{}: {}".format(k, v) for k, v in config.items()])))

    # Set the seeds and the device
    use_cuda = torch.cuda.is_available()
    device_id = 'cpu'
    torch_seed = 0
    numpy_seed = 2
    torch.manual_seed(torch_seed)
    if use_cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        device_id = config.get('device_id')
    np.random.seed(numpy_seed)
    device = torch.device(device_id)

    # Coarse-to-fine params
    apply_c2f = config.get("c2f")
    num_clusters_position = config.get("nclusters_position")
    num_clusters_orientation = config.get("nclusters_orientation")

    # Create the model
    model = get_model(args.model_name, args.backbone_path, config).to(device)
    # Load the checkpoint if needed
    if args.checkpoint_path:

        msg = model.load_state_dict(torch.load(args.checkpoint_path, map_location=device_id), strict=False)
        logging.info(msg)
        logging.info("Initializing from checkpoint: {}".format(args.checkpoint_path))

    if args.mode == 'train' or args.mode == 'valid':

        # Set to train mode
        model.train()

        # Freeze parts of the model if indicated
        freeze = config.get("freeze")
        freeze_exclude_phrase = config.get("freeze_exclude_phrase")
        if isinstance(freeze_exclude_phrase, str):
            freeze_exclude_phrase = [freeze_exclude_phrase]
        if freeze:
            for name, parameter in model.named_parameters():
                freeze_param = True
                for phrase in freeze_exclude_phrase:
                    if phrase in name:
                        freeze_param = False
                        break
                if freeze_param:
                        parameter.requires_grad_(False)

        # Set the dataset and data loader
        no_augment = config.get("no_augment")
        if no_augment:
            transform = utils.test_transforms.get('baseline')
        else:
            transform = utils.train_transforms.get('baseline')

        equalize_scenes = config.get("equalize_scenes")
        if apply_c2f:
            dataset = ClusterPoseDataset(args.dataset_path, args.labels_file, transform)
            dataset_valid = ClusterPoseDataset(args.dataset_path, args.labels_file_valid, transform)
        else:
            dataset = CameraPoseDataset(args.dataset_path, args.labels_file, transform, equalize_scenes)

        loader_params = {'batch_size': config.get('batch_size'),
                                  'shuffle': True,
                                  'num_workers': config.get('n_workers')}
        dataloader = torch.utils.data.DataLoader(dataset, **loader_params)
        dataloader_valid = torch.utils.data.DataLoader(dataset_valid, **loader_params)

        x_weights, q_weights = dataset.calc_cluster_weights()
        x_weights = torch.FloatTensor(x_weights).to(device)
        q_weights = torch.FloatTensor(q_weights).to(device)

        # Set the loss
        pose_loss = CameraPoseLoss(config).to(device)
        nll_loss_x = torch.nn.NLLLoss(weight=x_weights)
        nll_loss_q = torch.nn.NLLLoss(weight=q_weights)

        # Set the optimizer and scheduler
        params = list(model.parameters()) + list(pose_loss.parameters())
        optim = torch.optim.Adam(filter(lambda p: p.requires_grad, params),
                                 lr=config.get('lr'),
                                 eps=config.get('eps'),
                                 weight_decay=config.get('weight_decay'))
        scheduler = torch.optim.lr_scheduler.StepLR(optim,
                                                    step_size=config.get('lr_scheduler_step_size'),
                                                    gamma=config.get('lr_scheduler_gamma'))

        # Get training details
        n_freq_print = config.get("n_freq_print")
        n_freq_checkpoint = config.get("n_freq_checkpoint")
        n_epochs = config.get("n_epochs")

        # Train
        checkpoint_prefix = join(utils.create_output_dir('out'),utils.get_stamp_from_log())
        n_total_samples = 0.0
        loss_vals = []
        sample_count = []

        if args.mode == 'valid':
            valid(model, dataloader_valid, device, freeze, apply_c2f, pose_loss, nll_loss_x, nll_loss_q, n_freq_print, 0)
            return

        for epoch in range(n_epochs):

            # Resetting temporal loss used for logging
            running_loss = 0.0
            n_samples = 0

            for batch_idx, minibatch in enumerate(dataloader):
                for k, v in minibatch.items():
                    minibatch[k] = v.to(device)
                gt_pose = minibatch.get('pose').to(dtype=torch.float32)
                gt_scene = minibatch.get('scene').to(device)
                position_gt_cluster = minibatch.get('position_cluster_id').to(device)
                orientation_gt_cluster = minibatch.get('orientation_cluster_id').to(device)

                batch_size = gt_pose.shape[0]
                n_samples += batch_size
                n_total_samples += batch_size

                if freeze: # For TransPoseNet
                    model.eval()
                    with torch.no_grad():
                        transformers_res = model.forward_transformers(minibatch)
                    model.train()

                # Zero the gradients
                optim.zero_grad()

                # Forward pass to estimate the pose
                if freeze:
                    if apply_c2f:
                        res = model.forward_heads(transformers_res, minibatch)
                    else:
                        res = model.forward_heads(transformers_res)
                else:
                    res = model(minibatch)

                est_pose = res.get('rel_pose')
                #est_scene_log_distr = res.get('scene_log_distr')
                est_position_cluster_log_distr = res.get('position_cluster_log_distr')
                est_orientation_cluster_log_distr = res.get('orientation_cluster_log_distr')

                if est_position_cluster_log_distr is not None:
                    # Pose Loss + Scene Loss
                    if apply_c2f:
                        criterion = pose_loss(est_pose, gt_pose)  \
                                    #+ nll_loss_x(est_position_cluster_log_distr, position_gt_cluster) + nll_loss_q(est_orientation_cluster_log_distr, orientation_gt_cluster)
                    else:
                        criterion = pose_loss(est_pose, gt_pose)
                else:
                    # Pose loss
                    criterion = pose_loss(est_pose, gt_pose)

                # Collect for recoding and plotting
                running_loss += criterion.item()
                loss_vals.append(criterion.item())
                sample_count.append(n_total_samples)

                # Back prop
                criterion.backward()
                optim.step()

                # Record loss and performance on train set
                if batch_idx % n_freq_print == 0:
                    posit_err, orient_err = utils.pose_err(est_pose.detach(), gt_pose.detach())
                    logging.info("[Batch-{}/Epoch-{}] running camera pose loss: {:.3f}, "
                                 "camera pose error: {:.2f}[m], {:.2f}[deg]".format(
                                                                        batch_idx+1, epoch+1, (running_loss/n_samples),
                                                                        posit_err.mean().item(),
                                                                        orient_err.mean().item()))
            # Save checkpoint
            if (epoch % n_freq_checkpoint) == 0 and epoch > 0:
                torch.save(model.state_dict(), checkpoint_prefix + '_checkpoint-{}.pth'.format(epoch))

            # Scheduler update
            scheduler.step()

        logging.info('Training completed')
        torch.save(model.state_dict(), checkpoint_prefix + '_final.pth'.format(epoch))

        # Plot the loss function
        loss_fig_path = checkpoint_prefix + "_loss_fig.png"
        utils.plot_loss_func(sample_count, loss_vals, loss_fig_path)

    else: # Test
        if args.test_dataset_id is not None:
            f = open("{}_{}_report.csv".format(args.test_dataset_id,  utils.get_stamp_from_log()), 'w')
            f.write("scene,pos,ori\n")
            if args.test_dataset_id == "7scenes":
                scenes = ["chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs"]
                #scenes = ["heads", "office", "pumpkin", "redkitchen", "stairs"]
                for scene in scenes:
                    args.cluster_predictor_position = "./datasets/7Scenes/7scenes_training_pairs.csv_scene_{}_position_{}_classes.sav".format(scene, num_clusters_position)
                    args.cluster_predictor_orientation = "./datasets/7Scenes/7scenes_training_pairs.csv_scene_{}_orientation_{}_classes.sav".format(scene, num_clusters_orientation)

                    args.pairs_file = "./datasets/7Scenes_test/NN_7scenes_{}.csv".format(scene)
                    #args.pairs_file = "./datasets/7Scenes_train/7scenes_training_pairs_{}.csv".format(scene)
                    stats = test(args, config, model, apply_c2f, num_clusters_position, num_clusters_orientation, device)
                    f.write("{},{:.3f},{:.3f}\n".format(scene, np.nanmedian(stats[:, 0]),
                                                                                    np.nanmedian(stats[:, 1])))
            elif args.test_dataset_id == "cambridge":

                scenes = ["KingsCollege", "OldHospital", "ShopFacade", "StMarysChurch"]
                for scene in scenes:
                    args.cluster_predictor_position = "./datasets/CambridgeLandmarks/cambridge_four_scenes.csv_scene_{}_position_{}_classes.sav".format(
                        scene, num_clusters_position)
                    args.cluster_predictor_orientation = "./datasets/CambridgeLandmarks/cambridge_four_scenes.csv_scene_{}_orientation_{}_classes.sav".format(
                        scene, num_clusters_orientation)
                    args.labels_file = "./datasets/CambridgeLandmarks/abs_cambridge_pose_sorted.csv_{}_test.csv".format(scene)
                    stats = test(args, config, model, apply_c2f, num_clusters_position, num_clusters_orientation)
                    f.write("{},{:.3f},{:.3f}\n".format(scene, np.nanmedian(stats[:, 0]),
                                                                                    np.nanmedian(stats[:, 1])))
            else:
                raise NotImplementedError()
            f.close()
        else:
            _ = test(args, config, model, apply_c2f, num_clusters_position, num_clusters_orientation)


if __name__ == "__main__":
    main()
