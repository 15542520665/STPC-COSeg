"""STPC-COSeg test-only entry point."""

import argparse
import copy
import json
import os
import random
import time
from functools import partial

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.utils.data

from model.coseg_ablation import COSegAblation
from util import config
from util.common_util import AverageMeter, evaluate_metric
from util.data_util import collate_fn_limit_fs
from util.logger import get_logger
from util.s3dis_fs import S3DIS_FS_TEST, S3DIS_FSForVIS

try:
    from util.scannet_v2_fs import Scannetv2_FS_TEST
except ImportError:
    Scannetv2_FS_TEST = None


_BASE_CONFIG = {
    "use_cross_attention": False,
    "use_contrastive_loss": False,
    "use_dice_loss": False,
    "use_lovasz_loss": False,
    "use_enhanced_aggregator": False,
    "use_pos_encoding": False,
    "use_consistency_loss": False,
    "use_focal_loss": False,
    "use_boundary_loss": False,
    "use_label_smoothing": False,
    "use_multiscale_correlation": False,
    "use_enhanced_augmentation": False,
    "use_prototype_refinement": False,
    "use_sq_consistency": False,
    "use_proto_noise": False,
    "use_feat_dropout": False,
}


def _make_config(**overrides):
    cfg = copy.deepcopy(_BASE_CONFIG)
    cfg.update(overrides)
    return cfg


ABLATION_CONFIGS = {
    "baseline": _make_config(),
    "multiscale_corr": _make_config(use_multiscale_correlation=True),
    "dice_loss": _make_config(use_dice_loss=True),
    "proto_noise": _make_config(use_proto_noise=True),
    "dice_multiscale": _make_config(
        use_dice_loss=True,
        use_multiscale_correlation=True,
    ),
    "dice_proto_noise": _make_config(
        use_dice_loss=True,
        use_proto_noise=True,
    ),
    "multiscale_proto": _make_config(
        use_multiscale_correlation=True,
        use_proto_noise=True,
    ),
    "multiscale_dice_proto": _make_config(
        use_dice_loss=True,
        use_multiscale_correlation=True,
        use_proto_noise=True,
    ),
}

args = None
logger = None
ablation_config = None


def parse_args():
    parser = argparse.ArgumentParser(description="STPC-COSeg test-only entry point")
    parser.add_argument(
        "--config",
        type=str,
        default="config/s3dis_ablation.yaml",
        help="Path to the YAML config file.",
    )
    parser.add_argument(
        "--exp",
        type=str,
        default="multiscale_dice_proto",
        choices=list(ABLATION_CONFIGS.keys()),
        help="Ablation variant used to construct the model.",
    )
    parser.add_argument(
        "--weight",
        type=str,
        required=True,
        help="Path to the trained checkpoint for testing.",
    )
    parser.add_argument(
        "opts",
        help="Additional config overrides, e.g. save_path ./test_out eval_split test.",
        default=None,
        nargs=argparse.REMAINDER,
    )

    parsed = parser.parse_args()
    cfg = config.load_cfg_from_cfg_file(parsed.config)
    if parsed.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, parsed.opts)

    cfg.exp_name = parsed.exp
    cfg.weight = parsed.weight
    cfg.test = True
    cfg.evaluate = True
    cfg.resume = ""
    cfg.pretrain_backbone = ""
    cfg.distributed = False
    cfg.multiprocessing_distributed = False
    cfg.ngpus_per_node = len(cfg.train_gpu)
    return cfg


def seed_everything(seed):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def is_main_process():
    return not getattr(args, "multiprocessing_distributed", False)


def worker_init_fn(worker_id):
    seed = getattr(args, "manual_seed", None)
    if seed is not None:
        random.seed(seed + worker_id)


def load_model_weight(model, weight_path):
    if not os.path.isfile(weight_path):
        raise FileNotFoundError("Checkpoint not found: {}".format(weight_path))

    checkpoint = torch.load(weight_path, map_location="cuda")
    if "state_dict" not in checkpoint:
        raise KeyError("Checkpoint does not contain a 'state_dict' field: {}".format(weight_path))

    state_dict = checkpoint["state_dict"]
    if not isinstance(model, torch.nn.parallel.DistributedDataParallel):
        state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}

    model.load_state_dict(state_dict, strict=True)
    return model


def build_test_dataset():
    val_transform = None
    forvis = bool(getattr(args, "forvis", 0))

    if args.data_name == "s3dis":
        if forvis:
            val_data = S3DIS_FSForVIS(
                split="test",
                data_root=args.data_root,
                voxel_size=args.voxel_size,
                voxel_max=args.voxel_max,
                transform=val_transform,
                cvfold=args.cvfold,
                num_episode=args.num_episode,
                n_way=args.n_way,
                k_shot=args.k_shot,
                n_queries=args.n_queries,
                target_class=args.target_class,
            )
        else:
            val_data = S3DIS_FS_TEST(
                split=args.eval_split,
                data_root=args.data_root,
                voxel_size=args.voxel_size,
                voxel_max=args.voxel_max,
                transform=val_transform,
                cvfold=args.cvfold,
                num_episode=args.num_episode,
                n_way=args.n_way,
                k_shot=args.k_shot,
                n_queries=args.n_queries,
                num_episode_per_comb=args.num_episode_per_comb,
            )
    elif args.data_name == "scannetv2":
        if Scannetv2_FS_TEST is None:
            raise ImportError("util.scannet_v2_fs.Scannetv2_FS_TEST is required for ScanNetv2 testing.")
        if forvis:
            raise ValueError("Visualization mode is currently implemented for S3DIS only.")
        val_data = Scannetv2_FS_TEST(
            split=args.eval_split,
            data_root=args.data_root,
            voxel_size=args.voxel_size,
            voxel_max=args.voxel_max,
            transform=val_transform,
            cvfold=args.cvfold,
            num_episode=args.num_episode,
            n_way=args.n_way,
            k_shot=args.k_shot,
            n_queries=args.n_queries,
            num_episode_per_comb=args.num_episode_per_comb,
        )
    else:
        raise ValueError("Unsupported dataset: {}".format(args.data_name))

    if not forvis:
        logger.info("Preparing test data...")
        val_data.prepare_test_data()

    valid_classes = list(val_data.classes)
    return val_data, valid_classes, forvis


def build_test_loader(val_data, forvis):
    return torch.utils.data.DataLoader(
        val_data,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        collate_fn=partial(collate_fn_limit_fs, include_scene_names=forvis),
    )


def validate(val_loader, model, valid_classes, forvis=False):
    logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    if forvis:
        target_class = val_loader.dataset.target_class
        pred_path = os.path.join(args.vis_save_path, target_class)
        os.makedirs(pred_path, exist_ok=True)

    torch.cuda.empty_cache()
    model.eval()
    end = time.time()

    for i, batch in enumerate(val_loader):
        if forvis:
            (
                support_x,
                support_y,
                support_offset,
                query_x,
                query_y,
                query_offset,
                sampled_classes,
                scene_names,
            ) = batch
        else:
            (
                support_x,
                support_y,
                support_offset,
                query_x,
                query_y,
                query_offset,
                sampled_classes,
            ) = batch

        data_time.update(time.time() - end)
        query_y = query_y.cuda(non_blocking=True)

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=args.use_amp):
                output, loss = model(
                    support_offset,
                    support_x,
                    support_y,
                    query_offset,
                    query_x,
                    query_y,
                    5,
                    sampled_classes=sampled_classes,
                )

        output = output.max(1)[1].squeeze(0)
        n = query_y.size(0)

        if getattr(args, "multiprocessing_distributed", False):
            loss *= n
            count = query_y.new_tensor([n], dtype=torch.long)
            dist.all_reduce(loss)
            dist.all_reduce(count)
            n = count.item()
            loss /= n

        intersection, union, target = evaluate_metric(
            output, query_y, sampled_classes, valid_classes, args.ignore_label
        )

        if forvis:
            query_name = scene_names[0]
            support_name = scene_names[1]
            save_dir = os.path.join(pred_path, "{}_{}".format(query_name, support_name))
            os.makedirs(save_dir, exist_ok=True)
            np.save(os.path.join(save_dir, "query.npy"), query_x.cpu().numpy())
            np.save(os.path.join(save_dir, "querylb.npy"), query_y.cpu().numpy())
            np.save(os.path.join(save_dir, "sup.npy"), support_x.cpu().numpy())
            np.save(os.path.join(save_dir, "suplb.npy"), support_y.cpu().numpy())
            np.save(os.path.join(save_dir, "pred.npy"), output.cpu().numpy())
            torch.cuda.empty_cache()

        if getattr(args, "multiprocessing_distributed", False):
            dist.all_reduce(intersection)
            dist.all_reduce(union)
            dist.all_reduce(target)

        intersection_meter.update(intersection.cpu().numpy())
        union_meter.update(union.cpu().numpy())
        target_meter.update(target.cpu().numpy())

        accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
        loss_meter.update(loss.item(), n)
        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % args.print_freq == 0:
            logger.info(
                "Test: [{}/{}] Data {data_time.val:.3f} ({data_time.avg:.3f}) "
                "Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                "Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) "
                "Accuracy {accuracy:.4f}.".format(
                    i + 1,
                    len(val_loader),
                    data_time=data_time,
                    batch_time=batch_time,
                    loss_meter=loss_meter,
                    accuracy=accuracy,
                )
            )

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    logger.info("Test result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.".format(mIoU, mAcc, allAcc))
    for class_index, class_id in enumerate(valid_classes):
        logger.info(
            "Class_{} Result: iou/accuracy {:.4f}/{:.4f}.".format(
                class_id, iou_class[class_index], accuracy_class[class_index]
            )
        )
    logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")
    return loss_meter.avg, mIoU, mAcc, allAcc


def save_test_result(loss_test, miou_test, macc_test, allacc_test):
    result = {
        "exp_name": args.exp_name,
        "ablation_config": ablation_config,
        "test_miou": float(miou_test),
        "test_macc": float(macc_test),
        "test_allacc": float(allacc_test),
        "test_loss": float(loss_test),
        "eval_split": args.eval_split,
        "data_name": args.data_name,
        "cvfold": args.cvfold,
        "n_way": args.n_way,
        "k_shot": args.k_shot,
        "num_episode": args.num_episode,
        "num_episode_per_comb": args.num_episode_per_comb,
        "weight": args.weight,
    }
    result_path = os.path.join(args.save_path, "test_result.json")
    with open(result_path, "w", encoding="utf-8") as result_file:
        json.dump(result, result_file, indent=2)
    logger.info("Test result saved to {}".format(result_path))


def main():
    global args, logger, ablation_config

    args = parse_args()
    ablation_config = ABLATION_CONFIGS[args.exp_name]
    args.save_path = os.path.join(args.save_path, args.exp_name)
    os.makedirs(args.save_path, exist_ok=True)

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu_id) for gpu_id in args.train_gpu)
    seed_everything(args.manual_seed)

    logger = get_logger(args.save_path)
    logger.info("=" * 60)
    logger.info("STPC-COSeg Test Only")
    logger.info("Experiment: {}".format(args.exp_name))
    logger.info("Weight: {}".format(args.weight))
    logger.info("Ablation Config: {}".format(ablation_config))
    logger.info("=" * 60)
    logger.info(args)

    model = COSegAblation(args, ablation_config=ablation_config).cuda()
    model = load_model_weight(model, args.weight)
    logger.info("Loaded checkpoint: {}".format(args.weight))
    logger.info("#Model parameters: {}".format(sum(param.nelement() for param in model.parameters())))

    val_data, valid_classes, forvis = build_test_dataset()
    val_loader = build_test_loader(val_data, forvis)
    loss_test, miou_test, macc_test, allacc_test = validate(
        val_loader, model, valid_classes, forvis=forvis
    )
    save_test_result(loss_test, miou_test, macc_test, allacc_test)
    logger.info("==> Test done!")


if __name__ == "__main__":
    main()
