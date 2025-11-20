# python src/compute_loss.py --input /home/comp/f2256768/CPDD/data/images/real --limit 100

import torch
import yaml
import os
import sys
import glob
import copy
import argparse
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.helper import init_model, load_checkpoint
from src.masks.multiblock import MaskCollator as MBMaskCollator
from src.masks.utils import apply_masks
from src.utils.tensors import repeat_interleave_batch

import csv
import datetime

def compute_loss_for_images(
    image_paths,
    checkpoint_path,
    config_path,
    output_path='results.csv',
    device='cuda' if torch.cuda.is_available() else 'cpu',
    n_repeats=16,
    limit=None
):
    print(f"Using device: {device}")
    
    # 1. 加载配置
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        return

    with open(config_path, 'r') as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
    
    model_name = args['meta']['model_name']
    pred_depth = args['meta']['pred_depth']
    pred_emb_dim = args['meta']['pred_emb_dim']
    patch_size = args['mask']['patch_size']
    crop_size = args['data']['crop_size']
    
    print(f"Loading model: {model_name}, patch_size: {patch_size}, crop_size: {crop_size}")

    # 2. 初始化模型
    # 先在 CPU 上初始化以节省显存，避免 deepcopy 的峰值
    print("Initializing models on CPU...")
    encoder, predictor = init_model(
        device='cpu',
        patch_size=patch_size,
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_emb_dim=pred_emb_dim,
        model_name=model_name
    )
    target_encoder, _ = init_model(
        device='cpu',
        patch_size=patch_size,
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_emb_dim=pred_emb_dim,
        model_name=model_name
    )

    # 3. 加载权重
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # msg = encoder.load_state_dict(checkpoint['encoder'])
            # 处理 encoder 权重键名不匹配问题
            # 检查 checkpoint 中的 key 是否包含 'module.' 前缀 (DDP 训练产生的)
            encoder_state_dict = checkpoint['encoder']
            if any(k.startswith('module.') for k in encoder_state_dict.keys()):
                print("Detected 'module.' prefix in encoder checkpoint, removing it...")
                encoder_state_dict = {k.replace('module.', ''): v for k, v in encoder_state_dict.items()}
            
            msg = encoder.load_state_dict(encoder_state_dict)
            print(f"Loaded encoder with msg: {msg}")
            
            # msg = predictor.load_state_dict(checkpoint['predictor'])
            predictor_state_dict = checkpoint['predictor']
            if any(k.startswith('module.') for k in predictor_state_dict.keys()):
                print("Detected 'module.' prefix in predictor checkpoint, removing it...")
                predictor_state_dict = {k.replace('module.', ''): v for k, v in predictor_state_dict.items()}
            
            msg = predictor.load_state_dict(predictor_state_dict)
            print(f"Loaded predictor with msg: {msg}")
            
            # msg = target_encoder.load_state_dict(checkpoint['target_encoder'])
            target_encoder_state_dict = checkpoint['target_encoder']
            if any(k.startswith('module.') for k in target_encoder_state_dict.keys()):
                print("Detected 'module.' prefix in target_encoder checkpoint, removing it...")
                target_encoder_state_dict = {k.replace('module.', ''): v for k, v in target_encoder_state_dict.items()}
            
            msg = target_encoder.load_state_dict(target_encoder_state_dict)
            print(f"Loaded target_encoder with msg: {msg}")
            
            del checkpoint
            
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("WARNING: Continuing with random weights (results will be meaningless)...")
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}, using random initialization.")

    encoder.to(device)
    predictor.to(device)
    target_encoder.to(device)

    encoder.eval()
    predictor.eval()
    target_encoder.eval()

    # 4. 准备 Transform
    transform = transforms.Compose([
        transforms.Resize((crop_size, crop_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 5. 初始化 Mask Collator
    mask_collator = MBMaskCollator(
        input_size=crop_size,
        patch_size=patch_size,
        pred_mask_scale=args['mask']['pred_mask_scale'],
        enc_mask_scale=args['mask']['enc_mask_scale'],
        aspect_ratio=args['mask']['aspect_ratio'],
        nenc=args['mask']['num_enc_masks'],
        npred=args['mask']['num_pred_masks'],
        allow_overlap=args['mask']['allow_overlap'],
        min_keep=args['mask']['min_keep']
    )

    print(f"Processing {len(image_paths)} images...")
    print("-" * 60)
    print(f"{'Image Path':<40} | {'Average Loss':<15}")
    print("-" * 60)

    results = {}
    
    # 准备 CSV 写入
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    total_loss_sum = 0.0
    count = 0

    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Image Path', 'Average Loss'])

        for i, img_path in enumerate(image_paths):
            if limit is not None and i >= limit:
                print(f"Reached limit of {limit} images. Stopping.")
                break

            if not os.path.exists(img_path):
                print(f"Error: Image not found at {img_path}")
                continue

            try:
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img).unsqueeze(0).to(device) # [1, C, H, W]

                # 构造 Batch (重复 N 次以求平均)
                batch_imgs = img_tensor.repeat(n_repeats, 1, 1, 1)

                # 生成 Mask
                dummy_input = [0] * n_repeats
                masks_enc, masks_pred = mask_collator(dummy_input)
                
                masks_enc = [m.to(device) for m in masks_enc]
                masks_pred = [m.to(device) for m in masks_pred]

                # Forward & Loss
                with torch.no_grad():
                    h = target_encoder(batch_imgs)
                    h = F.layer_norm(h, (h.size(-1),))
                    h = apply_masks(h, masks_pred)
                    h = repeat_interleave_batch(h, n_repeats, repeat=len(masks_enc))

                    z = encoder(batch_imgs, masks_enc)
                    z = predictor(z, masks_enc, masks_pred)

                    loss = F.smooth_l1_loss(z, h)
                
                loss_val = loss.item()
                print(f"[{i+1}/{len(image_paths)}] {os.path.basename(img_path):<40} | {loss_val:.6f}")
                results[img_path] = loss_val
                
                total_loss_sum += loss_val
                count += 1

                writer.writerow([img_path, loss_val])
                csvfile.flush()

            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        
        if count > 0:
            global_avg = total_loss_sum / count
            print("-" * 60)
            print(f"Global Average Loss: {global_avg:.6f}")
            print("-" * 60)
            writer.writerow(['TOTAL_AVERAGE', global_avg])
        else:
            print("No images processed.")

    print(f"\nResults saved to {output_path}")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute I-JEPA loss for images')
    parser.add_argument('--input', type=str, 
                        default='/home/comp/f2256768/CPDD/data/image',
                        help='Path to image file or directory')
    parser.add_argument('--checkpoint', type=str, 
                        default='/home/comp/f2256768/CPDD/models/IN22K-vit.g.16-600e.pth.tar',
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, 
                        default='/home/comp/f2256768/CPDD/I-JEPA/configs/in22k_vitg16_ep44.yaml',
                        help='Path to config file')
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    default_output = os.path.join('src/results', f'loss_{timestamp}.csv')
    
    parser.add_argument('--output', type=str, default=default_output, help='Path to output CSV file')
    parser.add_argument('--repeats', type=int, default=16, help='Number of repeats for averaging loss')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of images to process')
    
    args = parser.parse_args()

    image_paths = []
    if os.path.isdir(args.input):
        for root, dirs, files in os.walk(args.input):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    image_paths.append(os.path.join(root, file))
        image_paths.sort()
    elif os.path.isfile(args.input):
        image_paths = [args.input]
    else:
        print(f"Error: Input path {args.input} does not exist")
        sys.exit(1)

    if not image_paths:
        print("No images found.")
        sys.exit(1)

    compute_loss_for_images(
        image_paths,
        args.checkpoint,
        args.config,
        output_path=args.output,
        n_repeats=args.repeats,
        limit=args.limit
    )
