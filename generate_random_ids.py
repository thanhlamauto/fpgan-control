#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  SPDX-License-Identifier: CC-BY-NC-4.0

import sys
import os
from pathlib import Path
from tqdm import tqdm
import argparse
import torch
import torchvision.transforms as transforms
import zipfile
import io


_PWD = Path(__file__).absolute().parent
sys.path.append(str(_PWD / 'src'))
from evaluation.inference_class import Inference


def duplicate_noise(noise, n):
    noise = [noise[i].expand(n, -1, -1, -1).clone() for i in range(len(noise))]
    return noise


@torch.no_grad()
def make_pid(inference_model, number_images_per_ids):
    device = inference_model.device
    id_latent = torch.randn(1, 256, device=device).repeat(number_images_per_ids, 1)
    id_noise = duplicate_noise(inference_model.model.module.make_noise(1), number_images_per_ids)
    pose_latent = torch.randn(number_images_per_ids, 256, device=device)

    latent = torch.cat([id_latent, pose_latent], dim=1)
    images, _ = inference_model.model([latent], noise=id_noise)
    images = images.mul(0.5).add(0.5).clamp(min=0., max=1.)

    image_list = torch.split(images, 1)

    return image_list


def generate_synth_ids(model_dir, save_path, number_of_ids, number_images_per_ids):
    to_pil = transforms.ToPILImage()
    inference_model = Inference(model_dir)
    instance_num = 0
    
    # Create zip file path
    zip_file_path = os.path.join(save_path, 'fpgan_output.zip')
    os.makedirs(os.path.dirname(zip_file_path) if os.path.dirname(zip_file_path) else '.', exist_ok=True)
    
    print(f'Generating {number_of_ids} IDs with {number_images_per_ids} images each...')
    print(f'Output will be saved to: {zip_file_path}')
    
    # Open zip file for writing
    with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for id_num in tqdm(range(number_of_ids), desc="Generating IDs"):
            id_name = 'ID_%07d' % id_num
            id_image_list = make_pid(inference_model, number_images_per_ids)
            
            for im_num, image in enumerate(id_image_list):
                image = image.cpu()
                image = to_pil(image[0])
                save_name = '%s_im%03d_instance%07d.png' % (id_name, im_num, instance_num)
                # Create path inside zip: ID_0000000/ID_0000000_im000_instance0000000.png
                zip_path = os.path.join(id_name, save_name)
                
                # Save image to zip
                img_buffer = io.BytesIO()
                image.save(img_buffer, format='PNG')
                zip_file.writestr(zip_path, img_buffer.getvalue())
                instance_num += 1
    
    print(f'\nGeneration complete!')
    print(f'Total images generated: {instance_num}')
    print(f'Zip file saved to: {zip_file_path}')
    print(f'Zip file size: {os.path.getsize(zip_file_path) / (1024**3):.2f} GB')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default=os.path.join(_PWD, 'models/id06fre20_fingers384_id_noise_same_id_idl005_posel000_large_pose_20230606-082209'))
    parser.add_argument('--save_path', type=str, default='where to save the images')
    parser.add_argument('--number_of_ids', type=int, default=100)
    parser.add_argument('--number_images_per_id', type=int, default=11)

    args = parser.parse_args()

    generate_synth_ids(
            args.model_dir,
            args.save_path,
            args.number_of_ids,
            args.number_images_per_id
        )