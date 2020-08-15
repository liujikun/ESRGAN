import torchvision.transforms as transforms
import yaml
from easydict import EasyDict as edict
import os
from os.path import join, exists
import numpy as np
import random
import cv2
from torch.utils import data
import torch
import sys

sys.path.append(os.getcwd())

from utils.img_utils import Distortion, Distortion_v2
import sys
from functools import reduce
import random
from PIL import Image
import subprocess
from utils.generate_mask_parts import *
from utils.warp_keypoints import *


def filter_data(wash_data: list()) -> dict():
    dict_tmp = {}
    for wash_data_items in wash_data:
        with open(wash_data_items, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                ref_list = line.split(' ')
                pic_id = ref_list[0].split('.')[0]
                pic_quality = ref_list[1]
                pic_orientation = ref_list[2]
                pic_gender = ref_list[3]
                pic_glass = ref_list[4]
                dict_tmp[pic_id] = True
                id_orientation = ''.join(
                    set(pic_id) & set(pic_orientation))
                id_gender = ''.join(
                    set(pic_id) & set(pic_gender))
                id_glass = ''.join(
                    set(pic_id) & set(pic_glass))

                if id_glass == '':
                    if pic_glass == '戴':
                        pic_glass = '有'
                    else:
                        pic_glass = '无'

                    id_glass = ''.join(
                        set(pic_id) & set(pic_glass))
                if id_orientation == '' or id_gender == '' or len(id_glass) != len(pic_glass) or pic_quality != '清晰':
                    dict_tmp[pic_id] = False
    return dict_tmp


WARP_COMMAND = "utils/keypoints_warp_dingdong/build/keypoints_warping --ranks {0} " \
               "--img_size {1} --src_img_path {2} --target_img_path {3} --output_path {4} --landmark_scale {5}"


# --------------------------------------
# Read and output a grid_tensor
# --------------------------------------
def save_and_get_flow(src_img_path, target_img_path, output_path,
        img_size=1024, ranks=0, landmark_scale=1024):
    whole_command = WARP_COMMAND.format(ranks, img_size, src_img_path, target_img_path,
            output_path, landmark_scale)
    random_id = int(subprocess.run(whole_command.split(" "), stdout=subprocess.PIPE).stdout)
    # Should get a output number from the command line
    npz_obj = np.load(os.path.join(output_path, "out_{}_{}_0.npz".format(ranks, random_id)))
    roll_angle = npz_obj["roll_angle"]
    grid = npz_obj["unclipped_flow"] # float32 (2 , 1024, 1024)
    grid = np.clip(grid, -1, 1)
    # grid = np.transpose(grid, (1, 2, 0))
    grid_tensor = torch.from_numpy(grid.copy())

    return grid_tensor, roll_angle

# def save_and_get_flow(src_img_path, target_img_path, output_path, img_size=1024, ranks=0):
#     whole_command = WARP_COMMAND.format(ranks, img_size, src_img_path, target_img_path, output_path)
#     random_id = int(subprocess.run(whole_command.split(" "), stdout=subprocess.PIPE).stdout)
#     # Should get a random number from the command line
#     grid = np.load(os.path.join(output_path, "out_{}_{}.npz".format(ranks, random_id)))[
#         "unclipped_flow"]  # float32 (2 , 1024, 1024)
#     grid = np.clip(grid, -1, 1)
#     grid_tensor = torch.from_numpy(grid.copy())
#
#     return grid_tensor


class Dataset_base(data.Dataset):
    def __init__(self, Transforms, config):
        # super(dataset_base, self).__init__()
        self.pair_data = {'data_path': config.pair_data.path,
                          'online_distortion': config.pair_data.distortion.online,
                          'ground_truth_list': config.pair_data.ground_truth_list,
                          'reference_list': config.pair_data.reference_list,
                          'region': config.pair_data.region,
                          }

        self.unpair_data = {'data_path': config.unpair_data.path,
                            'real_data_list': config.unpair_data.real_data_list,
                            }
        self.transforms = Transforms

        self.config = config
        self.ranks = 0
        # self.Distort = Distortion(config.pair_data.distortion)
        if config.Schedule_Method.image_size[0] > 512:
            ratio = (config.Schedule_Method.image_size[0] / 512) * 0.85
        else:
            ratio = 1
        self.Distort = Distortion_v2(config.pair_data.distortion, ratio)
        if config.pair_data.use_guide:
            self.Data, self.Gt, self.Ref, self.table = self.Prepare_Gdata()
        else:
            self.Data, self.Gt, self.table = self.Prepare_Data()
        self.transforms_to_tensor = transforms.Compose([
            transforms.ToPILImage(),

            transforms.Scale(size=self.Size,
                             interpolation=Image.BILINEAR),
            transforms.ToTensor(), ])

    def Prepare_Data(self):
        assert len(self.pair_data['ground_truth_list']) >= 1
        look_up_table = dict()
        groud_truth_list = list()
        real_data_list = list()
        # if self.config.Schedule_Method.Data_Distribution.gain:
        self.gain_list = dict()
        # else:
        # self.gain_list = None
        if self.config.pair_data.region.USE_REGION:
            suffix_pair = self.config.pair_data.region.suffix_region[
                self.config.pair_data.region.Name]
            self.Size = (self.config.Schedule_Method.Final_resolution *
                         2, self.config.Schedule_Method.Final_resolution)
        else:
            suffix_pair = [self.config.pair_data.suffix]
            self.Size = (self.config.Schedule_Method.Final_resolution,
                         self.config.Schedule_Method.Final_resolution)

        if self.config.unpair_data.region.USE_REGION:
            suffix_unpair = self.config.unpair_data.region.suffix_region[
                self.config.unpair_data.region.Name]
        else:
            suffix_unpair = [self.config.unpair_data.suffix]

        if self.config.pair_data.wash_data.Flag:
            dict_wash = filter_data(
                self.config.pair_data.wash_data.data_list)

        for grl in self.pair_data['ground_truth_list']:
            with open(grl, 'r') as f:
                for line in f.readlines():
                    line = line.strip()
                    if self.config.pair_data.wash_data.Flag:
                        line_tmp_0 = line[:-9]
                        if (line_tmp_0 not in dict_wash.keys()) or (not dict_wash[line_tmp_0]):
                            continue
                    for suffix in suffix_pair:
                        if self.config.pair_data.region.USE_REGION:
                            line_tmp = line[:-9] + suffix + '.png'
                        else:
                            line_tmp = line.split('.')[0] + suffix + '.png'
                        item = os.path.join(self.pair_data['data_path'],
                                            line_tmp)
                        look_up_table[item] = 'pair_data'
                        groud_truth_list += [item]

        if self.config.unpair_data.use_unpair_data:
            for rdl in self.unpair_data['real_data_list']:
                with open(rdl, 'r') as f:
                    for line in f.readlines():
                        line = line.strip()
                        for suffix in suffix_unpair:
                            line_tmp = line.split('.')[0] + \
                                       suffix + '.png'
                            item = os.path.join(self.pair_data['data_path'],
                                                line_tmp)
                        look_up_table[item] = 'unpair_data'
                        if self.gain_list is not None:
                            self.gain_list[item] = line_tmp.split('/')[-2]
                        real_data_list += [item]
        if self.config.PHASE == 'Test':
            groud_truth_list = random.sample(groud_truth_list, k=self.config.pair_data.sample)
            if len(real_data_list) > 0:
                real_data_list = random.sample(
                    real_data_list, k=self.config.unpair_data.sample)
        return groud_truth_list + real_data_list, groud_truth_list, look_up_table

    # prepare guid data
    def Prepare_Gdata(self):
        assert len(self.pair_data['ground_truth_list']) >= 1
        look_up_table = dict()
        groud_truth_list = {}
        groud_truth_path = []
        reference_list = {}
        real_data_list = {}
        # if self.config.Schedule_Method.Data_Distribution.gain:
        self.gain_list = dict()
        # else:
        # self.gain_list = None
        if self.config.pair_data.region.USE_REGION:
            suffix_pair = self.config.pair_data.region.suffix_region[
                self.config.pair_data.region.Name]
            self.Size = (self.config.Schedule_Method.Final_resolution *
                         2, self.config.Schedule_Method.Final_resolution)
        else:
            suffix_pair = [self.config.pair_data.suffix]
            self.Size = (self.config.Schedule_Method.Final_resolution,
                         self.config.Schedule_Method.Final_resolution)

        if self.config.unpair_data.region.USE_REGION:
            suffix_unpair = self.config.unpair_data.region.suffix_region[
                self.config.unpair_data.region.Name]
        else:
            suffix_unpair = [self.config.unpair_data.suffix]

        if self.config.pair_data.wash_data.Flag:
            dict_wash = filter_data(
                self.config.pair_data.wash_data.data_list)

        for grl in self.pair_data['ground_truth_list']:
            with open(grl, 'r') as f:
                for line in f.readlines():
                    line = line.strip()
                    if self.config.pair_data.wash_data.Flag and "newly_collect" not in grl:
                        line_tmp_0 = line[:-9]
                        if (line_tmp_0 not in dict_wash.keys()) or (not dict_wash[line_tmp_0]):
                            continue
                    for suffix in suffix_pair:
                        # added new collected data
                        if "newly_collect" in grl:
                            suffix = "_285_1024"
                        if self.config.pair_data.region.USE_REGION:
                            line_tmp = line[:-9] + suffix + '.png'
                        else:
                            # line_tmp = line.split('.')[0] + suffix + '.png'
                            line_tmp = line[:-4] + suffix + '.png'

                        item = os.path.join(self.pair_data['data_path'],
                                            line_tmp)
                        look_up_table[item] = 'pair_data'
                        if "newly_collect" in grl:
                            id = line.split('/')[-2]
                        else:
                            # old id
                            # id = line.split('/')[1].split('_')[0]
                            id = line.split('/')[0] + line.split('/')[1].split('_')[0]
                        if id not in groud_truth_list:
                            groud_truth_list[id] = []
                        groud_truth_list[id] += [item]
                        groud_truth_path += [item]

        for ref in self.pair_data['reference_list']:
            with open(ref, 'r') as f:
                for line in f.readlines():
                    line = line.strip()
                    if self.config.pair_data.wash_data.Flag and "newly_collect" not in ref:
                        line_tmp_0 = line[:-9]
                        if (line_tmp_0 not in dict_wash.keys()) or (not dict_wash[line_tmp_0]):
                            continue
                    for suffix in suffix_pair:
                        if "newly_collect" in ref:
                            suffix = "_285_1024"
                        if self.config.pair_data.region.USE_REGION:
                            line_tmp = line[:-9] + suffix + '.png'
                        else:
                            # line_tmp = line.split('.')[0] + suffix + '.png'
                            line_tmp = line[:-4] + suffix + '.png'
                        item = os.path.join(self.pair_data['data_path'],
                                            line_tmp)
                        look_up_table[item] = 'pair_data'
                        if "newly_collect" in ref:
                            id = line.split('/')[-2]
                        else:
                            # id = line.split('/')[1].split('_')[0]
                            id = line.split('/')[0] + line.split('/')[1].split('_')[0]
                        if id not in reference_list:
                            reference_list[id] = []
                        reference_list[id] += [item]
        # to do: use unpaired data
        if self.config.unpair_data.use_unpair_data:
            for rdl in self.unpair_data['real_data_list']:
                with open(rdl, 'r') as f:
                    for line in f.readlines():
                        line = line.strip()
                        for suffix in suffix_unpair:
                            line_tmp = line.split('.')[0] + \
                                       suffix + '.png'
                            item = os.path.join(self.pair_data['data_path'],
                                                line_tmp)
                        look_up_table[item] = 'unpair_data'
                        if self.gain_list is not None:
                            self.gain_list[item] = line_tmp.split('/')[-2]
                        real_data_list += [item]

        groud_truth_list_, reference_list_ = {}, {}
        if self.config.PHASE == 'Test':
            for i in range(self.config.pair_data.sample):
                key = random.sample(reference_list.keys(), k=1)[0]
                groud_truth_list_[key] = groud_truth_list[key]
                reference_list_[key] = reference_list[key]
            if len(real_data_list) > 0:
                real_data_list = random.sample(real_data_list, k=self.config.unpair_data.sample)
        else:
            groud_truth_list_, reference_list_ = groud_truth_list, reference_list
        return groud_truth_path, groud_truth_list_, reference_list_, look_up_table

    def Read_img(self, path, mask=False):
        if self.config.memcache.server:
            # sensetime memcache
            import mc
            server_list_config_file = self.config.memcache.server_path
            client_config_file = self.config.memcache.client_path
            # 然后获取一个mc对象
            mclient = mc.MemcachedClient.GetInstance(
                server_list_config_file, client_config_file)
            value = mc.pyvector()
            mclient.Get(path, value)
            value_str = mc.ConvertBuffer(value)
            img_array = np.frombuffer(value_str, np.uint8)
            if not mask:
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)  # RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)  # RGB
        else:
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if not mask:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def renew(self, transforms):
        self.transforms = transforms

    def __len__(self):
        if self.config.DEBUG and self.config.PHASE == 'Train':
            return self.config.DEBUG_LEN
        return len(self.Data)

    def __getitem__(self, idx):
        if self.config.Schedule_Method.Data_Distribution.Type == 'unpair':
            id_gt = random.sample(range(len(self.Gt)), k=1)[0]
        elif self.config.Schedule_Method.Data_Distribution.Type == 'pair':
            if self.config.pair_data.use_guide:
                sample_id = random.sample(self.Ref.keys(), k=1)[0]
                # random select id for GT image, chance = 1/4
                if self.config.pair_data.cross_id:
                    chance = random.randint(0, 4)
                    if chance == 0:
                        temp_id = random.sample(self.Ref.keys(), k=2)
                        ref_id = temp_id[0] if temp_id[0] != sample_id else temp_id[1]
                    else:
                        ref_id = sample_id
                else:
                    ref_id = sample_id
                path_ref = random.sample(self.Ref[ref_id], k=1)
                path_gt = random.sample(self.Gt[sample_id], k=1)
                # add mask path
                img_path = path_gt[0].split('/')[-1]
                if self.config.pair_data.mask_path != "None":
                    path_mask = self.config.pair_data.mask_path + img_path.split('.')[0] + '_mask.' + \
                                img_path.split('.')[-1]
                    # replace the mask path
                    if '1024' in self.config.pair_data.suffix:
                        path_mask = path_mask.replace('1024', '512')
                else:
                    path_mask = None

                if self.config.pair_data.skinseg_path != "None":
                    path_skin_gt = ''.join(path_gt).replace('.', '_skinseg.')
                    path_skin_ref = ''.join(path_ref).replace('.', '_skinseg.')
            else:
                id_gt = idx

        try:
            gt_image = self.Read_img(''.join(path_gt))
            ref_image = self.Read_img(''.join(path_ref))
            d_image = gt_image
            # process seg mask
            if self.config.pair_data.mask_path != "None":
                if "1440" in ''.join(path_mask):
                    seg_mask = gt_image
                else:
                    # seg_mask = self.Read_img(''.join(path_mask), mask=True)
                    seg_mask = gt_image

            # if use skin seg mask
            if self.config.Schedule_Method.Generator.Loss_Method.mask_skin:
                mix_process = False
                try:
                    d_skinseg = self.Read_img(''.join(path_skin_gt), mask=True)
                    ref_skinseg = self.Read_img(''.join(path_skin_ref), mask=True)
                except:
                    mix_process = True

        except Exception as e:
            print(str(e))
            print('gt_image: {} Or mask_image: {} not exit!'.format(path_gt,
                                                                    path_mask))
            return None

        Size = (gt_image.shape[1], gt_image.shape[0])
        # if self.Size == (self.config.Schedule_Method.image_size[0], self.config.Schedule_Method.image_size[1]):
        #    gt_image = cv2.resize(gt_image, (self.config.Schedule_Method.image_size[0], self.config.Schedule_Method.image_size[1]), interpolation=cv2.INTER_AREA)
        #    ref_image = cv2.resize(ref_image, (self.config.Schedule_Method.image_size[0], self.config.Schedule_Method.image_size[1]), interpolation=cv2.INTER_AREA)
        #    if (gt_image.shape[1], gt_image.shape[0]) != self.Size or (d_image.shape[1], d_image.shape[0]) != self.Size:
        #        return None

        if self.table[self.Data[idx]] == 'pair_data':
            # d_image = self.Distort._op_defocus(d_image)
            d_image, gain = self.Distort.Distort_random_v3(d_image, Size)
            # chance = np.random.randint(0, 2)
            chance = 1
            if seg_mask is not None and chance == 0:
                seg_mask = np.uint8(seg_mask / 255)
                seg_mask_ = cv2.GaussianBlur(seg_mask, (21, 21), 0)
                seg_mask_ = np.expand_dims(seg_mask_, axis=2)
                # seg_mask_ = np.repeat(seg_mask_., 3, axis=0)
                d_image = d_image * seg_mask_ + (1 - seg_mask_) * gt_image
            gain = 1
        else:
            gain = int(self.gain_list[self.Data[idx]][0])
        distortion_index = np.random.uniform(0.0, 1.0, 1)

        # ref_path_full and gt_path_full are the full paths of images
        if self.config.Schedule_Method.Warp.USE_KEYPOINT:
            try:
                grid_tensor, roll_angle = save_and_get_flow(''.join(path_ref), ''.join(path_gt),
                                                self.config.temp_path,
                                                img_size=self.Size[0],
                                                ranks=self.ranks,
                                                landmark_scale=1024
                                                )
                # grid_tensor = get_dense_flow(''.join(path_ref).replace('png', 'txt'),
                #                              ''.join(path_gt).replace('png', 'txt'), image_size=self.Size[0])
                grid_tensor[torch.isnan(grid_tensor)] = 0.0
            except Exception as e:
                print(str(e))
                print("Cannot get grid tensor")
                return None
        else:
            grid_tensor = torch.ones(1)
            roll_angle = 0.

        if roll_angle != 0.0:
            ref_w, ref_h = ref_image.shape[:2]
            M = cv2.getRotationMatrix2D((int(ref_w/2), int(ref_h/2)), roll_angle, 1)
            ref_image = cv2.warpAffine(ref_image, M, (ref_w, ref_h))

        # modify the transforms operation
        sample = {'d_image': self.transforms(d_image),
                  'gt_image': self.transforms(gt_image),
                  'gt_mask': self.transforms(seg_mask),
                  'ref_image': self.transforms(ref_image),
                  'distortion_index': distortion_index,
                  'gain': gain,
                  'grid': grid_tensor
                  }

        # use landmark mask
        # change it to sample version: landmark 14%,  face 90%,
        if self.config.Schedule_Method.Generator.Loss_Method.mask_parts:
            path_gt = "".join(path_gt)
            path_ref = "".join(path_ref)
            if mix_process:
                if "newly_collect" in path_gt:
                    path_gt = path_gt.replace('crop_0_285_1024', "285_0")
                if "newly_collect" in path_ref:
                    path_ref = path_ref.replace('crop_0_285_1024', "285_0")
            if self.config.Schedule_Method.Generator.Loss_Method.use_ref_mask:
                landmark_file = path_ref.replace('png', 'txt')
            else:
                landmark_file = path_gt.replace('png', 'txt')
            rel = self.config.Schedule_Method.image_size[0]
            chance = random.randint(0, 5)
            file_exist = os.path.isfile(landmark_file)
            if chance <= 3 or not file_exist:
                mask_parts = torch.ones((1, rel, rel))  # use full image
            elif chance == 4:
                mask_parts = get_mask_from_landmarks(landmark_file, mask_size=rel, use_face=False)  # use landmarks
            else:
                mask_parts = get_mask_from_landmarks(landmark_file, mask_size=rel, use_face=True)  # use face

            sample.update({'mask_parts': mask_parts})

        # add skin seg masks
        if self.config.Schedule_Method.Generator.Loss_Method.mask_skin:
            # if skinseg mask is not exist, use landmark
            path_gt = "".join(path_gt)
            path_ref = "".join(path_ref)
            if mix_process:
                if "newly_collect" in path_gt:
                    path_gt = path_gt.replace('crop_0_285_1024', "285_0")
                if "newly_collect" in path_ref:
                    path_ref = path_ref.replace('crop_0_285_1024', "285_0")
                
                d_skinseg = get_mask_from_landmarks(path_gt.replace('png', 'txt'),
                                                    mask_size=self.config.Schedule_Method.image_size[0])
                ref_skinseg = get_mask_from_landmarks(path_ref.replace('png', 'txt'),
                                                      mask_size=self.config.Schedule_Method.image_size[0])
            sample.update({'d_skinseg': self.transforms_to_tensor(d_skinseg),
                           'ref_skinseg': self.transforms_to_tensor(ref_skinseg)})

        # add landmark marks as model input
        if self.config.Schedule_Method.use_landmarks:
            d_landmark = get_instance_map(''.join(path_gt).replace('png', 'txt'),
                                          mask_size=self.config.Schedule_Method.image_size[0])
            ref_landmark = get_instance_map(''.join(path_ref).replace('png', 'txt'),
                                            mask_size=self.config.Schedule_Method.image_size[0])
            sample.update({'d_landmark': d_landmark,
                           'ref_landmark': ref_landmark})
        return sample
