import torch.utils.data as data
import os
import os.path
from numpy.random import randint
from transforms import *
import matplotlib.pyplot as plt
import imageio

class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])  # -2 for rgb+flow or depth+flow

    @property
    def label(self):
        return int(self._data[2])

class TrearDataSet(data.Dataset):
    def __init__(self, root_path, list_file,
                 num_segments=3, new_length=1,
                 image_tmpl='img_{:04d}.jpg', transform=None,
                 random_shift=True, test_mode=False):

        self.root_path = root_path  
        self.list_file = list_file  # Train or test address
        self.num_segments = num_segments  # The number of segments which divides the video
        self.new_length = new_length  # new_length is set by input data set
        self.transform = transform
        self.random_shift = random_shift  # If it is a false, get index by _get_val_indices
        self.test_mode = test_mode
        self.image_tmpl = image_tmpl

        self._parse_list()

    def _load_image(self, directory, idx): 
        rgb_path = './data/Video_files/' + directory + '/color'
        depth_path = './data/Video_files/' + directory + '/depth'
        img1 = Image.open(rgb_path + '/color_{:04d}.jpeg'.format(idx))
            
        img2 = np.array(imageio.imread(depth_path + '/depth_{:04d}.png'.format(idx)))
        img2 = ((img2) / (img2.max()) * 255).astype(np.uint8)
        img2 = Image.fromarray(img2).convert('RGB')

        '''
        img2 = Image.open(depth_path + '/depth_{:04d}.png'.format(idx)).convert('RGB')
        print(img1.mode,type(img1))
        print(img2.mode,type(img2))
        plt.imshow(img2)
        plt.show()
        plt.imshow(img1)
        plt.show()
        #'''

        return [img1, img2]


    def _parse_list(self):
        self.video_list = [VideoRecord(x.strip().split(' ')) for x in open(self.list_file)]

    def _sample_indices(self, record):
        average_duration = (record.num_frames - self.new_length + 1) // self.num_segments

        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration, size=self.num_segments)
        elif record.num_frames > self.num_segments:
            offsets = np.sort(randint(record.num_frames - self.new_length + 1, size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))

        return offsets + 1

    def _get_val_indices(self, record):
        if record.num_frames > self.num_segments + self.new_length - 1:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1

    def _get_test_indices(self, record):
        tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
        offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])  # for appearance
        return offsets + 1

    def __getitem__(self, index):
        record = self.video_list[index]
        if not self.test_mode:
            segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
        else:
            segment_indices = self._get_test_indices(record)
        if len(segment_indices) == 0:
            print(record.path, record.num_frames)
            raise ValueError('segment_idx is null!')
        return self.get(record, segment_indices)

    def get(self, record, indices):
        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length):
                seg_imgs = self._load_image(record.path, p)  # origin
                images.extend(seg_imgs)
                
                if p < record.num_frames:
                    p += 1
                    
        process_data = self.transform(images)
        return process_data, record.label

    def __len__(self):
        return len(self.video_list)

