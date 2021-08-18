#!/usr/bin/python
#-*- coding: utf-8 -*-
# Video 25 FPS, Audio 16000HZ

import torch
import numpy
import time, pdb, argparse, subprocess, os, math, glob
import cv2
import python_speech_features

from scipy import signal
from scipy.io import wavfile
from SyncNetModel import *
from shutil import rmtree


# ==================== Get OFFSET ====================

def calc_pdist(image_features, audio_features, vshift=10):
    
    win_size = vshift*2+1
    audio_features2 = torch.nn.functional.pad(audio_features, (0, 0, vshift, vshift))
    dists = []
    for i in range(0, len(image_features)):
        dist = torch.nn.functional.pairwise_distance(image_features[[i], :].repeat(win_size, 1), audio_features2[i:i + win_size, :])
        dists.append(dist)
    return dists

# ==================== MAIN DEF ====================

class SyncNetInstance(torch.nn.Module):

    def __init__(self, dropout = 0, num_layers_in_fc_layers = 1024):

        super(SyncNetInstance, self).__init__();
        self.model = S(num_layers_in_fc_layers = num_layers_in_fc_layers).cuda();

    def evaluate(self, opt, cropped_video_file):

        self.model.eval();
        # Convert files
        if os.path.exists(os.path.join(opt.tmp_dir,opt.reference)):
          rmtree(os.path.join(opt.tmp_dir,opt.reference))
        os.makedirs(os.path.join(opt.tmp_dir,opt.reference))

        # ========== EXTRACT FRAMES ==========
        command = ("ffmpeg -y -i %s -threads 1 -f image2 %s" % (cropped_video_file, os.path.join(opt.tmp_dir, opt.reference, '%06d.jpg')))
        output = subprocess.call(command, shell=True, stdout=None)

        # ========== EXTRACT audio ==========
        command = ("ffmpeg -y -i %s -async 1 -ac 1 -vn -acodec pcm_s16le -ar 16000 %s" % (cropped_video_file, os.path.join(opt.tmp_dir, opt.reference, 'audio.wav')))
        output = subprocess.call(command, shell=True, stdout=None)
        
        # Load video
        images = []
        file_images = glob.glob(os.path.join(opt.tmp_dir,opt.reference,'*.jpg'))
        file_images.sort()
        for file_image in file_images:
            images.append(cv2.imread(file_image))

        im = numpy.stack(images,axis=3)
        im = numpy.expand_dims(im,axis=0)
        im = numpy.transpose(im,(0,3,4,1,2))
        image_tv = torch.autograd.Variable(torch.from_numpy(im.astype(float)).float())
        print('End extracting frames')

        # Load audio
        sample_rate, audio = wavfile.read(os.path.join(opt.tmp_dir,opt.reference,'audio.wav'))
        mfcc = zip(*python_speech_features.mfcc(audio,sample_rate))
        mfcc = numpy.stack([numpy.array(i) for i in mfcc])
        cc = numpy.expand_dims(numpy.expand_dims(mfcc,axis=0),axis=0)
        cct = torch.autograd.Variable(torch.from_numpy(cc.astype(float)).float())

        # Check audio and video input length
        img_nbr = len(images)
        audio_nbr = len(audio)
        if (float(audio_nbr)/16000) != (float(img_nbr)/25) :
            print("WARNING: Audio (%.4fs) and video (%.4fs) lengths are different."%(float(audio_nbr)/16000,float(img_nbr)/25))

        min_length = min(img_nbr,math.floor(audio_nbr/640))
        # Generate video and audio feats
        lastframe = min_length-5
        lastframe = min_length # ADDED by ITHIU
        image_features = []
        cc_features = []
        tS = time.time()
        for i in range(0, lastframe, opt.batch_size):
            im_batch = []
            end = min(lastframe, i + opt.batch_size)
            for vframe in range(i, end):
                if (vframe > lastframe - 5):
                    #offset = 5 - (lastframe - vframe)
                    #item = image_tv[:, :, vframe - offset:lastframe, :, :]
                    item = image_tv[:, :, lastframe - 5:lastframe, :, :]
                else:
                    item = image_tv[:, :, vframe:vframe + 5, :, :]
                im_batch.append(item)

            im_in = torch.cat(im_batch,0)
            im_out  = self.model.forward_lip(im_in.cuda());
            image_features.append(im_out.data.cpu())

            cc_batch = []
            end = min(lastframe, i + opt.batch_size)
            for vframe in range(i, end):
                if (vframe > end - 20):
                    #offset = 20 - (lastframe - vframe)
                    real_end = cct.shape[3]
                    item = cct[:, :, :, real_end - 20 : real_end]
                    #item = cct[:, :, :, lastframe * 4 - 20: lastframe * 4]
                else :
                    item = cct[:, :, :, vframe*4 : vframe*4 + 20]
                cc_batch.append(item)
            #cc_batch = [ cct[:,:,:,vframe*4:vframe*4+20] for vframe in range(i,min(lastframe,i+opt.batch_size)) ]
            cc_in = torch.cat(cc_batch,0)
            cc_out  = self.model.forward_aud(cc_in.cuda())
            cc_features.append(cc_out.data.cpu())

        image_features = torch.cat(image_features,0)
        cc_features = torch.cat(cc_features,0)

        # Compute offset
        print('Compute time %.3f sec.' % (time.time()-tS))
        dists = calc_pdist(image_features,cc_features,vshift=opt.vshift)
        mdist = torch.mean(torch.stack(dists,1),1)

        minval, minidx = torch.min(mdist,0)
        offset = opt.vshift-minidx
        conf   = torch.median(mdist) - minval

        fdist   = numpy.stack([dist[minidx].numpy() for dist in dists])
        # fdist   = numpy.pad(fdist, (3,3), 'constant', constant_values=15)
        fconf   = torch.median(mdist).numpy() - fdist
        fconfm  = signal.medfilt(fconf,kernel_size=9)
        
        numpy.set_printoptions(formatter={'float': '{: 0.3f}'.format})
        print('Framewise conf: ')
        print(fconfm)
        print('AV offset: \t%d \nMin dist: \t%.3f\nConfidence: \t%.3f' % (offset,minval,conf))

        dists_npy = numpy.array([ dist.numpy() for dist in dists ])
        return offset.numpy(), conf.numpy(), dists_npy

    def extract_feature(self, opt, videofile):

        self.model.eval();
        cap = cv2.VideoCapture(videofile)# Load video
        frame_num = 1;
        images = []
        while frame_num:
            frame_num += 1
            ret, image = cap.read()
            if ret == 0:
                break

            images.append(image)

        im = numpy.stack(images,axis=3)
        im = numpy.expand_dims(im,axis=0)
        im = numpy.transpose(im,(0,3,4,1,2))

        imtv = torch.autograd.Variable(torch.from_numpy(im.astype(float)).float())
        
        # ========== ==========
        # Generate video feats
        # ========== ==========

        lastframe = len(images)-4
        im_feat = []

        tS = time.time()
        for i in range(0,lastframe,opt.batch_size):
            
            im_batch = [ imtv[:,:,vframe:vframe+5,:,:] for vframe in range(i,min(lastframe,i+opt.batch_size)) ]
            im_in = torch.cat(im_batch,0)
            im_out  = self.model.forward_lipfeat(im_in.cuda());
            im_feat.append(im_out.data.cpu())

        im_feat = torch.cat(im_feat,0)

        # ========== ==========
        # Compute offset
        # ========== ==========
            
        print('Compute time %.3f sec.' % (time.time()-tS))

        return im_feat


    def loadParameters(self, path):

        loaded_state = torch.load(path, map_location=lambda storage, loc: storage);
        self_state = self.model.state_dict();
        for name, param in loaded_state.items():
            self_state[name].copy_(param);
