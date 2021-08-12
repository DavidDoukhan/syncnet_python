#!/usr/bin/python

import sys, time, os, pdb, argparse, pickle, subprocess, glob, cv2
import numpy as np
from shutil import rmtree

import scenedetect
from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
from scenedetect.frame_timecode import FrameTimecode
from scenedetect.stats_manager import StatsManager
from scenedetect.detectors import ContentDetector

from scipy.interpolate import interp1d
from scipy.io import wavfile
from scipy import signal

from detectors import S3FD
import time as t
import pandas
from SyncNetInstance import *
import os.path
from os import path
from utils.get_ava_active_speaker_performance import run_evaluation

parser = argparse.ArgumentParser(description = "FaceTracker");
parser.add_argument('--initial_model', type=str, default="data/syncnet_v2.model")
parser.add_argument('--batch_size', type=int, default='20', help='')
parser.add_argument('--vshift', type=int, default='15', help='')
parser.add_argument('--action',       type=str, default='')
parser.add_argument('--ava_dir',   type=str, default='../AVA_talknet_lite')
parser.add_argument('--data_dir',       type=str, default='../TMP/SYNCNET/WORK')
parser.add_argument('--result_dir',       type=str, default='../TMP/SYNCNET/RESULT')
parser.add_argument('--videofile',      type=str, default='',   help='Input video file')
parser.add_argument('--reference',      type=str, default='',   help='Video reference')
parser.add_argument('--facedet_scale',  type=float, default=0.25, help='Scale factor for face detection')
parser.add_argument('--crop_scale',     type=float, default=0.40, help='Scale bounding box')
parser.add_argument('--min_track',      type=int, default=100,  help='Minimum facetrack duration')
parser.add_argument('--frame_rate',     type=int, default=25,   help='Frame rate')
parser.add_argument('--num_failed_det', type=int, default=25,   help='Number of missed detections allowed before tracking is stopped');
parser.add_argument('--min_face_size',  type=int, default=100,  help='Minimum face size in pixels')

opt = parser.parse_args();

setattr(opt,'avi_dir',os.path.join(opt.data_dir,'pyavi'))
setattr(opt,'tmp_dir',os.path.join(opt.data_dir,'pytmp'))
setattr(opt,'work_dir',os.path.join(opt.data_dir,'pywork'))
setattr(opt,'crop_dir',os.path.join(opt.data_dir,'pycrop'))
setattr(opt,'frames_dir',os.path.join(opt.data_dir,'pyframes'))

# ========== ========== ========== ==========
# # IOU FUNCTION
# ========== ========== ========== ==========

def bb_intersection_over_union(boxA, boxB):
  
  xA = max(boxA[0], boxB[0])
  yA = max(boxA[1], boxB[1])
  xB = min(boxA[2], boxB[2])
  yB = min(boxA[3], boxB[3])
  interArea = max(0, xB - xA) * max(0, yB - yA)
  boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
  boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
  iou = interArea / float(boxAArea + boxBArea - interArea)
  return iou

# ========== ========== ========== ==========
# # FACE TRACKING
# ========== ========== ========== ==========
def track_shot(opt,scenefaces):

  iouThres  = 0.5     # Minimum IOU between consecutive face detections
  tracks    = []
  while True:
    track     = []
    for framefaces in scenefaces:
      for face in framefaces:
        if track == []:
          track.append(face)
          framefaces.remove(face)
        elif face['frame'] - track[-1]['frame'] <= opt.num_failed_det:
          iou = bb_intersection_over_union(face['bbox'], track[-1]['bbox'])
          if iou > iouThres:
            track.append(face)
            framefaces.remove(face)
            continue
        else:
          break

    if track == []:
      break
    elif len(track) > opt.min_track:
      framenum    = np.array([ f['frame'] for f in track ])
      bboxes      = np.array([np.array(f['bbox']) for f in track])
      frame_i   = np.arange(framenum[0],framenum[-1]+1)
      bboxes_i    = []
      for ij in range(0,4):
        interpfn  = interp1d(framenum, bboxes[:,ij])
        bboxes_i.append(interpfn(frame_i))
      bboxes_i  = np.stack(bboxes_i, axis=1)
      if max(np.mean(bboxes_i[:,2]-bboxes_i[:,0]), np.mean(bboxes_i[:,3]-bboxes_i[:,1])) > opt.min_face_size:
        tracks.append({'frame':frame_i,'bbox':bboxes_i})
  return tracks

# ========== ========== ========== ==========
# # VIDEO CROP AND SAVE
# ========== ========== ========== ==========
def crop_video(opt,track,cropfile):

  flist = glob.glob(os.path.join(opt.frames_dir,opt.reference,'*.jpg'))
  flist.sort()
  fourcc = cv2.VideoWriter_fourcc(*'XVID')
  vOut = cv2.VideoWriter(cropfile+'t.avi', fourcc, opt.frame_rate, (224,224))
  dets = {'x':[], 'y':[], 's':[]}
  for det in track['bbox']:
    dets['s'].append(max((det[3]-det[1]),(det[2]-det[0]))/2) 
    dets['y'].append((det[1]+det[3])/2) # crop center x 
    dets['x'].append((det[0]+det[2])/2) # crop center y

  # Smooth detections
  dets['s'] = signal.medfilt(dets['s'],kernel_size=13)   
  dets['x'] = signal.medfilt(dets['x'],kernel_size=13)
  dets['y'] = signal.medfilt(dets['y'],kernel_size=13)

  for fidx, frame in enumerate(track['frame']):
    cs  = opt.crop_scale
    bs  = dets['s'][fidx]   # Detection box size
    bsi = int(bs*(1+2*cs))  # Pad videos by this amount
    image = cv2.imread(flist[frame])
    frame = np.pad(image,((bsi,bsi),(bsi,bsi),(0,0)), 'constant', constant_values=(110,110))
    my  = dets['y'][fidx]+bsi  # BBox center Y
    mx  = dets['x'][fidx]+bsi  # BBox center X
    face = frame[int(my-bs):int(my+bs*(1+2*cs)),int(mx-bs*(1+cs)):int(mx+bs*(1+cs))]
    vOut.write(cv2.resize(face,(224,224)))

  audiotmp    = os.path.join(opt.tmp_dir,opt.reference,'audio.wav')
  audiostart  = (track['frame'][0])/opt.frame_rate
  audioend    = (track['frame'][-1]+1)/opt.frame_rate
  vOut.release()

  # ========== CROP AUDIO FILE ==========
  command = ("ffmpeg -y -i %s -ss %.3f -to %.3f %s" % (os.path.join(opt.avi_dir,opt.reference,'audio.wav'),audiostart,audioend,audiotmp)) 
  output = subprocess.call(command, shell=True, stdout=None)

  if output != 0:
    pdb.set_trace()

  sample_rate, audio = wavfile.read(audiotmp)

  # ========== COMBINE AUDIO AND VIDEO FILES ==========
  command = ("ffmpeg -y -i %st.avi -i %s -c:v copy -c:a copy %s.avi" % (cropfile,audiotmp,cropfile))
  output = subprocess.call(command, shell=True, stdout=None)

  if output != 0:
    pdb.set_trace()

  print('Written %s'%cropfile)
  os.remove(cropfile+'t.avi')
  print('Mean pos: x %.2f y %.2f s %.2f'%(np.mean(dets['x']),np.mean(dets['y']),np.mean(dets['s'])))
  return {'track':track, 'proc_track':dets}

# ========== ========== ========== ==========
# # FACE DETECTION
# ========== ========== ========== ==========
def inference_video(opt):

  DET = S3FD(device='cuda')
  flist = glob.glob(os.path.join(opt.frames_dir,opt.reference,'*.jpg'))
  flist.sort()
  dets = []
  for fidx, fname in enumerate(flist):
    start_time = time.time()
    image = cv2.imread(fname)
    image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    bboxes = DET.detect_faces(image_np, conf_th=0.9, scales=[opt.facedet_scale])
    dets.append([]);
    for bbox in bboxes:
      dets[-1].append({'frame':fidx, 'bbox':(bbox[:-1]).tolist(), 'conf':bbox[-1]})
    elapsed_time = time.time() - start_time
    print('%s-%05d; %d dets; %.2f Hz' % (os.path.join(opt.avi_dir,opt.reference,'video.avi'),fidx,len(dets[-1]),(1/elapsed_time)))
  savepath = os.path.join(opt.work_dir,opt.reference,'faces.pckl')
  with open(savepath, 'wb') as fil:
    pickle.dump(dets, fil)
  return dets

# ========== ========== ========== ==========
# # SCENE DETECTION
# ========== ========== ========== ==========

def scene_detect(opt):

  video_path = os.path.join(opt.avi_dir,opt.reference,'video.avi')
  video_manager = VideoManager([video_path])
  stats_manager = StatsManager()
  scene_manager = SceneManager(stats_manager)
  # Add ContentDetector algorithm (constructor takes detector options like threshold).
  scene_manager.add_detector(ContentDetector())
  base_timecode = video_manager.get_base_timecode()
  video_manager.set_downscale_factor()
  video_manager.start()
  scene_manager.detect_scenes(frame_source=video_manager)
  scene_list = scene_manager.get_scene_list(base_timecode)
  savepath = os.path.join(opt.work_dir,opt.reference,'scene.pckl')

  if scene_list == []:
    scene_list = [(video_manager.get_base_timecode(),video_manager.get_current_timecode())]

  with open(savepath, 'wb') as fil:
    pickle.dump(scene_list, fil)
  print('%s - scenes detected %d'%(os.path.join(opt.avi_dir,opt.reference,'video.avi'),len(scene_list)))
  return scene_list

def delete_dirs():

  # ========== DELETE EXISTING DIRECTORIES ==========
  if os.path.exists(os.path.join(opt.work_dir,opt.reference)):
    rmtree(os.path.join(opt.work_dir,opt.reference))

  if os.path.exists(os.path.join(opt.crop_dir,opt.reference)):
    rmtree(os.path.join(opt.crop_dir,opt.reference))

  if os.path.exists(os.path.join(opt.avi_dir,opt.reference)):
    rmtree(os.path.join(opt.avi_dir,opt.reference))

  if os.path.exists(os.path.join(opt.frames_dir,opt.reference)):
    rmtree(os.path.join(opt.frames_dir,opt.reference))

  if os.path.exists(os.path.join(opt.tmp_dir,opt.reference)):
    rmtree(os.path.join(opt.tmp_dir,opt.reference))

  if os.path.exists(opt.result_dir):
    rmtree(opt.result_dir)

def create_dirs():
  # ========== MAKE NEW DIRECTORIES ==========
  os.makedirs(os.path.join(opt.work_dir, opt.reference))
  os.makedirs(os.path.join(opt.crop_dir, opt.reference))
  os.makedirs(os.path.join(opt.avi_dir, opt.reference))
  os.makedirs(os.path.join(opt.frames_dir, opt.reference))
  os.makedirs(os.path.join(opt.tmp_dir, opt.reference))
  os.makedirs(opt.result_dir)

def run_pipeline():

  delete_dirs()
  create_dirs()

  # ========== CONVERT VIDEO AND EXTRACT FRAMES ==========
  file = os.path.join(opt.avi_dir,opt.reference,'video.avi')
  command = ("ffmpeg -y -i %s -qscale:v 2 -async 1 -r 25 %s" % (opt.videofile, file))
  output = subprocess.call(command, shell=True, stdout=None)

  command = ("ffmpeg -y -i %s -qscale:v 2 -threads 1 -f image2 %s" % (file, os.path.join(opt.frames_dir,opt.reference,'%06d.jpg')))
  output = subprocess.call(command, shell=True, stdout=None)

  command = ("ffmpeg -y -i %s -ac 1 -vn -acodec pcm_s16le -ar 16000 %s" % (os.path.join(opt.avi_dir,opt.reference,'video.avi'),os.path.join(opt.avi_dir,opt.reference,'audio.wav')))
  output = subprocess.call(command, shell=True, stdout=None)

  # ========== FACE DETECTION ==========
  faces = inference_video(opt) # creates faces.pckl

  # ========== SCENE DETECTION ==========
  scenes = scene_detect(opt) # creates scene.pckl

  # ========== FACE TRACKING ==========
  alltracks = []
  vidtracks = []
  for scene in scenes:
    scene_frame_start = scene[0].frame_num
    scene_frame_end = scene[1].frame_num
    if scene_frame_end - scene_frame_start >= opt.min_track :
      scene_faces = faces[scene_frame_start:scene_frame_end]
      tracks = track_shot(opt, scene_faces)
      alltracks.extend(tracks)

  # ========== FACE TRACK CROP ==========
  for ii, track in enumerate(alltracks):
    crop_file = os.path.join(opt.crop_dir,opt.reference,'%05d'%ii)
    vidtracks.append(crop_video(opt, track, crop_file))

  # ========== SAVE RESULTS ==========
  savepath = os.path.join(opt.work_dir,opt.reference,'tracks.pckl')
  with open(savepath, 'wb') as fil:
    pickle.dump(vidtracks, fil)
  rmtree(os.path.join(opt.tmp_dir,opt.reference))

def evaluate_network():

  start = t.time()
  s = SyncNetInstance();
  s.loadParameters(opt.initial_model);
  print("Model %s loaded." % opt.initial_model);

  f = open(opt.ava_dir + '/csv/video_ids.csv', 'r')
  video_names = [video_name[:-1] for video_name in f]
  #video_names = ['a5mEmM6w_ks.mkv']
  video_names = ['taubira1.mp4']
  #video_names = ['kMy-6RtoOVU.mkv']

  for video_name in video_names:
    delete_dirs()
    opt.reference = video_name.split('.')[0]
    create_dirs()
    path = opt.ava_dir + opt.video_dir + video_name
    opt.videofile = path
    start = t.time()
    generate_face_scene_pckl_files() # creates  *.avi files
    cropped_files = glob.glob(os.path.join(opt.crop_dir, opt.reference, '0*.avi'))
    cropped_files.sort()
    print('---------------- Cropping took {} s.'.format(t.time()-start))
    start = t.time()
    confidences = []
    for idx, cropped_file in enumerate(cropped_files):
      offset, conf, dist = s.evaluate(opt, cropped_video_file=cropped_file)
      conf1 = conf.tolist()
      confidences.append(conf1)
    print('---------------- evaluate {} s.'.format(t.time() - start))
    build_result_file(confidences)
    print('evaluate_network took {} s'.format(t.time()-start))

def generate_face_scene_pckl_files() :

    # ========== CONVERT VIDEO  ==========
    video_file = os.path.join(opt.avi_dir, opt.reference, 'video.avi')
    command = ("ffmpeg -y -i %s -qscale:v 2 -async 1 -r 25 %s" % ( opt.videofile, video_file))
    output = subprocess.call(command, shell=True, stdout=None)

    # ========== EXTRACT FRAMES ==========
    file = os.path.join(opt.frames_dir, opt.reference, '%06d.jpg')
    command = ("ffmpeg -y -i %s -qscale:v 2 -threads 1 -f image2 %s" % (video_file , file))
    output = subprocess.call(command, shell=True, stdout=None)

    # ========== extract audio ==========
    audio_file = os.path.join(opt.avi_dir, opt.reference, 'audio.wav')
    command = ("ffmpeg -y -i %s -ac 1 -vn -acodec pcm_s16le -ar 16000 %s" % (video_file, audio_file))
    output = subprocess.call(command, shell=True, stdout=None)

    # ========== FACE DETECTION ==========
    faces = inference_video(opt)

    # ========== SCENE DETECTION ==========
    scenes = scene_detect(opt)

    # ========== FACE TRACKING ==========
    alltracks = []
    for scene in scenes:
      scene_frame_start = scene[0].frame_num
      scene_frame_end = scene[1].frame_num
      if scene_frame_end - scene_frame_start >= opt.min_track:
        scene_faces = faces[scene_frame_start:scene_frame_end]
        tracks = track_shot(opt, scene_faces)
        alltracks.extend(tracks)

    # ========== FACE TRACK CROP ==========
    for ii, track in enumerate(alltracks):
      crop_file = os.path.join(opt.crop_dir, opt.reference, '%05d' % ii)
      crop_video(opt, track, crop_file)

def build_result_file(predScores):

  evalOrig = opt.ava_dir + '/csv/' + opt.ava_ref_file
  evalCsvSave = opt.result_dir + os.sep + 'val_res.csv'
  open(evalCsvSave, 'w').close() # creates evalCsvSave file if does not exist and empty it otherwise
  evalLines = open(evalOrig).read().splitlines()[1:]
  labels = pandas.Series(['SPEAKING_AUDIBLE' for line in evalLines])
  scores = pandas.Series(predScores)
  evalRes = pandas.read_csv(evalOrig)
  evalRes['score'] = scores
  evalRes['label'] = labels
  evalRes.drop(['label_id'], axis=1, inplace=True)
  evalRes.drop(['instance_id'], axis=1, inplace=True)
  evalRes.to_csv(evalCsvSave, index=False)
  cmd = "python -O utils/get_ava_active_speaker_performance.py -g %s -p %s " % (evalOrig, evalCsvSave)
  computation  = subprocess.run(cmd, shell=True, capture_output=True)
  mAP, precision, recall = run_pipeline.compute_ava_perf(evalOrig, evalCsvSave)
  return run_evaluation(evalOrig, evalCsvSave)
  print('mAP: ', mAP)

def compute_ava_perf2(val_orig, eval_csv_save):
  mAP = run_evaluation(val_orig, eval_csv_save)
  mAP, precision, recall = float(str(stdout).split(' ')[2][:5])
  return mAP, precision, recall

def compute_ava_perf(val_orig, eval_csv_save):
  cmd = "python -O utils/get_ava_active_speaker_performance.py -g %s -p %s " % (val_orig, eval_csv_save)
  stdout = subprocess.run(cmd, shell=True, capture_output=True, check=True).stdout
  mAP, precision, recall = float(str(stdout).split(' ')[2][:5])
  return mAP, precision, recall

if __name__ == "__main__":

    if (opt.action == 'evaluate'):
      opt.ava_dir = '../AVA_talknet'
      opt.video_dir = 'orig_videos/trainval/'  # can be test train, val
      opt.ava_ref_file = 'val_orig.csv'
      evaluate_network()
    elif (opt.action == 'evaluate_lite'):
      opt.ava_dir = '../AVA_talknet_lite/'
      opt.video_dir = 'orig_videos/test/' # can be test train, val
      opt.ava_ref_file = 'val_orig.csv'
      evaluate_network()
    elif (opt.action == 'run_pipeline'):
      run_pipeline()
    else:
      raise ValueError('action parameter should be either \'evaluate\' or \'run_pipeline\'')

