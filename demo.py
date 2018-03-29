#!/usr/bin/python
#David Fouhey
#
#Sample code for loading VLOG
#
#This is released to clarify things although I hope the README is 
#sufficiently clear
#
# Assumes you have the data stored as 
#   .../a/b/c/v_asdfabc/000/clip.mp4
#           i.e.,
#   .../id(-3)/id(-2)/id(-1)/id/%03d/clip.mp4
#
# and have extracted the frames for the hand-labels as
#   .../frame_cache/a/b/c/v_asdfabc/000/frame_000061.jpg
#           i.e.,
#   .../frame_cache/id(-3)/id(-2)/id(-1)/id/%03d/frame_%06d.jpg
#

import os, sys
import numpy as np
import random

if not os.path.exists("DATA_ROOT"):
    print "Missing DATA_ROOT file!"
    print "Enter path of downloaded VLOG labels"
    print "(or type it into a file named DATA_ROOT)"
    path = raw_input(">")
    file("DATA_ROOT","w").write(path)
    print "Ok, try again"
    sys.exit(1)

if not os.path.exists("CLIP_ROOT"):
    print "Missing CLIP_ROOT file!"
    print "Enter path of clips"
    print "(or type it into a file named CLIP_ROOT)"
    path = raw_input(">")
    file("CLIP_ROOT","w").write(path)
    print "Ok, try again"
    sys.exit(1)

if not os.path.exists("FRAME_ROOT"):
    print "Missing FRAME_ROOT file!"
    print "Enter path of frames"
    print "(or type it into a file named FRAME_ROOT)"
    path = raw_input(">")
    file("FRAME_ROOT","w").write(path)
    print "Ok, try again"
    sys.exit(1)

DATA_ROOT = file("DATA_ROOT").read().strip()
CLIP_ROOT = file("CLIP_ROOT").read().strip()
FRAME_ROOT = file("FRAME_ROOT").read().strip()

readtxt = lambda fn: file(fn).read().strip().split("\n") 

#metadata
manifest = readtxt(DATA_ROOT+"/manifest.txt")
splits = map(int,readtxt(DATA_ROOT+"/splitId.txt"))
uids = map(int,readtxt(DATA_ROOT+"/uploaderId.txt"))

#Labels

#clip hand/object
#Nx30
LObj = np.load(DATA_ROOT+"/hand_object/hand_object.npy")
LObjName = readtxt(DATA_ROOT+"/hand_object/hand_object_labels.txt")

#clip scene type
#Nx1
LClass = np.load(DATA_ROOT+"/scene_category/scene_category_full.npy")
LClassName = readtxt(DATA_ROOT+"/scene_category/category_label.txt")

#clip proxemic type
#Nx1
LProx = np.load(DATA_ROOT+"/scene_proxemic/scene_proxemic_full.npy")
LProxName = readtxt(DATA_ROOT+"/scene_proxemic/proxemic_label.txt")

#frames
#M lines of the form:
#   frame phase label
LHand = readtxt(DATA_ROOT+"/hand_state/hand_state.txt")
LHandName = readtxt(DATA_ROOT+"/hand_state/hand_state_label.txt")

#hand boxes
#M lines of the form:
#   frame phase [x1 y1 x2 y2]*
LBox = readtxt(DATA_ROOT+"/hand_detect/hand_box_labels.txt")

#which data to show
verify = [False, False, False, False, True]

#Object contact
if verify[0]:
    random.seed(0)
    count = 0
    while count < 10:
        i = random.randint(0,len(manifest)-1)
        clipPath = CLIP_ROOT+"/"+manifest[i]+"/clip.mp4"
        if not os.path.exists(clipPath):
            continue
        l = LObj[i,:]
        print "%s:" % manifest[i],
        for j in range(l.size):
            if l[j]>0:
                print "(Y) %s" % (LObjName[j]),
            if l[j]==0:
                print "(I) %s" % (LObjName[j]),
        print 
        com = "cvlc %s" % clipPath
        os.system(com)
        count += 1

#Scene category/proxemics
for vi in range(1,3):
    if not verify[vi]: continue
    random.seed(vi+1)
    count = 0
    while count < 10:
        i = random.randint(0,len(manifest)-1)
        clipPath = CLIP_ROOT+"/"+manifest[i]+"/clip.mp4"
        if not os.path.exists(clipPath):
            continue
        L = LClass if vi == 1 else LProx
        LN = LClassName if vi == 1 else LProxName
        l = L[i]
        #unlabeled and inconclusive data is -1
        if l < 0: continue
        ln = LN[int(l)] if l >= 0 else "inconclusive/unlabeled"
        print "%s %d %s" % (manifest[i],int(l),ln)
        com = "cvlc %s" % clipPath
        os.system(com)
        count += 1


#Hand labels
if verify[3]:
    random.seed(1)
    count = 0
    while count < 10:
        l = LHand[random.randint(0,len(LHand)-1)].split()
        framePath = FRAME_ROOT+"/"+l[0]
        if not os.path.exists(framePath):
            continue
        print "Phase %s ; Label %s" % (l[1],LHandName[int(l[2])])
        os.system("eog %s" % framePath)
        count += 1
        

#Hand boxes
if verify[4]:
    import cv2
    random.seed(1)
    count = 0
    while count < 10:
        l = LBox[random.randint(0,len(LBox)-1)].split()
        frame, phase = l[0], l[1]
        s = 2
        I = cv2.imread(FRAME_ROOT+"/"+frame)
        h, w = I.shape[0], I.shape[1]
        while s < len(l):
            #bounding box is (x1,y1,x2,y2) in NORMALIZED coordinates 
            bb = map(float,l[s:s+4])
            cv2.rectangle(I, (int(bb[0]*w),int(bb[1]*h)), (int(bb[2]*w),int(bb[3]*h)),(0,0,255),5)
            s += 4
        cv2.imshow('asdf',I)
        cv2.waitKey(0)
        count += 1

