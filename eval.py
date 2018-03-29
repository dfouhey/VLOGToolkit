#!/usr/bin/python
#David Fouhey
#Vlog benchmarking tools

import argparse
import pdb
import os
import sys
import numpy as np
import scipy.stats as stats
from sklearn.metrics import average_precision_score

####################
### Statistics stuff 
###

def _pctclip(X,p):
    """np.percentile but clipping to [0,100]"""
    p = [min(100,max(0,v)) for v in p]
    return np.percentile(X,p)

def _bciPercentileBias(replicates,allData,alpha):
    """Simple percentile CI with bias correction"""
    #The jackknife, the bootstrap and other resampling plans 
    #(See page 118)
    #https://statistics.stanford.edu/sites/default/files/BIO%2063.pdf
    alpha = alpha/2

    #get z for the bias and the desired alpha
    #note that ppf(0.5) = 0 so if the correction is positive, this pushes the 
    #distribution to the right; you can think of this as reintepreting the 
    #samples on either side of the median: ideally you should have 50% on 
    #either side

    z0 = stats.norm.ppf(np.mean(replicates <= allData))
    #this will be ~1.96 for a 95% CI
    za = stats.norm.ppf(alpha)

    higha, lowa = stats.norm.cdf(2*z0-za), stats.norm.cdf(2*z0+za)
    return _pctclip(replicates,[lowa*100,higha*100])

class GroupSample:
    """Sample according to groups rather than independently"""

    def __init__(self,G):
        #self.G maps key => inds with that key
        self.G = {}
        for i, gid in enumerate(G):
            if gid not in self.G: self.G[gid] = []
            self.G[gid].append(i)

        #helpers
        self.groups = self.G.keys() 
        self.npids = np.arange(0,len(G))
        self.numGroups = len(self.groups)
        
    def bootSamp(self):

        #first pick the groups
        groups = np.random.random_integers(0,self.numGroups-1,self.numGroups)
        #then return the sample
        return np.concatenate([self.npids[self.G[self.groups[g]]] for g in groups],axis=0)

####################
### Evaluation stuff 
###

def evaluateNBinary(Y,Yh,doBCI=False,bciGroup=None,bSampleCount=1000):
    assert Y.shape[0] == Yh.shape[0],"Inconsistent number of samples"
    assert Y.shape[1] == Yh.shape[1],"Inconsistent number of classes"
    N, K = Yh.shape[0], Yh.shape[1]
    APsAll = np.zeros((1,K+1))
    for j in range(K):

        keep = Y[:,j] != 0
        APsAll[0,j] = average_precision_score(Y[keep,j]>0,Yh[keep,j])

    APsAll[0,-1] = np.nanmean(APsAll[0,:K])

    if doBCI:
        np.random.seed(1)
        if bciGroup is None:
            bciGroup = np.arange(0,Y.shape[0])
        assert bciGroup.shape[0] == Y.shape[0]
        groupSampler = GroupSample(bciGroup)

        APsBCSample = np.zeros((bSampleCount,K+1))
        APsBC = np.zeros((2,K+1))
        for k in range(bSampleCount):
            #note: bsample can be larger or smaller than N
            bSample = groupSampler.bootSamp()
            Yb, Yhb = Y[bSample,:], Yh[bSample,:]
            for j in range(K):
                keep = Yb[:,j] != 0
                APsBCSample[k,j] = average_precision_score(Yb[keep,j],Yhb[keep,j])
        APsBCSample[:,-1] = np.nanmean(APsBCSample[:,:K],axis=1)
        for j in range(K+1):
            APsBC[:,j] = _bciPercentileBias(APsBCSample[:,j],APsAll[0,j],0.05)
       
        APsAll = np.concatenate([APsAll,APsBC],axis=0)

    return APsAll

def evaluateAccuracy(Y,Yh,doBCI=False,bciGroup=None,bSampleCount=10):
    N = Y.shape[0]
    acc = np.mean(Y==Yh) 

    if doBCI:
        np.random.seed(1)
        if bciGroup is None:
            bciGroup = np.arange(0,N)
        assert bciGroup.shape[0] == Y.shape[0]
        groupSampler = GroupSample(bciGroup)
        boots = np.zeros(bSampleCount)
        for k in range(bSampleCount):
            bSample = groupSampler.bootSamp()
            Yb, Ybh = Y[bSample], Yh[bSample]
            boots[k] = np.mean(Yb==Ybh)
        ci = _bciPercentileBias(boots,acc,0.05)
        return np.array([acc]+list(ci))
    else:
        return np.array([acc])


def evaluateTopN(Y,Yh,K=5,doBCI=False,bciGroup=None,bSampleCount=1000):
    #transform once into the order
    for i in range(Yh.shape[0]):
        Yh[i,:] = np.argsort(-Yh[i,:])
  
    correct = np.sum(np.equal(Yh[:,:K],np.reshape(Y,(-1,1))),axis=1)

    acc = np.mean(correct)

    if doBCI:
        N = Y.shape[0]
        np.random.seed(1)
        if bciGroup is None:
            bciGroup = np.arange(0,N)
        assert bciGroup.shape[0] == Y.shape[0]
        groupSampler = GroupSample(bciGroup)
        boots = np.zeros(bSampleCount)
        for k in range(bSampleCount):
            bSample = groupSampler.bootSamp()
            boots[k] = np.mean(correct[bSample])
        ci = _bciPercentileBias(boots,acc,0.05)
        return np.array([acc]+list(ci))
    else:        
        return np.array([acc])
        

#######################
### Data managing stuff 
###

def imgPathToVideoId(manifest,imageLines):
    """Given manifest and image lines, find the corresponding video"""
    #way faster to hash
    manifestH = {m:mi for mi,m in enumerate(manifest)}

    imageVideoId = np.zeros((len(imageLines),),dtype=np.int64)
    for li,l in enumerate(imageLines):
        vpath = l[:l.find("frame")]
        imageVideoId[li] = manifestH[vpath]
    return imageVideoId


if __name__ == "__main__":
    #N          - Number of videos in dataset (wc -l manifest.txt)
    #NTest      - Number of videos in test set (wc -l manifestTest.txt)
    #NImgTest   - Number of images in the test frame set (grep test hand_state/hand_state.txt  | wc -l)
    
    DATA_ROOT = "/home/dfouhey/dev/eco/cameraReadyRelease/pack/"


    parser = argparse.ArgumentParser("VLOG Evaluation Kit")
    parser.add_argument("--benchmark",type=str,default="",help="[hand_object|hand_touch|scene_category|scene_proxemic]",required=True)
    parser.add_argument("--predictions",type=str,default="",help="prediction file (.npy)",required=True)
    parser.add_argument("--dobci",action="store_true")
    parser.add_argument("--breakdown",type=str,default="",help="report results broken down by [scene_category|scene_proxemics]")
    args = parser.parse_args()

    scene_proxemic = np.load(DATA_ROOT+"/scene_proxemic/scene_proxemic_full.npy")

    predictionFile = args.predictions
    benchmarkId = args.benchmark
    doBCI = args.dobci
    breakdown = args.breakdown 


    manifest = file(DATA_ROOT+"/manifest.txt").read().strip().split("\n")
    uploaderId = np.loadtxt(DATA_ROOT+"/uploaderId.txt")

    #Load prediction
    prediction = None
    if predictionFile.endswith(".npy"):
        prediction = np.load(predictionFile)
    elif predictionFile.endswith(".txt"):
        prediction = np.loadtxt(predictionFile)
    else:
        print "Can't load prediction"
        sys.exit(1)


    #Load breakdown file
    #Per video id:
    #Per video categories (don't include -1)
    #Name for category
    breakdownVal = np.zeros((len(manifest),))
    breakdownCats = np.array([0])
    breakdownNames = ["All"]
    if breakdown == "scene_category":
        breakdownVal = np.load(DATA_ROOT+"/scene_category/scene_category.npy")
        breakdownCats = np.unique(breakdownVal[breakdownVal>=0])
        breakdownNames = file(DATA_ROOT+"/scene_category/category_label.txt").read().strip().split("\n")
        
    elif breakdown == "scene_proxemic":
        breakdownVal = np.load(DATA_ROOT+"/scene_proxemic/scene_proxemic_full.npy")
        breakdownCats = np.unique(breakdownVal[breakdownVal>=0])
        breakdownNames = file(DATA_ROOT+"/scene_proxemic/proxemic_label.txt").read().strip().split("\n")

    elif breakdown != "":
        print "No such breakdown", breakdown, "doing default"

    ##################################
    #From here it's just special cases

    if benchmarkId == "hand_object":
        ###Given Nx30 binary p(human touches object_j in video_i) predictions, compute AP, mAP in last col
        #load split info
        splitId = np.loadtxt(DATA_ROOT+"/splitId.txt")
        isTest = splitId == 0

        #load uploaderId for test
        uploaderIdTest = uploaderId[isTest]

        #load gt, and then subsample to test
        gtFn = DATA_ROOT+"/hand_object/hand_object.npy"
        YAll = np.load(gtFn)
        YTest = YAll[isTest,:]
        breakdownVal = breakdownVal[isTest]

        Yh = prediction
        
        for catI in range(len(breakdownNames)):
            use = np.equal(breakdownVal,breakdownCats[catI])
            print breakdownNames[catI]
            if args.dobci:
                pred = evaluateNBinary(YTest[use],prediction[use,:],doBCI=True,bciGroup=uploaderIdTest[use])
            else:
                pred = evaluateNBinary(YTest[use],prediction[use,:])

            for i in range(pred.shape[0]):
                for j in range(pred.shape[1]):
                    print "%.2f" % (pred[i,j]*100),
                print 

    elif benchmarkId == "hand_state":
        ###Given NImgTestx9 posteriors for images, compute accuracy

        #load the gt
        gtFn = DATA_ROOT+"/hand_state/hand_state.txt"
        lines = map(lambda l: l.split(),file(gtFn).read().strip().split("\n"))
        lines = [l for l in lines if l[1] == "test"]
        labels = np.array([int(l[2]) for l in lines])
        paths = [l[0] for l in lines]
        
        videoIds = imgPathToVideoId(manifest,paths)

        #get uploader ids for bci
        breakdownVal = breakdownVal[videoIds]
        uploaderId = uploaderId[videoIds]

        for catI in range(len(breakdownNames)):
            use = np.equal(breakdownVal,breakdownCats[catI])
            print breakdownNames[catI]
            if args.dobci:
                print evaluateAccuracy(labels[use],prediction[use],doBCI=True,bciGroup=uploaderId) 
            else:
                print evaluateAccuracy(labels[use],prediction[use])

    elif benchmarkId == "scene_category_places":
        ###Given Nx365 posteriors for the places 365 dataset, compute top-5 accuracy
       
        placesCats = {'bathroom':45,'bedroom':52,'dining-room':121,'kitchen':203,'living-room':215,'none-of-the-above':-1}
        catNames = file(DATA_ROOT+"/scene_category/category_label.txt").read().strip().split("\n")

        gtFn = DATA_ROOT+"/scene_category/scene_category_full.npy"
        Y = np.load(gtFn)
        Yh = prediction
        uploaderId = np.loadtxt(DATA_ROOT+"/uploaderId.txt")

        placeIds = map(placesCats.__getitem__,catNames)
        keep = np.zeros(Y.shape[0]).astype(np.bool)
        #transform to places ids 
        for i in range(Y.shape[0]):
            y = Y[i]
            if y < 0 or catNames[y] == "none-of-the-above": continue
            keep[i] = True
            Y[i] = placeIds[y] 

        for catI in range(len(breakdownNames)):
            keep2 = np.logical_and(keep,np.equal(breakdownVal,breakdownCats[catI]))
            print breakdownNames[catI]
            if args.dobci:
                res = evaluateTopN(Y[keep2],Yh[keep2,:],doBCI=True,bciGroup=uploaderId[keep2],K=5)
            else:
                res = evaluateTopN(Y[keep2],Yh[keep2,:],K=5)
            print res

    elif benchmarkId == "scene_category_places_5":
        ###Given Nx365 posteriors for the places 365 dataset, compute forced-choice accuracy
        placesCats = [52,203,45,215,121]

        catNames = file(DATA_ROOT+"/scene_category/category_label.txt").read().strip().split("\n")

        gtFn = DATA_ROOT+"/scene_category/scene_category_full.npy"
        Y = np.load(gtFn)
        Yh = prediction
        Yh = Yh[:,placesCats]
        Yh = np.argmax(Yh,axis=1)
        uploaderId = np.loadtxt(DATA_ROOT+"/uploaderId.txt")

        keep = np.zeros(Y.shape[0]).astype(np.bool)
        for i in range(Y.shape[0]):
            y = Y[i]
            if y < 0 or catNames[y] == "none-of-the-above": continue
            keep[i] = True

        for catI in range(len(breakdownNames)):
            keep2 = np.logical_and(keep,np.equal(breakdownVal,breakdownCats[catI]))
            print breakdownNames[catI]
            if args.dobci:
                res = evaluateAccuracy(Y[keep2],Yh[keep2],doBCI=True,bciGroup=uploaderId[keep2])
            else:
                res = evaluateAccuracy(Y[keep2],Yh[keep2])
            print res

    elif benchmarkId.startswith("scene_category"):
        ###Given NTestx6 predictions for the scene category, compute accuracy
        splitId = np.loadtxt(DATA_ROOT+"/splitId.txt")
        isTest = splitId == 0

        gtFn = DATA_ROOT+"/scene_category/scene_category_full.npy"
        Y = np.load(gtFn)
        Y = Y[isTest]
        Yh = prediction


        if benchmarkId == "scene_category_5":
            Yh = np.argmax(Yh[:,:5],axis=1)
            keep = np.logical_and(Y>=0,Y<5)
        else:
            Yh = np.argmax(Yh,axis=1)
            keep = Y >= 0

        breakdownVal = breakdownVal[isTest]
        uploaderId = uploaderId[isTest]

        for catI in range(len(breakdownNames)):
            keep2 = np.logical_and(keep,np.equal(breakdownVal,breakdownCats[catI]))
            print breakdownNames[catI]
            if args.dobci:
                res = evaluateAccuracy(Y[keep2],Yh[keep2],doBCI=True,bciGroup=uploaderId[keep2])
            else:
                res = evaluateAccuracy(Y[keep2],Yh[keep2])
            print res


    elif benchmarkId == "scene_proxemic":
        ###Given NTestx4 predictions for the scene category, compute accuracy
        splitId = np.loadtxt(DATA_ROOT+"/splitId.txt")
        isTest = splitId == 0

        gtFn = DATA_ROOT+"/scene_proxemic/scene_proxemic_full.npy"
        Y = np.load(gtFn)
        Y = Y[isTest]
        Yh = np.argmax(prediction,axis=1)

        keep = Y >= 0

        breakdownVal = breakdownVal[isTest]
        uploaderId = uploaderId[isTest]

        for catI in range(len(breakdownNames)):
            keep2 = np.logical_and(keep,np.equal(breakdownVal,breakdownCats[catI]))
            print breakdownNames[catI]
            if args.dobci:
                res = evaluateAccuracy(Y[keep2],Yh[keep2],doBCI=True,bciGroup=uploaderId[keep2])
            else:
                res = evaluateAccuracy(Y[keep2],Yh[keep2])
            print res


