import os
import glob
import pickle
import numpy as np
import pandas as pd
import random
from collections import Counter
import time
import multiprocessing as mp
from joblib import Parallel, delayed

from trackml.dataset import load_event,load_dataset
from trackml.randomize import shuffle_hits
from trackml.score import score_event

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

from sklearn.cluster import DBSCAN
from sklearn.neighbors import KDTree,BallTree
from sklearn.preprocessing import StandardScaler

from scipy.optimize import fsolve,curve_fit
from skopt import gp_minimize

from myutils import appendToIndex,makeFoldSliceInds,backGenWrapper,renumber


SEED = 1234
random.seed(SEED)
np.random.seed(SEED)


def assignColumns(df1,cols,colnames):
  for i,colname in enumerate(colnames):
    df1[colname] = cols[:,i]

# Cartesian to polar conversion
def toPolar(x,y):
  r = (x**2+y**2)**0.5
  phi = np.arctan2(y,x)
  return r,phi



'''
Concatenate multiple dataframes with labeled tracks into one dataframe.
The final dataframe will have unique labels for each track.
'''
def combineByTrackId(dflist):
  dfs = []
  currentMax = 0
  for df in dflist:
    if len(df)==0: continue
    df = df.copy()
    newTrackIds = renumber(df.track_id)+1+currentMax
    df['track_id'] = newTrackIds
    dfs.append(df)
    currentMax = np.max(newTrackIds)
  combined = pd.concat(dfs,axis='rows',sort=True)
  return combined

'''
Helper function to scale a subset of columns in dataframe
'''
def standardScale(df,colNames):
  vals = df[colNames].values
  vals = StandardScaler().fit_transform(vals)
  assignColumns(df,vals,colNames)



'''
This function creates a mask to keep only the first occurence
of hits that have the same volume,layer and module numbers,
but not the same x,y,z coordinates.  Since they are guaranteed
to be from different tracks.
'''
def maskSimilar(
  vlm_, # volume, layer and module numbers for points
  xyz_  # x,y,z coordinates for points
  ):
  rows,cols = vlm_.shape
  vlm_comp = (vlm_==vlm_.reshape(rows,1,cols)).all(axis=-1)
  xyz_comp = (xyz_!=xyz_.reshape(rows,1,cols)).any(axis=-1)
  comps = vlm_comp & xyz_comp
  comps = np.tril(comps,-1)
  mask = comps.any(axis=1)
  mask = ~mask
  return mask 

    


N_NEIGHBORS = 18  # Number of neighbors when doing track extension

dfcols0 = ['hit_id','particle_id','track_id','tokeep','dist','r0','volume_id', 'layer_id', 'module_id','np','nt','eps','dz','x','y','z']
dfcols1 = ['hit_id','particle_id','track_id','r0','r0_diff','ax_diff','ay_diff','volume_id', 'layer_id', 'module_id','np','nt','eps','dz','x','y','z']
dfcols2 = ['hit_id','particle_id','track_id','r0','r0_diff','ax_diff','ay_diff','volume_id', 'layer_id', 'module_id','np','nt','eps','dz','x','y','z']
dfcols3 = ['hit_id','particle_id','track_id','r0','volume_id', 'layer_id', 'module_id','np','nt','eps','dz','x','y','z']




  #####################
'''
Function to find tracks for a single event.
This function is intended to be called from multiple threads
to parallelize the processing of multiple events at once.
'''
def findTracks(
  origdf, # Dataframe with hit data
  eventNum # The event number
  ):
  pars = 0.035,0.035
  aWeight,r0Weight = 1,1  # Weights for dbscan parameters
  angRateEnd,angRateSteps = 0.004, 210 # Maximum angle rate for unrolling and number of steps in one direction.
  r0Thresh = 1  # Threshold value for r0 when doing track extension
  closestAdjust = 1.4 # Threshold value scaler for points closer to z origin, for track extension.
  fzThresh,czThresh = pars   # Threshold values for distances when doing track extension
  
     
  starttime = time.time()
  allLabeled = [] # Keep dataframes with all hits that have been assigned tracks
  df = origdf.copy()
  df['origz'] = df.z # Save original z coordinates to use with offsets.
  df['r'],df['phi'] = toPolar(df.x,df.y) # Convert hit coordinates to polar. 


  # Create angle rate steps to use in helix unrolling.
  dzs = []
  mStart,mEnd,mTotal = 0.00002,angRateEnd,angRateSteps  
  #mStart,mEnd,mTotal = 0.00001,angRateEnd,angRateSteps    
  m = (mEnd/mStart)**(1/(mTotal-1))
  for i in range(0,mTotal):
    ss = mStart*(m**i)
    dzs.append(ss)
    dzs.append(-ss)

  '''
  Make a dataframe to keep track of the maximum cluster size found by
  dbscan for a set of parameters.  This is used to save time by not doing
  another dbscan with the same set of parameters when possible.
  '''  
  mi = pd.MultiIndex.from_tuples([], names=('epsInd','zOffInd','angInd'))
  maxTrackSizes = pd.DataFrame(index=mi,columns=['maxts'])
  
  dfremaining = None # Variable to save the remaining hits after each iteration.
  
  for minSize in [9,7,5,4]: # Minimum track size that will be accepted as valid.
    #df['minSize'] = minSize    
    epss = np.linspace(0.0002,0.004,6) # Epsilon values to do dbscans.
    
    for epsInd in range(len(epss)):
      eps = epss[epsInd]
      #df['eps'] = eps
      #df['epsInd'] = epsInd      
      

      zOffsets = [0,3,-3,6,-6,9,-9,12,-12] 
      for zOffInd in range(len(zOffsets)): # Offsets for z coordinate.
        zOffset = zOffsets[zOffInd]
        #df['zOffInd'] = zOffInd        
        df['z'] = df.origz + zOffset
  
        
        for angInd in range(len(dzs)):  # Step through angle rates        
          iterKey = (epsInd,zOffInd,angInd)

          '''
          Skip doing dbscans if we know that we will not get clusters large
          enough to be used for making tracks from this set of parameters.
          '''
          mts = maxTrackSizes.reset_index()
          mts = mts.loc[(mts.epsInd>=epsInd) & (mts.zOffInd==zOffInd) & (mts.angInd==angInd)]
          if len(mts)>0:
            maxts = mts.maxts.min()
            if minSize>maxts:
              #print('skipping','minsize',minSize,'epsInd',epsInd,'zOffInd',zOffInd,'angInd',angInd)
              continue              
            
          #df['angInd'] = angInd              
          dz = dzs[angInd]
          #print('minsize',minSize,'epsInd',epsInd,'zOffInd',zOffInd,'angInd',angInd)
          time1 = time.time()          
          df.reset_index(drop=True,inplace=True)

          '''
          Make dbscan features.  We take the final unrolled angle
          as the tangent to the helix circle at the origin.  This
          is used to determine theta and r0 for the point.
          '''
          tangent = (df.phi + (dz * df.z))
          df['ax'] = np.cos(tangent)
          df['ay'] = np.sin(tangent)
          theta = tangent-(np.pi/2)
          df['inv_r0'] = (2*np.cos(df.phi-theta))/df.r
          df['r0'] = 1/df.inv_r0

          
          feats =  ['ax','ay','inv_r0'] 
       
          
          standardScale(df,feats)
          featVals = df[feats].values
          
          # Scale the features with weights
          featVals *= np.array([aWeight,aWeight,r0Weight]).reshape(1,-1)  

          cl = DBSCAN(eps=eps, min_samples=1, algorithm='ball_tree',leaf_size=5,n_jobs=-1)    
          labels = cl.fit_predict(featVals)          

          df['track_id'] = labels 
          

          ################
          # Make note of track sizes found by dbscan
          u,uinverse,ucounts = np.unique(df.track_id,return_inverse=True,return_counts=True)
          maxucounts = np.max(ucounts)
          maxTrackSizes.loc[iterKey,['maxts']] = maxucounts            
          
          if maxucounts<minSize:
            continue
          
          nt = ucounts[uinverse]
          df['nt'] = nt
          
          ###################
          # Order the unique track ids by decreasing size for track extension.
          utrackid = df.loc[df.nt>=minSize,['nt','track_id']].drop_duplicates('track_id').sort_values('nt',ascending=False).track_id.values
         
    
          vlm = df[['volume_id','layer_id','module_id']].values
          xyz = df[['x','y','z']].values
          z = df['z'].values
        
          featVals = df[feats].values
          tree = BallTree(featVals) # Set up for getting distances using parameters          
          
          # Flags to indicate if a point is still available for track assignment.
          available = np.full(len(featVals),True)
          
          '''
          Do track extension by looking at the closest neighbors on both
          ends of the current tracks.  The two ends are determined by the
          points on a track that are the closest and furthest from the
          origin along the z axis.
          '''  
          for tid in utrackid:
            dftrack = df[df.track_id==tid]
            dftrack1 = dftrack

            trackInds = dftrack1.index.values
            # Use only points that have not been processed yet yet.
            trackInds[available[trackInds]]
            trackLen = len(trackInds)
            if trackLen<4: continue
          
            if False:
              newTrackInds = trackInds
            else:
              # Determine the closest and furthest point along z axis
              dft = df.loc[trackInds]
              dft.sort_values('z',inplace=True)
              if abs(dft.z.iloc[-1])>abs(dft.z.iloc[0]):              
                furthestInd = dft.index[-1]
                closestInd = dft.index[0] 
              else:
                furthestInd = dft.index[0]
                closestInd = dft.index[-1]              
              furthestz,closestz = z[furthestInd],z[closestInd]
              
              # Set flag for track being on positive or negative side of z axis
              if furthestz>0: posZ = True
              else: posZ = False             
              
              '''
              Find neighbors for furthest point on track.
              These neighbors are candidates to extend the track.
              '''
              dist,ind = tree.query(featVals[furthestInd].reshape(1,-1),k=N_NEIGHBORS)
              dist,ind = dist[0],ind[0]            
              avInd = available[ind]
              if len(avInd)<4: continue
              dist,ind = dist[avInd],ind[avInd]
              dffz = df.loc[ind]
              dffz['dist'] = dist
              dffzr0 = dffz.r0.values
              #dffz['r0Ratio'] = np.abs((dffzr0-dffzr0[0])/dffzr0[0])              
              if posZ:
                dffz['tokeep'] = (dffz.z>=furthestz) & (dffz.dist<fzThresh)
              else:
                dffz['tokeep'] = (dffz.z<=furthestz) & (dffz.dist<fzThresh)
              #dffz = dffz[dfcols0]
              
              '''
              Find neighbors for closest point on track.
              These neighbors are candidates to extend the track
              '''
              dist,ind = tree.query(featVals[closestInd].reshape(1,-1),k=N_NEIGHBORS)
              dist,ind = dist[0],ind[0]            
              avInd = available[ind]
              if len(avInd)<4: continue
              dist,ind = dist[avInd],ind[avInd]
              dfcz = df.loc[ind]
              dfcz['dist'] = dist
              dfczr0 = dfcz.r0.values
              #dfcz['r0Ratio'] = np.abs((dfczr0-dfczr0[0])/dfczr0[0])               
              if posZ:
                dfcz['tokeep'] = (dfcz.z<=closestz*closestAdjust) & (dfcz.z>0) &(dfcz.dist<czThresh)
              else:
                dfcz['tokeep'] = (dfcz.z>=closestz*closestAdjust) & (dfcz.z<0) & (dfcz.dist<czThresh)
              #dfcz = dfcz[dfcols0]
              
              # Collect all indices for new extended track
              ind1 = np.concatenate([trackInds,dffz.index[dffz.tokeep==True],dfcz.index[dfcz.tokeep==True]])             
              
              dfoo = df.loc[ind1]
              dfoo = dfoo.drop_duplicates('hit_id')
              ind1 = dfoo.index
              
              '''
              Remove points with same volume,layer and module
              but different x,y,z.
              '''
              simMask = maskSimilar(vlm[ind1],xyz[ind1])
              ind1 = ind1[simMask]            
              #dfo = df.loc[ind1].copy()           
  

              ind2 = ind1#[keepMask]
              if len(ind2)==0: continue
  
              #dfn = dfo.loc[ind2]
  

                         
              newTrackInds = ind2
            
            
            if len(newTrackInds)<4: continue 
  
  
            df.loc[newTrackInds,['track_id']] = tid # Assign track id to all points
            available[newTrackInds] = False  # Mark points as unavailable
                  

          labeled = df.loc[~available]  

          allLabeled.append(labeled)
          df = df.loc[available]  # Points assigned to track are no longer in the pool of points.
          dfremaining = df.copy()
          u,ucounts = np.unique(df.track_id,return_counts=True)
          maxucounts = np.max(ucounts)   
          maxTrackSizes.loc[iterKey,['maxts']] = maxucounts          
         
    
  print('time',time.time()-starttime)    
  if dfremaining is None: dfremaining = df.copy()  
  allLabeled.append(dfremaining)
  #allLabeled = [al for al in allLabeled if len(al)>0]
  df = combineByTrackId(allLabeled)
    
    
  subdf = pd.DataFrame({"event_id":eventNum, "hit_id":df.hit_id, "track_id":df.track_id})
  subdf['event_id'] = subdf['event_id'].astype('int64')
  subdf['hit_id'] = subdf['hit_id'].astype('int64')
  subdf['track_id'] = subdf['track_id'].astype('int64')
  
  #score = score_event(valTruth, subdf)
  #print(pars,score)
  fpath = '../temp/testsub/sub' + eventNum + '.pickle'
  subdf.to_pickle(fpath)
  return subdf
  #####################

'''
Process all events in test set in parallel.
The resulting submission for each event is saved in a separate file.
When all events have been processed all of the event submissions
are combined into a single submission file.
'''
directory = '../input/test' 
filepaths = glob.glob('../input/test/*-hits.csv')
filenames = [os.path.split(fp)[1] for fp in filepaths]
eventNumbers = sorted([f.split('-')[0][5:] for f in filenames])  
filepaths_testsub = glob.glob('../temp/testsub/sub*.pickle')
filenames_testsub = [os.path.split(fp)[1] for fp in filepaths_testsub]
eventNumbers_testsub = [os.path.splitext(f)[0][3:] for f in filenames_testsub]
eventNumbers_test_todo = [en for en in eventNumbers if en not in eventNumbers_testsub]
print('Number of events to do:',len(eventNumbers_test_todo))
delayedCalls = []  # Create function calls for each event
for eventNumber in eventNumbers_test_todo:
  pathPrefix = '../input/test/event' + eventNumber
  hits, = load_event(pathPrefix,parts=['hits'])
  delayedCalls.append(delayed(findTracks)(hits,eventNumber))
# Call the functions in parallel
with Parallel(n_jobs=mp.cpu_count()) as par:
  parResults = par(delayedCalls)

# Combine all event submissions into one.
subdfs = []
for eventNumber in eventNumbers:
  fp = '../temp/testsub/sub' + eventNumber + '.pickle'  
  subdf = pd.read_pickle(fp)
  pathPrefix = '../input/test/event' + eventNumber
  hits, = load_event(pathPrefix,parts=['hits'])
  if len(subdf)!=len(hits): raise Exception('submission length does not match hits',eventNumber)
  if (subdf.event_id!=int(eventNumber)).any(): raise Exception('submission event does not match hits',eventNumber)
  
  subdfs.append(subdf)
print('making submission')
submissiondf = pd.concat(subdfs,axis='rows')
submissiondf.to_csv('../temp/submissiondf.csv.gz',index=False,compression='gzip')  
print('Done events')
########################
