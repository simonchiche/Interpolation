'''
Performs an interpolation of the peak-to-peak electric field at 6 antenna positions for each shower in database.
Check interpolation efficiency.

run ant_trig_db [argv1]

argv1: str
    path+name of shower database
'''
import os
import logging   #for...you guessed it...logging
import numpy as np
import matplotlib.pyplot as plt
from zhairesppath import GetZHSEffectiveactionIndex, GetEffectiveactionIndex


import hdf5fileinout as hdf5io
from StarshapeInterpolation3D import do_interpolation_hdf5

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 14}

plt.rc('font', **font)

usetrace='efield'
DISPLAY=False





EfieldTraces = []


try:
  InputFilename = "./Stshp_Proton_0.681_53.3_0.0_1" 
  print("InputFileName:"+InputFilename)
  
  OutputFilename = InputFilename + '.Interpolated3D.'+str(usetrace)+'.hdf5'
  #
  if(os.path.isfile(OutputFilename)):
    print("already computed")
    #sys.exit()
  
  CurrentRunInfo=hdf5io.GetRunInfo(InputFilename)
  CurrentEventName=hdf5io.GetEventName(CurrentRunInfo,0) #using the first event of each file (there is only one for now)
  CurrentAntennaInfo=hdf5io.GetAntennaInfo(InputFilename,CurrentEventName)
  
  antennamin=0
  antennamax= 160 # WE ARE GETTING THE RANDOM antennas!
  xpoints=CurrentAntennaInfo['X'].data[antennamin:antennamax]
  ypoints=CurrentAntennaInfo['Y'].data[antennamin:antennamax]
  zpoints=CurrentAntennaInfo['Z'].data[antennamin:antennamax]
  #print("positions", xpoints[0], ypoints[0], zpoints[0])
 

  PositionsPlane = np.column_stack((xpoints,ypoints,zpoints)) # [Number_of_antennas, 3] in meters
  #print(np.shape(PositionsPlane))
  

  antennamin=160
  antennamax=176 # WE ARE GETTING THE RANDOM antennas!
  AntID=CurrentAntennaInfo['ID'].data[antennamin:antennamax]
  xpoints=CurrentAntennaInfo['X'].data[antennamin:antennamax]
  print(np.shape(xpoints))
  ypoints=CurrentAntennaInfo['Y'].data[antennamin:antennamax]
  zpoints=CurrentAntennaInfo['Z'].data[antennamin:antennamax]
  t0points=CurrentAntennaInfo['T0'].data[antennamin:antennamax]
  t0points=CurrentAntennaInfo['T0'].data[0:160]


  #NewPos=np.stack((xpoints,ypoints,zpoints,t0points), axis=1)
  NewPos=np.stack((xpoints,ypoints,zpoints), axis =1) # interpolated positions in meters [Number_antennas,3]
  desiredtime = t0points # time for the interpolated antennas in ns 
  

  for i in range(160):
      Antennaid = "A" + "%d" %i
      Efield=hdf5io.GetAntennaEfield(InputFilename,CurrentEventName,Antennaid,OutputFormat="numpy")
      EfieldTraces.append(Efield)

  #Time = [0.5, -48.5, 2000] # TODO: check this part with Matias    
  
  
  CurrentEventInfo=hdf5io.GetEventInfo(InputFilename,CurrentEventName)
  GroundAltitude=hdf5io.GetGroundAltitude(CurrentEventInfo)
  xmaxposition=hdf5io.GetXmaxPosition(CurrentEventInfo)[0]
  print(xmaxposition)
  XmaxAltitude=hdf5io.GetXmaxAltitude(CurrentEventInfo)

  XmaxDistance=hdf5io.GetEventXmaxDistance(CurrentRunInfo,0)

  TargetShower = 0
  desired_trace = do_interpolation_hdf5(TargetShower, \
  VoltageTraces = None, FilteredVoltageTraces = None, antennamin=0, antennamax=159, DISPLAY=False, usetrace="efield")
  
  
except FileNotFoundError:
  logging.error("file not found or invalid:")

