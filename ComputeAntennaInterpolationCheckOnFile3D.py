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
from zhairesppath import GetZHSEffectiveactionIndex


import hdf5fileinout as hdf5io
from StarshapeInterpolation3D import do_interpolation_hdf5

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 14}

plt.rc('font', **font)

usetrace='efield'
DISPLAY=False


Energy = 0.681 # EeV
Zenith = 126.69  # GRANDconventions
Azimuth = 180.0 # GARND conventions
Inclination = 60.79 # degrees

Shower_parameters = np.array([Energy, Zenith, Azimuth, Inclination])
Xmax_distance = 10500.0 # meters
GroundAltitude = 1050.0 # meters
xXmax = 8420.0 # meters
yXmax = 0.0 # meters
zXmax = 7330.0 #meters
xmaxposition = np.array([xXmax, yXmax, zXmax])


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
  #.
  
  #one way of putting the antenna information as the numpy array this script was designed to use:
  
   # =============================================================================
  #   New implementation
  # =============================================================================

  antennamin=0
  antennamax= 160 # WE ARE GETTING THE RANDOM antennas!
  xpoints=CurrentAntennaInfo['X'].data[antennamin:antennamax]
  ypoints=CurrentAntennaInfo['Y'].data[antennamin:antennamax]
  zpoints=CurrentAntennaInfo['Z'].data[antennamin:antennamax]
  #print("positions", xpoints[0], ypoints[0], zpoints[0])
 
    
  nant = 0 # selected antenna
  nref = GetZHSEffectiveactionIndex(xmaxposition[0],xmaxposition[1],xmaxposition[2], \
  xant=xpoints[nant],yant=ypoints[nant],zant=zpoints[nant],ns=325,kr=-0.1218,stepsize = 20000) #averaged ref index
  dant = np.sqrt((xmaxposition[0]-xpoints[nant])**2+ (xmaxposition[1]-ypoints[nant])**2 +  \
                 (xmaxposition[2]-zpoints[nant])**2) # distance antenna-Xmax
  t0ref = (dant*nref/(3*1e8))*1e9 # spherical arrival time for the antenna "0"
  
  
  t0all=[] # diffrences between the arrival time for the antenna 0 and other antennas
  for i in range(160):
      nant = i
      nref = GetZHSEffectiveactionIndex(xmaxposition[0],xmaxposition[1],xmaxposition[2],\
      xant=xpoints[nant],yant=ypoints[nant],zant=zpoints[nant],ns=325,kr=-0.1218,stepsize = 20000)
      dant = np.sqrt((xmaxposition[0]-xpoints[nant])**2+ (xmaxposition[1]-ypoints[nant])**2 +\
                     (xmaxposition[2]-zpoints[nant])**2)
      t0 = (dant*nref/(3*1e8))*1e9 - t0ref
      t0all.append(t0)
  
    
  t0points=CurrentAntennaInfo['T0'].data[antennamin:antennamax]
  plt.plot(t0points-t0all)
  plt.xlabel("antenna id")
  plt.ylabel("t0_zhaires - t0_spherical [ns]")
  plt.tight_layout()
  plt.savefig("spherical_time.pdf")
  plt.show()
  
  
  print(t0)

  PositionsPlane = np.column_stack((xpoints,ypoints,zpoints)) # [Number_of_antennas, 3] in meters
  print(np.shape(PositionsPlane))
  

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
  
  CurrentRunInfo=hdf5io.GetRunInfo(InputFilename)
  CurrentEventName=hdf5io.GetEventName(CurrentRunInfo,0)
  correctedTime = []
  for i in range(160):
      Antennaid = "A" + "%d" %i
      Efield=hdf5io.GetAntennaEfield(InputFilename,CurrentEventName,Antennaid,OutputFormat="numpy")
      EfieldTraces.append(Efield)
      correctedTime.append(t0points[i] - Efield[:,0])
  
  CurrentEventNumber= 0
  CurrentRunInfo=hdf5io.GetRunInfo(InputFilename)
  CurrentEventName=hdf5io.GetEventName(CurrentRunInfo,CurrentEventNumber)
  #CurrentEventInfo=hdf5io.GetEventInfo(InputFilename,CurrentEventName)
  #CurrentShowerSimInfo=hdf5io.GetShowerSimInfo(InputFilename,CurrentEventName)
  CurrentSignalSimInfo=hdf5io.GetSignalSimInfo(InputFilename,CurrentEventName)

  Time = [0.5, -48.5, 2000] # TODO: check this part with Matias    
  #Efield_antennas = (4096, 4) Time Ex, Ey, Ew
  ## to define Traces (4096,4*number_antenna)
  # =============================================================================
  #     end
  # =============================================================================
  #print(np.shape(NewPos))
  ####################################################
  ############### Make a plot of the LDF for the paper, fast.
  ###############
  CurrentEventInfo=hdf5io.GetEventInfo(InputFilename,CurrentEventName)
  GroundAltitude=hdf5io.GetGroundAltitude(CurrentEventInfo)
  #xmaxposition=hdf5io.GetXmaxPosition(CurrentEventInfo)
  print(xmaxposition)
  XmaxAltitude=hdf5io.GetXmaxAltitude(CurrentEventInfo)
  #xXmax=xmaxposition[0][0]
  #yXmax=xmaxposition[0][1]
  #zXmax=xmaxposition[0][2]
  #print(xXmax, yXmax, zXmax)
  xpoints=CurrentAntennaInfo['X'].data[3:160:4]
  ypoints=CurrentAntennaInfo['Y'].data[3:160:4]
  zpoints=CurrentAntennaInfo['Z'].data[3:160:4]
  XmaxDistance=hdf5io.GetEventXmaxDistance(CurrentRunInfo,0)
  #print(XmaxDistance)
  p2p = hdf5io.get_p2p_hdf5(InputFilename,antennamax='All',antennamin=0,usetrace='efield')/75
  #print("p2p",p2p)
  p2px=p2p[0]
  p2px=p2px[3:160:4]
  p2py=p2p[3]
  p2py=p2py[3:160:4]
  p2pz=p2p[2]
  p2pz=p2pz[3:160:4]
  #print("xpoints",xpoints)
  #print("ypoints",ypoints)
  #print("zpoints",xpoints)
  #print("p2px",p2px)
  Zenith = hdf5io.GetEventZenith(CurrentRunInfo,0)
  #print("#Zenith --> ", Zenith)
  Azimuth = hdf5io.GetEventAzimuth(CurrentRunInfo,0)
  #print("#Azimuth --> ", Azimuth)
  Inclination = hdf5io.GetEventBFieldIncl(CurrentEventInfo)
  #print(Inclination)
  

  # =============================================================================
  #                              End
  # =============================================================================
    #
  desired_trace = do_interpolation_hdf5(Shower_parameters, Time, EfieldTraces, Xmax_distance, xmaxposition, GroundAltitude, PositionsPlane, NewPos, desiredtime, VoltageTraces = None, FilteredVoltageTraces = None, antennamin=0, antennamax=159, EventNumber=0, DISPLAY=DISPLAY, usetrace=usetrace)


except FileNotFoundError:
  logging.error("file not found or invalid:")

