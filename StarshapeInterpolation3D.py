'''Script to perform an interpolation between to electric field traces at a desired position
TODO: use magnetic field values and shower core from config-file
'''
import numpy as np
from scipy import signal
from scipy import spatial
import logging
import os
from os.path import split
import sys
from copy import deepcopy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from os.path import split, join, realpath
import astropy.units as u
#ZHAIRESPYTHON=os.environ["ZHAIRESPYTHON"]
#sys.path.append(ZHAIRESPYTHON)
import hdf5fileinout as hdf5io

def ComputeAntennaAlpha(XmaxPosition,AntennaX,AntennaY,AntennaZ, ShowerDirection, Cerenkov=1, Sign=False):
# WARNING: BE SURE THAT ANTENNA POSITIONS AND XMAX POSITIONS ARE ON THE SAME COORDINATE SYSTEM (I.E., THE ARE BOTH WRT SEA LEVEL, OR BOTH WRT GROUND LEVEL)
#this functions computes the angle (in degrees or Cerenkov Angles) between ShowerDirection (a versor pointing in the direction the shower moves)
#and the versor from XmaxPosition and the antenna at AntennaX, AntennaY, AntennaZ.
#The optional parameter Cerenkov is used for rescaling the resulting angle to units of the Cerenkov Angle (must be given in degrees))
#The optional parameter Sign asigns (arbitrarily) a sign to the angle depending if it is to the left or to the right (with respect to the x axis) (usefull for plotting
  v1=ShowerDirection
  v2=np.array([AntennaX-XmaxPosition[0],AntennaY-XmaxPosition[1],AntennaZ-XmaxPosition[2]])
  #Returns the angles in radians between vectors 'v1' and 'v2'
  v2=v2/np.linalg.norm(v2)
  cosang = np.dot(v1, v2)
  sinang = np.linalg.norm(np.cross(v1, v2))
  w = np.arctan2(sinang, cosang)
  Angle=(w*180/np.pi)/Cerenkov
  if(Sign):
    Angle=Angle*np.sign(v2[0]-v1[0])
  return Angle

def ComputeAntennaPhi(AntennaX,AntennaY, ReferenceDirection=np.array([1,0])):
# WARNING: BE SURE THAT ANTENNA POSITIONS AND ReferenceDirection are with respect to the core of the shower
#this functions computes the angle (in degrees) in the ground plane between the antena position and a reference direction (that could be another antena position
#The optional parameter Cerenkov is used for rescaling the resulting angle to units of the Cerenkov Angle (must be given in degrees))
#The optional parameter Sign asigns (arbitrarily) a sign to the angle depending if it is to the left or to the right (with respect to the x axis) (usefull for plotting
  v1=ReferenceDirection[0:2]
  v2=np.array([AntennaX,AntennaY])
  #Returns the angles in radians between vectors 'v1' and 'v2'
  #v2=v2/np.linalg.norm(v2)
  #print(v1,v2,AntennaX,AntennaY)
  w1 = np.arctan2(v1[1], v1[0])
  w2 = np.arctan2(v2[1], v2[0])
  Angle=((w2-w1)*180/np.pi)
  return Angle


#======================================
def unwrap(phi, ontrue=None):
    """Unwrap the phase so that the absolute difference
      between 2 consecutive phases remains below Pi

    Parameters:
    ----------
        phi: numpy array, float
            phase of the signal trace
        ontrue: str
            printing option, default=None

    Returns:
    ----------
        phi_unwrapped: numpy array, float
            unwarpped phase of the signal trace

    Adapted by E. Hivon (2020-02) from A. Zilles' unwrap
    """
    eps = np.finfo(np.pi).resolution
    thr = np.pi - eps
    pi2 = 2. * np.pi
    phi_unwrapped = np.zeros(phi.shape)
    p0  = phi_unwrapped[0] = phi[0]
    l   = 0
    for i0, p1 in enumerate(phi[1:]):
        i = i0 + 1
        dp = p1 - p0
        if (np.abs(dp) > thr):
            dl = np.floor_divide(abs(dp), pi2) + 1
            if (dp > 0):
                l -= dl
            else:
                l += dl
        phi_unwrapped[i] = p1 + l * pi2
        p0 = p1
        if ontrue is not None:
            print(i, phi[i],           phi[i-1],           abs(phi[i] - phi[i-1]),
                  l, phi_unwrapped[i], phi_unwrapped[i-1], abs(phi_unwrapped[i] - phi_unwrapped[i-1]))

    return phi_unwrapped
#======================================

#original by anne
def unwrap_anne(phi, ontrue=None):
    """Unwrap the phase to a strictly decreasing function.
    Parameters:
    ----------
        phi: numpy array, float
            phase of the signal trace
        ontrue: str
            printing option, default=None
    Returns:
    ----------
        phi_unwrapped: numpy array, float
            unwarpped phase of the signal trace
    """

    phi_unwrapped = np.zeros(phi.shape)
    p0 = phi_unwrapped[0] = phi[0]
    pi2 = 2. * np.pi
    l = 0
    for i0, p1 in enumerate(phi[1:]):
        i = i0 + 1
        if p1 >= p0:
            l += np.floor_divide(p1 - p0, pi2) + 1
        phi_unwrapped[i] = p1 - l * pi2
        p0 = p1
        if ontrue is not None:
            print(i, phi[i], phi[i-1], l, phi_unwrapped[i], abs(phi[i] - phi[i-1]),
                  abs(phi[i] - phi[i-1] + np.pi), abs(phi[i] - phi[i-1] - np.pi), l)
    return phi_unwrapped

def MatiasPhaseInterploation(phi1,w1,phi2,w2):
  phi=np.zeros(phi1.shape)
  for i, p1 in enumerate(phi1[1:]):
    if(np.abs(phi1[i]-phi2[i])>np.pi):
      if(phi1[i]>phi2[i]):
        #print("1 ")
        phi[i]=w1 * (phi1[i]-2*np.pi) + w2 * phi2[i] +2*np.pi
      else:
        #print("2")
        phi[i]=w1 *phi1[i] + w2 * (phi2[i]-2*np.pi) + 2*np.pi
    else:
      #print("3")
      phi[i]=w1*phi1[i] + w2*phi2[i]
    #print(i,phi[i],phi1[i],phi2[i],w1,w2)
  return phi

def LinePlaneCollision(planeNormal, planePoint, rayDirection, rayPoint, epsilon=1e-6):

	ndotu = planeNormal.dot(rayDirection)
	if abs(ndotu) < epsilon:
		raise RuntimeError("no intersection or line is within plane")

	w = rayPoint - planePoint
	si = -planeNormal.dot(w) / ndotu
	Psi = w + si * rayDirection + planePoint
	return Psi


    ##Define plane
	#planeNormal = np.array([0, 0, 1])
	#planePoint = np.array([0, 0, 5]) #Any point on the plane

	##Define ray
	#rayDirection = np.array([0, -1, -1])
	#rayPoint = np.array([0, 0, 10]) #Any point along the ray

	#Psi = LinePlaneCollision(planeNormal, planePoint, rayDirection, rayPoint)
	#print ("intersection at", Psi)


def interpolate_trace2(interpolate_mode,t1, trace1, x1, t2, trace2, x2, xdes, upsampling=None,  zeroadding=None):
    """Interpolation of signal traces at the specific position in the frequency domain

    The interpolation of traces needs as input antenna position 1 and 2, their traces (filtered or not)
    in one component, their time, and the desired antenna position and returns the trace ( in x,y,z coordinate system) and the time from the desired antenna position.
    Zeroadding and upsampling of the signal are optional functions.

    IMPORTANT NOTE:
    The interpolation of the phases includes the interpolation of the signal arrival time. A linear interpolation implies a plane radio
    emission wave front, which is a simplification as it is hyperbolic in shape. However, the wave front can be estimated as a plane between two simulated observer positions
    for a sufficiently dense grid of observers, as then parts of the wave front are linear on small scales.

    This script bases on the diploma thesis of Ewa Holt (KIT, 2013) in the context of AERA/AUGER. It is based on the interpolation of the amplitude and the pahse in the frequency domain.
    This can lead to misidentifiying of the correct phase. We are working on the interplementaion on a more robust interpolation of the signal time.
    Feel free to include it if you have some time to work on it. The script is completely modular so that single parts can be substitute easily.


    Parameters:
    ----------
            t1: numpy array, float
                time in ns of antenna 1
            trace1: numpy array, float
                single component of the electric field's amplitude of antenna 1
            x1: numpy array, float
                position of antenna 1 in angular coordinates
            t2: numpy array, float
                time in ns of antenna 2
            trace2: numpy array, float
                single component of the electric field's amplitude of antenna 2
            x2: numpy array, float
                position of antenna 2 in angular coordinates
            xdes: numpy arry, float
                antenna position for which trace is desired, in (phi,alpha) in degrees

    Returns:
    ----------
        xnew: numpy array, float
            rough estimation of the time for signal at desired antenna position in ns
        tracedes: numpy array, float
            interpolated electric field component at desired antenna position
    """
    DISPLAY = False
    ERIC=False      #computes the trace with the old unwrapping method

    # calculating weights:
    # if lines are at constant alpha, we weight on phi. If not, we weight on alpha
    #print("new criteria:",np.absolute(x1[2]-x2[2]),x1[2],x2[2])
    #if(np.absolute(x1[2]-x2[2])< 0.05) :
    #  tmp1 = np.absolute(x1[1]-xdes[1])
    #  if(tmp1>180):
    #   tmp1=360-tmp1
    #  tmp2 = np.absolute(x2[1]-xdes[1])
    #  if(tmp2>180):
    #   tmp2=360-tmp2
    #  #print("weight in phi",tmp1,tmp2,x1[1],x2[1],xdes[1])
    #else:
    #  tmp1 = np.absolute(x1[2]-xdes[2])
    #  tmp2 = np.absolute(x2[2]-xdes[2])
    #  #print("weight in alpha",tmp1,tmp2,x1[2])

    #weights, using the geometrical distance
    if(interpolate_mode=="phi"):
      #tmp1 = np.linalg.norm(x1 - xdes)
      #tmp2 = np.linalg.norm(x2 - xdes)
      tmp1 = np.absolute(x1[1]-xdes[1])
      tmp2 = np.absolute(x2[1]-xdes[1])
    elif(interpolate_mode=="alpha"): #using angular distance in alpha
      tmp1 = np.absolute(x1[2]-xdes[2])
      tmp2 = np.absolute(x2[2]-xdes[2])
    else:
       print("unknown interpolation mode")
       weight1=0.5
       weight2=0.5


    tmp = 1. / (tmp1 + tmp2)
    weight1 = tmp2 * tmp
    weight2 = tmp1 * tmp

    if np.isinf(weight1):
        print("weight = inf")
        print(x1, x2, xdes)
        weight1 = 1.
        weight2 = 0.
    if np.isnan(weight1):
        print('Attention: projected positions equivalent')
        weight1 = 1.
        weight2 = 0.

    epsilon = np.finfo(float).eps
    if (weight1 > 1. + epsilon) or (weight2 > 1 + epsilon):
        print("weight larger than 1: ", weight1, weight2, x1, x2, xdes, np.linalg.norm(
            x2-x1), np.linalg.norm(x2-xdes), np.linalg.norm(xdes-x1))
    if weight1 + weight2 > 1 + epsilon:
        print("PulseShape_Interpolation.py: order in simulated positions. Check whether ring or ray structure formed first")
        print(weight1, weight2, weight1 + weight2)


    #################################################################################
    # Fourier Transforms  # t in ns, Ex in muV/m, Ey, Ez
    #first antenna
    f = trace1
    xnew = t1

    fsample = 1./((xnew[1]-xnew[0]))  # GHz

    freq = np.fft.rfftfreq(len(xnew), 1./fsample)
    FFT_Ey = np.fft.rfft(f)

    Amp = np.abs(FFT_Ey)
    phi = np.angle(FFT_Ey)

    #############################

    # second antenna
    f2 = trace2
    xnew2 = t2

    fsample2 = 1./((xnew2[1]-xnew2[0]))  # GHz

    freq2 = np.fft.rfftfreq(len(xnew2), 1./fsample2)
    FFT_Ey = np.fft.rfft(f2)

    Amp2 = np.abs(FFT_Ey)
    phi2 = np.angle(FFT_Ey)

    ### Get the pulse sahpe at the desired antenna position

    # get the phase
    phides=MatiasPhaseInterploation(phi,weight1,phi2,weight2)

    if DISPLAY:
        phides2 = phides.copy()

    #check the first frequency.
    eps = np.finfo(float).resolution
    if(phides[0]>np.pi-eps):
      #print(str(eps)+"1 adjusting the first phase "+str(phides[0])+" " + str(phides[0]-np.pi))
      phides[0]=np.pi-eps

    if(phides[0]<-np.pi+eps):
      #print(str(eps)+"2 adjusting the first phase "+str(phides[0])+" " + str(phides[0]+np.pi))
      phides[0]=-np.pi+eps

    #Eric re-unwrap: get -pi to +pi range back and check whether phidesis in between (im not wraping any more, but if by some numerical reason phase is out of range, this fixes it)
    phides = np.mod(phides + np.pi, 2. * np.pi) - np.pi

    if(ERIC):
      phi_unwrapped_eric = unwrap(phi, ontrue)
      phi2_unwrapped_eric = unwrap(phi2, ontrue)
      phides_eric = weight1 * phi_unwrapped_eric + weight2 * phi2_unwrapped_eric
      phides_eric= np.mod(phides_eric + np.pi, 2. * np.pi) - np.pi


    #################################################################################
    ### linear interpolation of the amplitude
    #Amp, Amp2

    Ampdes = weight1 * Amp + weight2 * Amp2
    if(ERIC):
      Ampdes_eric=Ampdes.copy() #we make a copy becouse this is modified next

    if DISPLAY:
        Ampdes2 = Ampdes.copy()

    # inverse FFT for the signal at the desired position
    Ampdes = Ampdes.astype(np.complex64)
    phides = phides.astype(np.complex64)
    Ampdes *= np.exp(1j * phides)
    # trace
    tracedes = (np.fft.irfft(Ampdes))
    tracedes = tracedes.astype(float)

    if(ERIC):
      Ampdes_eric=Ampdes_eric.astype(np.complex64)
      phides_eric=phides_eric.astype(np.complex64)
      Ampdes_eric*= np.exp(1j * phides_eric)
      tracedes_eric=(np.fft.irfft(Ampdes_eric))
      tracedes_eric=tracedes_eric.astype(float)

    #this is a crude interpolation of the time, just to output something (nothing makes more sense at this point)
    tdes=(xnew*weight1+xnew2*weight2)

    if(len(tdes)>len(tracedes) and ERIC==False):
     #tdes=tdes[0:-1] #and this is required becouse the inverse fft returns one less time bin
     #print("interpolate_trace: lenghts are different",len(tdes),len(tracedes))
     tracedes.resize(len(tdes))
     #print("interpolate_trace: tried to fix it",len(tdes),len(tracedes))

    if(ERIC==True):
      if(len(tdes)>len(tracedes_eric)):
         #tdes=tdes[0:-1] #and this is required becouse the inverse fft returns one less time bin
         tracedes_eric.resize(len(tdes))

    # PLOTTING

    if (DISPLAY):
        import matplotlib.pyplot as plt
        plt.rc('font', family='serif', size=12)

        width=7.2
        height=2*width/1.618
        flow=21
        fhigh=121

        ########PHASES
        fig1 = plt.figure(1,figsize=(width,height), facecolor='w', edgecolor='k',dpi=120)
        ax1=fig1.add_subplot(412)
        tmp=ax1.plot(freq*1000, phi,linestyle='-', color='tab:green', label="a",linewidth=2)
        tmp=ax1.plot(freq2*1000, phi2,linestyle='-', color='tab:red', label="b",linewidth=2)
        if(ERIC):
          tmp=ax1.plot(freq2*1000, phides_eric, linestyle='--',color="tab:orange", label="a + b (unwraping)")
        tmp=ax1.plot(freq2*1000, phides, 'k:', label="a + b (this work)")
        #ax1.set_xlabel(r"Frequency (MHz)")
        tmp=ax1.set_ylabel(r"$\theta_{j}[rad]$")
        tmp=ax1.set_xlim(flow,fhigh)

        ######## AMPLITUDES#########################################################################
        ax1=fig1.add_subplot(411)
        tmp=ax1.plot(freq*1000, Amp/1000, linestyle='-', color='tab:green', label="a",linewidth=2)
        tmp=ax1.plot(freq2*1000, Amp2/1000, linestyle='-', color='tab:red', label="b",linewidth=2)
        tmp=ax1.plot(freq2*1000, Ampdes2/1000, linestyle='--',color="tab:orange", label="a + b (unwraping)")
        tmp=ax1.plot(freq2*1000, Ampdes2/1000, 'k:',label="a + b (this work)")
        #tmp=ax1.set_xlabel(r"Frequency (MHz)")
        tmp=ax1.set_ylabel(r"$|S_{j}(k)|[mV/GHz]$")
        tmp=ax1.legend()
        tmp=ax1.set_xlim(flow,fhigh)
        fig1.set_tight_layout(True)

        #UNWRAPPED PHASES #############################################################################
        phi2_unwrapped = unwrap(phi2)
        phi_unwrapped = unwrap(phi)
        phides2_unwrapped= unwrap(phides2)

        ax1 = fig1.add_subplot(413)
        if(ERIC):
          ax1.plot(freq*1000, phi_unwrapped_eric, color='tab:green', label="a",linewidth=2)
          ax1.plot(freq2*1000, phi2_unwrapped_eric, color='tab:red', label="b",linewidth=2)
          ax1.plot(freq2*1000,  weight1 * phi_unwrapped_eric + weight2 * phi2_unwrapped_eric,linestyle='--',color="tab:orange", label="a + b (unwraping)" )
        ax1.plot(freq2*1000,  phides2_unwrapped,'k:', label="a + b (this work)" )
        ax1.set_xlabel(r"Frequency (MHz)")#, fontsize=16)
        ax1.set_ylabel(r"$unwrap. \theta_{j}[rad]$")#, fontsize=16)
        tmp=ax1.set_xlim(flow,fhigh)
        tmp=ax1.set_ylim(-190,-30)

        #SIGNAL #############################################################################################################
        #fig2 = plt.figure(2, dpi=120, facecolor='w', edgecolor='k')
        ax1=fig1.add_subplot(414)
        #ax1.plot(np.real(t1), np.real(trace1), 'g:', label= "antenna 1")
        #ax1.plot(np.real(t2), np.real(trace2), 'b:', label= "antenna 2")
        #ax1.plot(np.real(tdes), np.real(tracedes), 'r-', label= "Synthetized")
        #ax1.plot(np.real(tdes), np.real(tracedes_eric), 'k-', label= "Synthetized with unwraping")

        ax1.plot(np.arange(0,trace1.size), np.real(trace1), linestyle='-', color='tab:green', label="a",linewidth=2)
        ax1.plot(np.arange(0,trace2.size), np.real(trace2), linestyle='-', color='tab:red', label="b",linewidth=2)
        ax1.plot(np.arange(0,tracedes.size), np.real(tracedes), 'k:', label="a + b (this work)")
        if(ERIC):
          ax1.plot(np.arange(0,tracedes_eric.size), np.real(tracedes_eric), linestyle='--',color="tab:orange", label="a + b (unwraping)")

        tmp=ax1.set_ylim(-151,149)
        tmp=ax1.set_xlim(280,1180)
        ax1.set_xlabel(r"Sample")
        ax1.set_ylabel(r"$S_y [\mu V]$")
        #ax1.legend(loc='best')

        plt.show()

    if(ERIC):
        return tdes,tracedes_eric

    else:
        return tdes, tracedes

####################################################################################################################
def GetAntennaAngles(Zenith,Azimuth,GroundAltitude,XmaxDistance,positions_sims,positions_des):

  #Input Azimuth and Zenith of the shower in ZHAireS conventions, in degrees!
  #Overdistance is a distance in meters behind xmax, where the original cone of the starshape was simulated. For historical reasons we used 3000 in the first library. In general, it will be 0. It will only afect slighly the topography correction
  #positions_sims and positions_des are expected to be above sea level, in AIRES coordinates

  #if the positions are not on the ground plane, where the starshape is, i have to compute the conical projection

  pos_des_plane= np.zeros([len(positions_des[:,1]),3]) #desired positions projected on the shower plane
  pos_des_ground= np.zeros([len(positions_des[:,1]),3]) #desited positions projected on the ground (starshape) plane

  #define shower direction vector
  az_rad=np.deg2rad(180.+Azimuth) #Note ZHAIRES units expected
  zen_rad=np.deg2rad(180.-Zenith)

  # shower vector  = direction of line for backprojection
  v = np.array([np.cos(az_rad)*np.sin(zen_rad),np.sin(az_rad)*np.sin(zen_rad),np.cos(zen_rad)])

  #in this part im doing verything taking 0,0, ground as the origin of coordinates, so im translating things to 0,0,ground.
  planeNormal=v
  GXmaxPosition=v*(XmaxDistance) #this sets origin on 0,0,ground

  #Now, select a Point in the intersectin plane (the ground) and a point in the ray going from xmax (so we pick xmax) to the antenna position, that will be propagated down to the plane.
  rayPoint=GXmaxPosition
  planePoint=np.array([0,0,0]) #the shower core is alwats on the ground and we will take a plane perpendicular to the shower at ground level

  #in the conical projection, if the antennas are not on the same plane of the ground, they will be anyway projected correctly to
  #the ground plane, becouse the algorithm searches for the colision point in the line from the antena to xmax.

  for i in np.arange(0,len(positions_des[:,1])):
     #direction from Xmax To the antenna
     rayDirection=(positions_des[i,:]-np.array([0,0,GroundAltitude]))-GXmaxPosition
     pos_des_plane[i,:]=LinePlaneCollision(planeNormal, planePoint, rayDirection, rayPoint, epsilon=1e-6)     #the projection to the perpendicular plane
     pos_des_ground[i,:]=LinePlaneCollision(np.array([0,0,1]), planePoint, rayDirection, rayPoint, epsilon=1e-6) #the projection to the ground plane

  #project simulation positions to put them on the plot
  pos_sims_plane= np.zeros([len(positions_sims[:,1]),3])
  pos_sims_ground= np.zeros([len(positions_sims[:,1]),3])
  for j in np.arange(0,len(positions_sims[:,1])):
    #direction from Xmax To the antenna
    rayDirection=positions_sims[j,:]-np.array([0,0,GroundAltitude])-GXmaxPosition
    pos_sims_plane[j,:]=LinePlaneCollision(planeNormal, planePoint, rayDirection, rayPoint, epsilon=1e-6)
    pos_sims_ground[j,:]=LinePlaneCollision(np.array([0,0,1]), planePoint, rayDirection, rayPoint, epsilon=1e-6) #the projection to the ground plane

  #Separate function?: calculate alpha and phi for simulated and desired positions (projected on the ground) and store in list
  pos_sims_angles=[]
  pos_des_angles=[]

  for i in np.arange(0,len(positions_sims[:,1])):

    phi=ComputeAntennaPhi(pos_sims_ground[i,0],pos_sims_ground[i,1])
    alpha=ComputeAntennaAlpha(GXmaxPosition,pos_sims_ground[i,0],pos_sims_ground[i,1],pos_sims_ground[i,2]-GroundAltitude, -planeNormal, Cerenkov=1, Sign=False)

    if round(phi,2) == -180.00:  #this is to put  2 in (-180,180]
        phi*=-1
    pos_sims_angles.append([i, phi, alpha])

  for i in np.arange(0,len(pos_des_ground[:,1])):

    phi=ComputeAntennaPhi(pos_des_ground[i,0],pos_des_ground[i,1])
    alpha=ComputeAntennaAlpha(GXmaxPosition,pos_des_ground[i,0],pos_des_ground[i,1],pos_des_ground[i,2], -planeNormal, Cerenkov=1, Sign=False)

    if round(phi,2) == -180.00:  #this is to put  2 in (-180,180]
        phi*=-1
    pos_des_angles.append([i, phi,alpha])

  #compute the distance ratio

  #get distances
  dist_pos_des_to_xmax=spatial.distance.cdist(GXmaxPosition.reshape(1,-1),(positions_des-np.array([0,0,GroundAltitude])),'euclidean')

  meanprojection=0

  for j in np.arange(0,len(positions_sims[:,1])):
    #direction from Xmax To the antenna (and also the vector from xmax to the position of the antena
    rayDirection=positions_sims[j,:]-np.array([0,0,GroundAltitude])-GXmaxPosition
    projection=np.dot(rayDirection,v)
    meanprojection=(meanprojection*j+projection)/(j+1)
    print(" projection",j,projection, meanprojection)

  dist_pos_des_on_plane_to_xmax=[]
  for j in np.arange(0,len(positions_des[:,1])):
    dist_pos_des_on_plane_to_xmax.append(projection/np.cos(pos_des_angles[j,2]))
    print("distance",dist_pos_des_on_plane_to_xmax[j],dist_pos_des_to_xmax[j])

  #the interpolation will be done on the ground plane, but is intended to be used for a point outside of the ground plane, and thus with a different distance to xmax
  #if it is closer, it will have a higher amplitude, if it is farterher away it will be smaller. The fluence scales with the square of the distance, so the electric field does it linear
  #i have to multiply the putput by the ratio
  distanceratio=dist_pos_des_on_plane_to_xmax/dist_pos_des_to_xmax


  return pos_sims_angles, pos_des_angles, distanceratio

def get_w(zenith, azimuth, x, y, z, x_Xmax, y_Xmax, z_Xmax):
 
    #function that returns "w" at each antenna, i.e. the angle between the direction that goes from Xmax to the core and the direction that goes from Xmax to a given antenna
    
    pi = np.pi
    zenith = zenith*pi/180.0
    azimuth = azimuth*pi/180.0
    
    x_antenna = x - x_Xmax # distance along the x-axis between the antennas postions and Xmax
    y_antenna = y - y_Xmax
    z_antenna = z - z_Xmax
    
    uv = np.array([np.sin(zenith)*np.cos(azimuth), np.sin(zenith)*np.sin(azimuth) , np.cos(zenith)]) # direction of the shower
    u_antenna = np.array([x_antenna, y_antenna, z_antenna]) # direction of the unit vectors that goes from Xmax to the position of the antennas
    u_antenna /= np.linalg.norm(u_antenna, axis =0)
    w = np.arccos(np.dot(np.transpose(u_antenna), uv))
    w = w*180.0/pi # we calculte w in degrees
    
    return w

#def get_distplane(x,y,z, x_Xmax, y_Xmax, z_Xmax, alpha):
    
#    distplane = np.zeros(len(x))
    
#    for i in range(len(x)):
#        distplane[i] = np.cos(alpha[i]*np.pi/180.0)*np.sqrt((x[i]-x_Xmax)**2 + (y[i]-y_Xmax)**2 + (z[i]-z_Xmax)**2)
    
#    return distplane

def get_distplane(zenith, azimuth, x, y, z, x_Xmax, y_Xmax, z_Xmax):
 
    #function that returns "w" at each antenna, i.e. the angle between the direction that goes from Xmax to the core and the direction that goes from Xmax to a given antenna
    
    pi = np.pi
    zenith = zenith*pi/180.0
    azimuth = azimuth*pi/180.0
    
    x_antenna = x - x_Xmax # distance along the x-axis between the antennas postions and Xmax
    y_antenna = y - y_Xmax
    z_antenna = z - z_Xmax
    
    uv = np.array([np.sin(zenith)*np.cos(azimuth), np.sin(zenith)*np.sin(azimuth) , np.cos(zenith)]) # direction of the shower
    u_antenna = np.array([x_antenna, y_antenna, z_antenna]) # direction of the unit vectors that goes from Xmax to the position of the antennas
    distplane = np.dot(np.transpose(u_antenna), uv)
    
    
    return distplane

def get_center(distplane, x_Xmax, y_Xmax, z_Xmax, GroundLevel, zenith, azimuth):

    pi = np.pi
    zenith = zenith*pi/180.0
    azimuth = azimuth*pi/180.0
    uv = np.array([np.sin(zenith)*np.cos(azimuth), np.sin(zenith)*np.sin(azimuth) , np.cos(zenith)]) # direction of the shower

    distground = np.sqrt(x_Xmax**2 + y_Xmax**2 + (z_Xmax-GroundLevel)**2)

    dist_plane_ground = distground - distplane
    core = uv*(dist_plane_ground)
    core[2] = core[2] - GroundLevel
    
    return core

def get_in_shower_plane(x,y,z, core, zenith, inclination,azimuth):
    
    # function that returns the trcaes in the shower plane (v, vxb, vxvxb) from the traces in the geographic plane (x, y, z)
    
    x = x + core[0]
    y = y + core[1] # We move the core position in (0,0,0) before changing the reference frame
    z = z + core[2]
    n = len(x) # number of antennas
    
    # antennas positions in the  shower reference frame (v, vxB, vxvxB)
    v = np.zeros(n)   
    vxb = np.zeros(n)
    vxvxb = np.zeros(n)
    
    # we reper the direction of the shower
    pi = np.pi
    zenith = zenith*pi/180.0 # elevation = zenith - 90°
    inclination = inclination*pi/180.0 # inclination of the magnetic field
    azimuth = azimuth*pi/180.0 # azimuth of the shower
    
    # unit vectors 
    uv = np.array([np.sin(zenith)*np.cos(azimuth), np.sin(zenith)*np.sin(azimuth) , np.cos(zenith)]) # direction of the shower
    uB = np.array([np.cos(inclination), 0, -np.sin(inclination)]) # direction of the magnetic field
    
    
    uv_x_uB = np.cross(uv, uB) # unit vector along the vxb direction
    uv_x_uB /= np.linalg.norm(uv_x_uB) # normalisation
    
    uv_x_uvxB  = np.cross(uv, uv_x_uB) # unit vector along the vxvxb direction
    uv_x_uvxB /= np.linalg.norm(uv_x_uB) # normalisation
    
    P = np.transpose(np.array([uv, uv_x_uB, uv_x_uvxB])) # matrix to go from the shower reference frame to the geographic reference frame
    
    P_inv = np.linalg.inv(P) # matrix to go from the geographic reference frame to the shower reference frame
    
    # We calculate the positions in the shower plane
    Position_geo = np.array([x,y,z]) # position in the geographic reference frame
    Position_shower = np.dot(P_inv, Position_geo) # position in the shower reference frame
    
    # We deduce the different components
    v = Position_shower[0, :] 
    vxb = Position_shower[1, :]
    vxvxb =  Position_shower[2, :]
    
    plt.scatter(x, y)
    plt.show()
    return (v,vxb, vxvxb,)

def get_phi(vxb, vxvxb):
    
    phi =np.arctan2(vxvxb,vxb)*180/np.pi
    
    phi_sims = abs(phi[0:160])
    phi_des = abs(phi[160:])
    
    phi_sims = phi_sims.astype(int) +1
    phi_des = phi_des.astype(int)

    return phi_sims, phi_des
###############################################################################

def GetAntennaAnglesSimon(Zenith,Azimuth, xmax_position,positions_sims,positions_des, GroundLevel, Inclination):

# =============================================================================
#                                 loading data
# =============================================================================
    
    x_sims, y_sims, z_sims = positions_sims[:,0], positions_sims[:,1], positions_sims[:,2]
    x_des, y_des, z_des = positions_des[:,0], positions_des[:,1], positions_des[:,2]
    xXmax, yXmax, zXmax = xmax_position[0], xmax_position[1], xmax_position[2]
    
# =============================================================================
#                                 alpha angle
# =============================================================================
   
    w_sims_angles = get_w(Zenith, Azimuth, x_sims, y_sims, z_sims, xXmax, yXmax, zXmax)
    w_des_angles = get_w(Zenith, Azimuth, x_des, y_des, z_des, xXmax, yXmax, zXmax)
    
# =============================================================================
#                               distance ratio
# =============================================================================
    
    distplane = np.mean(get_distplane(Zenith, Azimuth, x_sims, y_sims, z_sims, xXmax, yXmax, zXmax))
    dist_desired = get_distplane(Zenith, Azimuth, x_des, y_des, z_des, xXmax, yXmax, zXmax)
    distanceratio = [distplane/dist_desired]

# =============================================================================
#                                Phi angle
# =============================================================================
    
    center_plane = get_center(distplane, xXmax, yXmax, zXmax, GroundLevel, Zenith, Azimuth)
    x_all, y_all, z_all = np.concatenate((x_sims, x_des)), np.concatenate((y_sims, y_des)),\
    np.concatenate((z_sims, z_des))
    (v, vxb, vxvxb) = get_in_shower_plane(x_all, y_all, z_all, center_plane, Zenith, Inclination, Azimuth)
    #phi_sims, phi_des = get_phi(x_all +center_plane[0], y_all+ center_plane[1])
    phi_sims = np.zeros(160)
    phi_des = np.zeros(16)
    for i in np.arange(0,len(positions_sims[:,1])): phi_sims[i]=ComputeAntennaPhi(vxb[i],vxvxb[i])
    for i in np.arange(0,len(positions_des[:,1])): phi_des[i]=ComputeAntennaPhi(vxb[160+i],vxvxb[160+i])

    #print(phi_sims)
    pos_sims_angles = np.transpose([np.zeros(len(phi_sims)), phi_sims, w_sims_angles])
    pos_des_angles = np.transpose([np.zeros(len(phi_des)), phi_des, w_des_angles])

    
    return pos_sims_angles, pos_des_angles, distanceratio #phi_sims, phi_des 

def GetAntennaAnglesOnTheGround(Zenith,Azimuth,GroundAltitude,XmaxDistance,positions_sims,positions_des,overdistance=3000):
#Input Azimuth and Zenith of the shower in ZHAireS conventions, in degrees!
#Overdistance is a distance in meters behind xmax, where the original cone of the starshape was simulated. For historical reasons we used 3000 in the first library. In general, it will be 0. It will only afect slighly the topography correction
#positions_sims and positions_des are expected to be above sea level, in AIRES coordinates

    #if the positions are not on the ground plane, where the starshape is, i have to compute the conical projection

    pos_des_plane= np.zeros([len(positions_des[:,1]),3]) #desired positions projected on the shower plane
    pos_des_ground= np.zeros([len(positions_des[:,1]),3]) #desited positions projected on the ground (starshape) plane

    #define shower direction vector
    az_rad=np.deg2rad(180.+Azimuth) #Note ZHAIRES units expected
    zen_rad=np.deg2rad(180.-Zenith)

    # shower vector  = direction of line for backprojection
    v = np.array([np.cos(az_rad)*np.sin(zen_rad),np.sin(az_rad)*np.sin(zen_rad),np.cos(zen_rad)])

    #in this part im doing verything taking 0,0, ground as the origin of coordinates, so im translating things to 0,0,ground.
    planeNormal=v
    GXmaxPosition=v*(XmaxDistance+overdistance) #this sets origin on 0,0,ground

    #Now, select a Point in the intersectin plane (the ground) and a point in the ray going from xmax (so we pick xmax) to the antenna position, that will be propagated down to the plane.
    rayPoint=GXmaxPosition
    planePoint=np.array([0,0,0]) #the starshape is always on the ground when generated for ZHAireS

    #in the conical projection, if the antennas are not on the same plane of the ground, they will be anyway projected correctly to
    #the ground plane, becouse the algorithm searches for the colision point in the line from the antena to xmax.

    for i in np.arange(0,len(positions_des[:,1])):
         #direction from Xmax To the antenna
         rayDirection=(positions_des[i,:]-np.array([0,0,GroundAltitude]))-GXmaxPosition
         pos_des_plane[i,:]=LinePlaneCollision(planeNormal, planePoint, rayDirection, rayPoint, epsilon=1e-6)     #the projection to the perpendicular plane
         pos_des_ground[i,:]=LinePlaneCollision(np.array([0,0,1]), planePoint, rayDirection, rayPoint, epsilon=1e-6) #the projection to the ground plane

    #project simulation positions to put them on the plot
    pos_sims_plane= np.zeros([len(positions_sims[:,1]),3])
    pos_sims_ground= np.zeros([len(positions_sims[:,1]),3])
    for j in np.arange(0,len(positions_sims[:,1])):
        #direction from Xmax To the antenna
        rayDirection=positions_sims[j,:]-np.array([0,0,GroundAltitude])-GXmaxPosition
        pos_sims_plane[j,:]=LinePlaneCollision(planeNormal, planePoint, rayDirection, rayPoint, epsilon=1e-6)
        pos_sims_ground[j,:]=LinePlaneCollision(np.array([0,0,1]), planePoint, rayDirection, rayPoint, epsilon=1e-6) #the projection to the ground plane

    #get distances
    dist_pos_des_to_xmax=spatial.distance.cdist(GXmaxPosition.reshape(1,-1),(positions_des-np.array([0,0,GroundAltitude])),'euclidean')
    dist_pos_des_ground_to_xmax=spatial.distance.cdist(GXmaxPosition.reshape(1,-1),pos_des_ground)

    #the interpolation will be done on the ground plane, but is intended to be used for a point outside of the ground plane, and thus with a different distance to xmax
    #if it is closer, it will have a higher amplitude, if it is farterher away it will be smaller. The fluence scales with the square of the distance, so the electric field does it linear
    #i have to multiply the putput by the ratio
    distanceratio=dist_pos_des_ground_to_xmax/dist_pos_des_to_xmax


    #Separate function?: calculate alpha and phi for simulated and desired positions (projected on the ground) and store in list
    pos_sims_angles=[]
    pos_des_angles=[]

    for i in np.arange(0,len(positions_sims[:,1])):

        phi=ComputeAntennaPhi(positions_sims[i,0],positions_sims[i,1])
        alpha=ComputeAntennaAlpha(GXmaxPosition,positions_sims[i,0],positions_sims[i,1],positions_sims[i,2]-GroundAltitude, -planeNormal, Cerenkov=1, Sign=False)

        if round(phi,2) == -180.00:  #this is to put  2 in (-180,180]
            phi*=-1
        pos_sims_angles.append([i, phi, alpha])

    for i in np.arange(0,len(pos_des_ground[:,1])):

        phi=ComputeAntennaPhi(pos_des_ground[i,0],pos_des_ground[i,1])
        alpha=ComputeAntennaAlpha(GXmaxPosition,pos_des_ground[i,0],pos_des_ground[i,1],pos_des_ground[i,2], -planeNormal, Cerenkov=1, Sign=False)

        if round(phi,2) == -180.00:  #this is to put  2 in (-180,180]
            phi*=-1
        pos_des_angles.append([i, phi,alpha])


    return pos_sims_angles, pos_des_angles, distanceratio#, pos_sims_ground, pos_des_ground, pos_sims_plane, pos_des_plane


def PlotStarshape(Selected_I, Selected_II, Selected_III, Selected_IV, positions_sims,pos_sims_ground, pos_des_ground,positions_des,i,Energy,Zenith,p2p,desp2p,usetrace):

    #Plot
    sigmaE=15
    npuntos=50

    #using y component
    myp2p=p2p[1]
    mydesp2p=desp2p[1]

    fig71 = plt.figure(71, facecolor='w', edgecolor='k',figsize=(10,4.5))
    ax1=fig71.add_subplot(111)
    #TMP=ax1.scatter(pos_sims_ground[:,0],pos_sims_ground[:,1])
    #TMP=ax1.scatter(pos_des_ground[:,0],pos_des_ground[:,1],label="desired on ground")


    #Desired position is a black cross with a collored point on it
    tmp=ax1.scatter(positions_des[i,0],positions_des[i,1],marker="P",s=121,color="k")
    #Desired Position on the ground is a star
    TMP=ax1.scatter(pos_des_ground[i,0],pos_des_ground[i,1],marker="*",s=121,color="k")
    #Desired positions
    TMP=ax1.scatter(positions_des[:,0],positions_des[:,1],c=mydesp2p,cmap=plt.cm.jet,vmin=sigmaE*3,vmax=sigmaE*20,s=16)
    #Simulated positions on the starshape
    im=ax1.scatter(positions_sims[:,0],positions_sims[:,1],c=myp2p,cmap=plt.cm.jet,vmin=sigmaE*3,vmax=sigmaE*20,s=16)

    #Plot Selected Antennas with colored diamonds
    TMP=ax1.scatter(pos_sims_ground[Selected_I,0],pos_sims_ground[Selected_I,1],marker="D",s=100,color="tab:blue",label="I")
    TMP=ax1.annotate("I", xy=(positions_sims[Selected_I,0],positions_sims[Selected_I,1]),xycoords="data",xytext=(npuntos*(-0.25),npuntos*0.2),textcoords="offset points")
    TMP=ax1.scatter(pos_sims_ground[Selected_II,0],pos_sims_ground[Selected_II,1],marker="D",s=100,color="tab:orange",label="II")
    TMP=ax1.annotate("II", xy=(positions_sims[Selected_II,0],positions_sims[Selected_II,1]),xytext=(npuntos*0.15,npuntos*0.2),textcoords="offset points")
    TMP=ax1.scatter(pos_sims_ground[Selected_III,0],pos_sims_ground[Selected_III,1],marker="D",s=100,color="tab:purple",label="III")
    TMP=ax1.annotate("III", xy=(positions_sims[Selected_III,0],positions_sims[Selected_III,1]),xytext=(npuntos*(0.2),npuntos*(-0.3)),textcoords="offset points")
    TMP=ax1.scatter(pos_sims_ground[Selected_IV,0],pos_sims_ground[Selected_IV,1],marker="D",s=100,color="tab:olive",label="IV")
    TMP=ax1.annotate("IV", xy=(positions_sims[Selected_IV,0],positions_sims[Selected_IV,1]),xytext=(npuntos*(-0.3),npuntos*(-0.5)),textcoords="offset points")

    #TMP=ax1.legend(loc=2)

    #If you want to annotate the antenna positions
    #for j in range(0,160):
    #  ax1.annotate(str(j), (positions_sims[j,0],positions_sims[j,1]))

    #for j in range(0,16):
    #  ax1.annotate(" "+str(j), (positions_des[j,0],positions_des[j,1]))

    xlim=np.max(np.abs(positions_sims[:,0]))
    ylim=np.max(np.abs(positions_sims[:,1]))
    limit=max(xlim,ylim)*1.05
    # To limit the precision in the title
    myTitle="{0:1.2f} EeV {1:.2f} deg".format(np.power(10,Energy),180-Zenith)
    tmp=ax1.set(xlabel='Northing [m]',ylabel='Easting [m]', xlim=(-limit,limit*2/3), ylim=(-limit/2.51,limit/2.51))
    ax1.set_ylabel('Easting [m]',labelpad=-7)
    bar=fig71.colorbar(im,ax=ax1)
    if(usetrace=="efield"):
      bar.set_label('P2P Amplitude in Y [$\mu$V/m]', rotation=270, labelpad=17)
    if(usetrace=="voltage" or usetrace=="filteredvoltage"):
      bar.set_label('P2P Amplitude in Y [$\mu$V]', rotation=270, labelpad=17)

    from matplotlib import ticker
    tick_locator = ticker.MaxNLocator(nbins=5)
    bar.ax.yaxis.set_major_locator(ticker.AutoLocator())
    bar.locator = tick_locator
    bar.update_ticks()

    fig71.set_tight_layout(True)
    #plt.show()





#gets as input the positions of the simulated antennas in (antid, phi, alpha) and the position of the desired antena in phi,alpha)
#returns 4 list of antenna positions, separated in quadrants in (phi,alpha) arround the desired antenna.
def SelectAntennasForInterpolation(pos_sims_angles,pos_des_angle):

    index_I=[]
    index_II=[]
    index_III=[]
    index_IV=[]

    for m in np.arange(0,len(pos_sims_angles)): # desired points as reference
        delta_phi = pos_des_angle[1] - pos_sims_angles[m][1]
        if delta_phi > 180:
            delta_phi = delta_phi -360
        elif delta_phi < -180:
            delta_phi = delta_phi + 360

        delta_alpha = pos_sims_angles[m][2]-pos_des_angle[2]

        distance=np.sqrt(delta_alpha*delta_alpha+delta_phi*delta_phi) #distance in the Delta alpha, delthaphiplande. Since alpha in covered in much more granularity,

        if delta_phi >= 0. and  delta_alpha >= 0:
            index_I.append((m,delta_phi,delta_alpha,distance))
        if delta_phi >= 0. and  delta_alpha <= 0:
            index_II.append((m,delta_phi,delta_alpha,distance))
        if delta_phi <= 0. and  delta_alpha <= 0:
            index_III.append((m,delta_phi,delta_alpha,distance))
        if delta_phi <= 0. and  delta_alpha >= 0:
            index_IV.append((m,delta_phi,delta_alpha,distance))

    if(False):
        fig74 = plt.figure(4, facecolor='w', edgecolor='k')
        ax1=fig74.add_subplot(111)
        ax1.scatter([item[1] for item in index_I],[item[2] for item in index_I],label="I")
        ax1.scatter([item[1] for item in index_II],[item[2] for item in index_II],label="II")
        ax1.scatter([item[1] for item in index_III],[item[2] for item in index_III],label="III")
        ax1.scatter([item[1] for item in index_IV],[item[2] for item in index_IV],label="IV")
        ax1.legend(loc=2)

    bailoutI=bailoutII=bailoutIII=bailoutIV=0

    if not index_I:
        print("list - Quadrant 1 - empty")
        bailoutI=1
    if not index_II:
        print("list - Quadrant 2 - empty")
        bailoutII=1
        index_II=index_I
    if not index_III:
        print("list - Quadrant 3 - empty")
        bailoutIII=1
        index_III=index_IV
    if not index_IV:
        print("list - Quadrant 4 - empty")
        bailoutIV=1

    if(bailoutI==1 or bailoutIV==1 or (bailoutII==1 and bailoutIII==0) or (bailoutII==0 and bailoutIII==1)):
      print("Point is outside of the starshape, discarding", )
      return -1,-1,-1,-1

    if(bailoutII==1 and bailoutIII==1 and bailoutIV==0 and bailoutI==0):
        print(" I cannot find antennas with smaller alpha, lets try using only the inner antennas")
        index_II=index_I
        index_III=index_IV

    #now i convert this to numpy arrays to be able to sort them
    index_I=np.array(index_I, dtype = [('index', 'i4'), ('delta_phi', 'f4'), ('delta_alpha', 'f4'), ('distance', 'f4')])
    index_II=np.array(index_II, dtype = [('index', 'i4'), ('delta_phi', 'f4'), ('delta_alpha', 'f4'), ('distance', 'f4')])
    index_III=np.array(index_III, dtype = [('index', 'i4'), ('delta_phi', 'f4'), ('delta_alpha', 'f4'), ('distance', 'f4')])
    index_IV=np.array(index_IV, dtype = [('index', 'i4'), ('delta_phi', 'f4'), ('delta_alpha', 'f4'), ('distance', 'f4')])


    index_I = np.sort(index_I, order=['distance','delta_alpha', 'delta_phi'])
    index_II = np.sort(index_II, order=['distance','delta_alpha', 'delta_phi'])
    index_III = np.sort(index_III, order=['distance','delta_alpha','delta_phi'])
    index_IV = np.sort(index_IV, order=['distance','delta_alpha', 'delta_phi'])


    #now, index_I and index_IV are the ones with alpha bigger than thed desired. I want the closest in angle.
    #index_II and index_III are the ones with smaller alphas (so differences will be negative). I want the closest in angle

    print("Selected I ",index_I[0][0],"Selected II ", index_II[0][0],"Selected III ", index_III[0][0],"Selected IV ",index_IV[0][0])
    Selected_I=index_I[0][0]
    Selected_II=index_II[0][0]
    Selected_III=index_III[0][0]
    Selected_IV=index_IV[0][0]

    return Selected_I,Selected_II,Selected_III,Selected_IV


def PlotQuadrantSeparation(Selected_I, Selected_II, Selected_III, Selected_IV,pos_sims_angles,pos_des_angles,i):

    #Plot
    fig72 = plt.figure(72, facecolor='w', edgecolor='k')
    ax1=fig72.add_subplot(111)
    ax1.scatter([item[1] for item in pos_sims_angles],[item[2] for item in pos_sims_angles],label="sims")
    ax1.scatter([item[1] for item in pos_des_angles],[item[2] for item in pos_des_angles],label="desired")
    ax1.scatter(pos_des_angles[i][1],pos_des_angles[i][2],label="desired now")
    ax1.scatter(pos_sims_angles[Selected_I][1],pos_sims_angles[Selected_I][2],marker="D",s=100,color="tab:blue",label="I")
    ax1.scatter(pos_sims_angles[Selected_II][1],pos_sims_angles[Selected_II][2],marker="D",s=100,color="tab:orange",label="II")
    ax1.scatter(pos_sims_angles[Selected_III][1],pos_sims_angles[Selected_III][2],marker="D",s=100,color="tab:purple",label="III")
    ax1.scatter(pos_sims_angles[Selected_IV][1],pos_sims_angles[Selected_IV][2],marker="D",s=100,color="tab:olive",label="IV")
    ax1.legend(loc=2)


def PerformInterpolation(EfieldTraces, Time, Selected_I, Selected_II, Selected_III, Selected_IV, distanceratio, pos_sims_angles, pos_des_angle, DesiredT0, tracetype, VoltageTraces, FilteredVoltageTraces, DEVELOPMENT=False,DISPLAY=False, PLOTPAPER=False,i=0):
#InputFilename, where the starshape sim is
#CurrentSignalInfo  (so that i dont have to get it from the file each time the interpolation is done, as usually this will be used inside a loop over the desired antennas.
#CurrentAntennaInfo (so that i dont have to get it from the file each time the interpolation is done, as usually this will be used inside a loop over the desired antennas.
#CurrentEventName (so that i dont have to access it each time the interpolation is done, as usually this will be used inside a loop over the desired antennas.
#Selected_I to IV (the position in de antena index, for the antennas to use in the interpolation. Basically the output of SelectAntennasForInterpolation.
#distanceratio is the distance ratio for the topography correction (distance to xmax from the actual position used on the starshape after the projecton/distance to xmax for the desired antenna)
#pos_sims_angles holds the angular coordinates of each antenna in the format [IndexNr,alpha,phi] (in degrees!)
#pos_des_angle holds the angular coordinate of the desired point, in the format [IndexNr,alpha,phi] (in degrees!)
#DesiredT0 is the arrival time expected for the desired position, taken from the spherical shower front approximation. (the same used for the antennas in the starshape)


    #This was doing the interpolation in angular units for both angles, as it would logically be done. However, this happens to work worse at the cherenkov angle, for some reason
    # probably becouse in very inclined showers theta is very deformed, and and weighting in theta does not represent the actual distance to the antenna, that might be more important
    #(although, by doing it in distance you mess with the polarization)
    
    # TODO: check this part with Matias
    
    tmin = np.min(Time)
    tmax = np.max(Time)
    tbinsize = Time[0]
    
    tbinsize=hdf5io.GetTimeBinSize(CurrentSignalSimInfo)
    tmin=hdf5io.GetTimeWindowMin(CurrentSignalSimInfo)
    tmax=hdf5io.GetTimeWindowMax(CurrentSignalSimInfo)
   # print(tmin, tmin2, tmax, tmax2, tbinsize, tbinsize2)

    ## the interpolation of the pulse shape is performed, in x, y and z component
    
    #print(Selected_I,type(Selected_I), "!!!!!!!")
    
    #AntennaID=hdf5io.GetAntennaID(CurrentAntennaInfo,Selected_I)
    if(tracetype=='efield'):
      #txt0=hdf5io.GetAntennaEfield(InputFilename,CurrentEventName,AntennaID,OutputFormat="numpy")
      txt0 = EfieldTraces[Selected_I]
    elif(tracetype=='voltage'):
      #txt0=hdf5io.GetAntennaVoltage(InputFilename,CurrentEventName,AntennaID,OutputFormat="numpy")
      txt0 = VoltageTraces[Selected_I]
    elif(tracetype=='filteredvoltage'):
      #txt0=hdf5io.GetAntennaFilteredVoltage(InputFilename,CurrentEventName,AntennaID,OutputFormat="numpy")
      txt0 = FilteredVoltageTraces[Selected_I]
    else:
      print('PerformInterpolation:You must specify either efield, voltage or filteredvoltage, bailing out')
      return 0
  


    #AntennaID=hdf5io.GetAntennaID(CurrentAntennaInfo,Selected_IV)
    if(tracetype=='efield'):
      #txt1=hdf5io.GetAntennaEfield(InputFilename,CurrentEventName,AntennaID,OutputFormat="numpy")
      txt1 = EfieldTraces[Selected_IV]
    elif(tracetype=='voltage'):
      #txt1=hdf5io.GetAntennaVoltage(InputFilename,CurrentEventName,AntennaID,OutputFormat="numpy")
      txt1 = VoltageTraces[Selected_IV]
    elif(tracetype=='filteredvoltage'):
      #txt1=hdf5io.GetAntennaFilteredVoltage(InputFilename,CurrentEventName,AntennaID,OutputFormat="numpy")
      txt1 = FilteredVoltageTraces[Selected_I]
    else:
      print('PerformInterpolation:You must specify either efield, voltage or filteredvoltage, bailing out')
      return 0

    #AntennaID=hdf5io.GetAntennaID(CurrentAntennaInfo,Selected_II)
    if(tracetype=='efield'):
      #txt2=hdf5io.GetAntennaEfield(InputFilename,CurrentEventName,AntennaID,OutputFormat="numpy")
      txt2 = EfieldTraces[Selected_II]
    elif(tracetype=='voltage'):
     # txt2=hdf5io.GetAntennaVoltage(InputFilename,CurrentEventName,AntennaID,OutputFormat="numpy")
      txt2 = VoltageTraces[Selected_II]
    elif(tracetype=='filteredvoltage'):
      #txt2=hdf5io.GetAntennaFilteredVoltage(InputFilename,CurrentEventName,AntennaID,OutputFormat="numpy")
      txt2 = FilteredVoltageTraces[Selected_I]
    else:
      print('PerformInterpolation:You must specify either efield, voltage or filteredvoltage, bailing out')
      return 0


    #AntennaID=hdf5io.GetAntennaID(CurrentAntennaInfo,Selected_III)
    if(tracetype=='efield'):
      #txt3=hdf5io.GetAntennaEfield(InputFilename,CurrentEventName,AntennaID,OutputFormat="numpy")
      txt3 = EfieldTraces[Selected_III]
    elif(tracetype=='voltage'):
      #txt3=hdf5io.GetAntennaVoltage(InputFilename,CurrentEventName,AntennaID,OutputFormat="numpy")
      txt3 = VoltageTraces[Selected_III]
    elif(tracetype=='filteredvoltage'):
      #txt3=hdf5io.GetAntennaFilteredVoltage(InputFilename,CurrentEventName,AntennaID,OutputFormat="numpy")
      txt3 = FilteredVoltageTraces[Selected_III]
    else:
      print('PerformInterpolation:You must specify either efield, voltage or filteredvoltage, bailing out')
      return 0



    #points I and IV have a higher alpha. In case they are different, lets take the average and the same theta of the desired point as the pointa
    meanalpha=(pos_sims_angles[Selected_I][2]+pos_sims_angles[Selected_IV][2])/2.0
    #and theta is already available, is the theta of the desired position
    point_a=np.array([0,pos_des_angle[1],meanalpha])
    #points II and III have a lower alpha. In case they are different, lets take the average and the same theta of the desired point as the pointb
    meanalpha=(pos_sims_angles[Selected_II][2]+pos_sims_angles[Selected_III][2])/2.0
    #and theta is already available
    point_b=np.array([0,pos_des_angle[1],meanalpha])

    point_I=np.array([0,pos_sims_angles[Selected_I][1],pos_sims_angles[Selected_I][2]])
    point_II=np.array([0,pos_sims_angles[Selected_II][1],pos_sims_angles[Selected_II][2]])
    point_III=np.array([0,pos_sims_angles[Selected_III][1],pos_sims_angles[Selected_III][2]])
    point_IV=np.array([0,pos_sims_angles[Selected_IV][1],pos_sims_angles[Selected_IV][2]])


    #print("phi point I",point_I)
    #print("phi point a",point_a)
    #print("phi point IV",point_IV)

    #print("phi point II",point_II)
    #print("phi point b",point_b)
    #print("phi point III",point_III)

    #print("1st Interpol x")
    tnew1, tracedes1x = interpolate_trace2("phi",txt0.T[0], txt0.T[1], point_I , txt1.T[0], txt1.T[1], point_IV, point_a ,upsampling=None, zeroadding=None)
    #print("1st Interpol y")
    tnew1, tracedes1y = interpolate_trace2("phi",txt0.T[0], txt0.T[2], point_I , txt1.T[0], txt1.T[2], point_IV, point_a ,upsampling=None, zeroadding=None)
    #print("1st Interpol z")
    tnew1, tracedes1z = interpolate_trace2("phi",txt0.T[0], txt0.T[3], point_I , txt1.T[0], txt1.T[3], point_IV, point_a ,upsampling=None, zeroadding=None)

    #print("2nd Interpol x")
    tnew2, tracedes2x = interpolate_trace2("phi",txt2.T[0], txt2.T[1], point_II , txt3.T[0], txt3.T[1], point_III, point_b ,upsampling=None, zeroadding=None)
    #print("2nd Interpol y")
    tnew2, tracedes2y = interpolate_trace2("phi",txt2.T[0], txt2.T[2], point_II , txt3.T[0], txt3.T[2], point_III, point_b ,upsampling=None, zeroadding=None)
    #print("2nd Interpol z")
    tnew2, tracedes2z = interpolate_trace2("phi",txt2.T[0], txt2.T[3], point_II , txt3.T[0], txt3.T[3], point_III, point_b ,upsampling=None, zeroadding=None)


    #print("alpha point I",point_I)
    #print("alpha point a",point_a)
    #print("alpha point IV",point_IV)

    #print("alpha point II",point_II)
    #print("alpha point b",point_b)
    #print("alpha point III",point_III)

    ###### Get the pulse shape of the desired position from points on a and b
    #print("Interpol x")
    tnew_desiredx, tracedes_desiredx =interpolate_trace2("alpha",tnew1, tracedes1x, point_a, tnew2, tracedes2x, point_b, pos_des_angle, zeroadding=None)
    #print("Interpol y")
    tnew_desiredy, tracedes_desiredy =interpolate_trace2("alpha",tnew1, tracedes1y, point_a, tnew2, tracedes2y, point_b, pos_des_angle, zeroadding=None)
    #print("Interpol z")
    tnew_desiredz, tracedes_desiredz =interpolate_trace2("alpha",tnew1, tracedes1z, point_a, tnew2, tracedes2z, point_b, pos_des_angle, zeroadding=None)


    #aplyinnfg distance ratio
    #print("distanceratio",distanceratio)
    tracedes_desiredx=tracedes_desiredx*distanceratio
    tracedes_desiredy=tracedes_desiredy*distanceratio
    tracedes_desiredz=tracedes_desiredz*distanceratio

    #print("langth traces:", len(txt0.T[0]), len(txt0.T[1]),len(txt1.T[0]),len(txt1.T[1]),len(txt2.T[0]), len(txt2.T[1]),len(txt3.T[0]),len(txt3.T[1]))
    #print("langth interpolated:", len(xnew1), len(tracedes1x),len(xnew2),len(tracedes2x),len(xnew_desiredx), len(tracedes_desiredx))

    #now, lets use timing solution
    ntbins=np.shape(txt0)[0]
    tnew_desiredx=np.linspace(DesiredT0+tmin+tbinsize,DesiredT0+tmin+ntbins*tbinsize,ntbins)
    tnew_desiredy=tnew_desiredx
    tnew_desiredz=tnew_desiredx

    if(round((tnew_desiredx[2]-tnew_desiredx[1]),5)!=round(tbinsize,5)):
     print("warning! different tbin sizes",tbinsize,tnew_desiredx[2]-tnew_desiredx[1])
     print(tmin,tmax,tbinsize,ntbins)

    if (len(tracedes_desiredx)!=len(tracedes_desiredy)!=len(tracedes_desiredz)!=len(tnew_desiredx)!=ntbins) :
       print("WARNING: Traces are differnt lenght!",len(tracedes_desiredx),len(tracedes_desiredy),len(tracedes_desiredz),ntbins)


    #Given the desired and simulated traces, computes the diferences between the simulated and synthesiszed traces
    
# =============================================================================
#            Maybe useful for test
# =============================================================================
    #if DEVELOPMENT and DISPLAY>2:

       #AntennaID=hdf5io.GetAntennaID(CurrentAntennaInfo,160+i)
       #if(tracetype=='efield'):
         # txtdes=hdf5io.GetAntennaEfield(InputFilename,CurrentEventName,AntennaID,OutputFormat="numpy")
      # elif(tracetype=='voltage'):
          #txtdes=hdf5io.GetAntennaVoltage(InputFilename,CurrentEventName,AntennaID,OutputFormat="numpy")
       #elif(tracetype=='filteredvoltage'):
          #txtdes=hdf5io.GetAntennaFilteredVoltage(InputFilename,CurrentEventName,AntennaID,OutputFormat="numpy")
       #else:
          #print('PerformInterpolation:You must specify either efield, voltage or filteredvoltage, bailing out')
          #return 0

       #Plot_Interpolated_Comparison(txtdes,tnew_desiredx,tracedes_desiredx,tnew_desiredy,tracedes_desiredy,tnew_desiredz,tracedes_desiredz,tbinsize,PLOTPAPER)
       #Plot_Interpolation_Process(txtdes,txt0,txt1,txt2,txt3,tracedes1y,tracedes2y,tnew_desiredy,tracedes_desiredy,distanceratio, tracetype, PLOTPAPER)

       #plt.show()

    desired_trace=np.column_stack((tnew_desiredx,tracedes_desiredx,tracedes_desiredy,tracedes_desiredz))


    return desired_trace


def Plot_Interpolated_Comparison(txtdes,xnew_desiredx,tracedes_desiredx,xnew_desiredy,tracedes_desiredy,xnew_desiredz,tracedes_desiredz,tbinsize, PLOTPAPER=True):

   width=14
   height=width/1.618
   #fig4b= plt.figure(4, figsize=(width,height), facecolor='w', edgecolor='k')

   fig5, ((ax5, ax6, ax11), (ax7, ax8, ax12), (ax9, ax10, ax13))  = plt.subplots(3, 3, sharey='row', sharex='col')
   print("first time",xnew_desiredx[0],"true",txtdes.T[0][1])

   if(np.shape(tracedes_desiredx) != np.shape(txtdes.T[1])):
     simulatedx=txtdes.T[1,0:-1]
     time=txtdes.T[0,0:-1]/1000

   else:
     simulatedx=txtdes.T[1]
     time=txtdes.T[0]/1000

   #Xcomponent


   ccovx=np.correlate(tracedes_desiredx - tracedes_desiredx.mean(), simulatedx - simulatedx.mean(), mode='full')
   npts= len(tracedes_desiredx)
   lagsx = np.arange(-npts + 1, npts)
   ccorx = ccovx / (npts * tracedes_desiredx.std() * simulatedx.std())
   maxlag = lagsx[np.argmax(ccorx)]

   print("x max correlation %g occurres at lag %d" % (np.max(ccorx),maxlag))

   if(maxlag>=0):
     difference=[tracedes_desiredx[i] - simulatedx[i-maxlag] for i in np.arange(maxlag,len(simulatedx))]
     timex= [time[i] for i in np.arange(maxlag,len(simulatedx))]
   else:
     difference=[tracedes_desiredx[i] - simulatedx[i-maxlag] for i in np.arange(0,len(simulatedx)+maxlag)]
     timex=[time[i] for i in np.arange(0,len(simulatedx)+maxlag)]

   tmp=ax5.axhline(0, xmin=0, xmax=1, alpha=1, color="k",lw=1)
   tmp=ax5.plot(xnew_desiredx/1000, tracedes_desiredx, linestyle='--',color='g', label = "Synthetized")
   tmp=ax5.plot(time, simulatedx, label = "Simulation")

   tmp=ax6.axhline(0, xmin=0, xmax=1, alpha=1, color="k",lw=1)
   tmp=ax6.plot(timex,difference, linestyle='--',color='g', label = "Difference")
   tmp=ax6.set(ylabel='Difference')

   tmp=ax11.axhline(0, xmin=0, xmax=1, alpha=1, color="k",lw=1)
   tmp=ax11.axvline(0, ymin=0, ymax=1, alpha=0.5, color="k",linestyle="--")
   tmp=ax11.plot(lagsx*tbinsize, ccorx*np.max(simulatedx))
   tmp=ax11.set_ylabel('cross-correlation')

   #Ycomponent

   if(np.shape(tracedes_desiredy) != np.shape(txtdes.T[2])):
     simulatedy=txtdes.T[2,0:-1]
     time=txtdes.T[0,0:-1]/1000
   else:
     simulatedy=txtdes.T[2]
     time=txtdes.T[0]/1000


   ccovy=np.correlate(tracedes_desiredy - tracedes_desiredy.mean(), simulatedy - simulatedy.mean(), mode='full')
   npts= len(tracedes_desiredy)
   lagsy = np.arange(-npts + 1, npts)
   ccory = ccovy / (npts * tracedes_desiredy.std() * simulatedy.std())
   maxlag = lagsy[np.argmax(ccory)]
   print("y max correlation %g occurres at lag %d" % (np.max(ccory),maxlag))

   if(maxlag>=0):
     difference=[tracedes_desiredy[i] - simulatedy[i-maxlag] for i in np.arange(maxlag,len(simulatedy))]
     timey= [time[i] for i in np.arange(maxlag,len(simulatedy))]
   else:
     difference=[tracedes_desiredy[i] - simulatedy[i-maxlag] for i in np.arange(0,len(simulatedy)+maxlag)]
     timey=[time[i] for i in np.arange(0,len(simulatedy)+maxlag)]

   tmp=ax7.axhline(0, xmin=0, xmax=1, alpha=1, color="k",lw=1)
   tmp=ax7.plot(xnew_desiredy/1000, tracedes_desiredy, linestyle='--',color='g', label = "Synthetized")
   tmp=ax7.plot(time, simulatedy, label = "Simulation")
   tmp=ax7.legend(loc='lower right')

   tmp=ax8.axhline(0, xmin=0, xmax=1, alpha=1, color="k",lw=1)
   tmp=ax8.plot(timey, difference, linestyle='--',color='g', label = "Difference")
   tmp=ax8.set(ylabel='Difference')

   tmp=ax12.set_ylabel('cross-correlation')
   tmp=ax12.axhline(0, xmin=0, xmax=1, alpha=1, color="k",lw=1)
   tmp=ax12.axvline(0, ymin=0, ymax=1, alpha=0.5, color="k",linestyle="--")
   tmp=ax12.plot(lagsy*tbinsize, ccory*np.max(simulatedy))

   #zcomponent
   if(np.shape(tracedes_desiredz) != np.shape(txtdes.T[3])):
     simulatedz=txtdes.T[3,0:-1]
     time=txtdes.T[0,0:-1]/1000
   else:
     simulatedz=txtdes.T[3]
     time=txtdes.T[0]/1000

   ccovz=np.correlate(tracedes_desiredz - tracedes_desiredz.mean(), simulatedz - simulatedz.mean(), mode='full')
   npts= len(tracedes_desiredz)
   lagsz = np.arange(-npts + 1, npts)
   ccorz = ccovz / (npts * tracedes_desiredz.std() * simulatedz.std())
   maxlag = lagsz[np.argmax(ccorz)]
   #print("z max correlation %g occurres at lag %d" % (np.max(ccorz),maxlag))

   if(maxlag>=0):
     difference=[tracedes_desiredz[i] - simulatedz[i-maxlag] for i in np.arange(maxlag,len(simulatedz))]
     timez= [time[i] for i in np.arange(maxlag,len(simulatedz))]
   else:
     difference=[tracedes_desiredz[i] - simulatedz[i-maxlag] for i in np.arange(0,len(simulatedz)+maxlag)]
     timez=[time[i] for i in np.arange(0,len(simulatedz)+maxlag)]

   tmp=ax9.axhline(0, xmin=0, xmax=1, alpha=1, color="k",lw=1)
   tmp=ax9.plot(xnew_desiredz/1000, tracedes_desiredz, linestyle='--',color='g', label = "Synthetized")
   tmp=ax9.plot(time, simulatedz, label = "Simulation")

   #ax10 = fig5.add_subplot(3,2,6)
   tmp=ax10.axhline(0, xmin=0, xmax=1, alpha=1, color="k",lw=1)
   tmp=ax10.plot(timez, difference, linestyle='--',color='g', label = "Difference")
   tmp=ax10.set(xlabel='Time [$\mu$s]')
   tmp=ax10.set(ylabel='Difference')
   #ax10.legend(loc='lower right')

   tmp=ax13.set_xlim(-110, 110)
   tmp=ax13.axhline(0, xmin=0, xmax=1, alpha=1, color="k",lw=1)
   tmp=ax13.axvline(0, ymin=0, ymax=1, alpha=0.5, color="k",linestyle="--")
   tmp=ax13.plot(lagsz*tbinsize, ccorz*np.max(simulatedz))
   tmp=ax13.set_xlabel('lag [ns]')
   tmp=ax13.set_ylabel('cross-correlation')

   maxlag = lagsz[np.argmax(ccorz)]
   print("z max correlation is at lag %d" % maxlag)
   print("correlation x %g y %g z %g" % (np.max(ccorx), np.max(ccory), np.max(ccorz)))

   if(PLOTPAPER):
       tmp=ax5.set_xlim(0.25, 0.55)
       tmp=ax5.set(ylabel='$S_X\;[\mu V]$')
       tmp=ax9.set(ylabel='$S_Z\;[\mu V]$',xlabel='Time [$\mu$s]')
       tmp=ax9.set_xlim(0.25, 0.55)
       tmp=ax8.set_xlim(0.25, 0.55)
       tmp=ax7.set(ylabel='$S_Y\;[\mu V]$')
       tmp=ax7.set_xlim(0.25, 0.55)
       tmp=ax6.set_xlim(0.25, 0.55)
       tmp=ax10.set_xlim(0.25, 0.55)
       tmp=ax11.set_xlim(-110, 110)
       tmp=ax12.set_ylim(-110, 110)

       print("REMEMBER THAT YOU HARD CODED THE LIMITS OF THE PLOTS FOR THE PAPER!: THAT IS WHY YOU DONT SEE THE TRACES!")



def Plot_Interpolation_Process(txtdes,txt0,txt1,txt2,txt3,tracedes1y,tracedes2y,xnew_desiredy,tracedes_desiredy,ratio, tracetype, PLOTPAPER=False):
#this is ploting the Y component, but could be generalized and make a

   width=14
   height=width/1.618
   fig4b= plt.figure(44, figsize=(width,height), facecolor='w', edgecolor='k')

   yminlim=np.min(txtdes.T[2])
   ymaxlim=np.max(txtdes.T[2])

   ax4b = fig4b.add_subplot(2,2,1)
   tmp=ax4b.axhline(0, xmin=0, xmax=1, alpha=1, color="k",lw=1)
   ax4b.plot(np.arange(0,len(txt0.T[0])), signal.savgol_filter(txt0.T[2],11,3), label = "I",color='tab:blue',lw=2)
   ax4b.plot(np.arange(0,len(txt1.T[0])), signal.savgol_filter(txt1.T[2],11,3), label = "IV",color='tab:olive',lw=2)
   ax4b.plot(np.arange(0,len(tracedes1y)), signal.savgol_filter(tracedes1y,11,3), linestyle='--',color='tab:green',label = "I+IV = a",lw=2)
   if(tracetype=='efield'):
     tmp=ax4b.set(ylabel='$S_Y$ [$\mu$V/m]',xlabel='Sample')
   else:
     tmp=ax4b.set(ylabel='$S_Y$ [$\mu$V]',xlabel='Sample')
   ax4b.legend(loc='best')
   tmp=ax4b.set_ylim(yminlim*1.4,ymaxlim*1.4)
   if(PLOTPAPER):
     tmp=ax4b.set_xlim(280,720)

   ax4b = fig4b.add_subplot(2,2,2)
   tmp=ax4b.axhline(0, xmin=0, xmax=1, alpha=1, color="k",lw=1)
   ax4b.plot(np.arange(0,len(txt2.T[0])), signal.savgol_filter(txt2.T[2],11,3), label = "II",color='tab:orange',lw=2)
   ax4b.plot(np.arange(0,len(txt3.T[0])), signal.savgol_filter(txt3.T[2],11,3), label = "III",color='tab:purple',lw=2)
   ax4b.plot(np.arange(0,len(tracedes2y)), signal.savgol_filter(tracedes2y,11,3), linestyle='--', color='tab:red',label = "II+III = b",lw=2)
   tmp=ax4b.set(xlabel='Sample')
   ax4b.legend(loc='best')
   tmp=ax4b.set_ylim(yminlim*1.4,ymaxlim*1.4)
   if(PLOTPAPER):
     tmp=ax4b.set_xlim(280,720)

   ax4b = fig4b.add_subplot(2,2,3)
   tmp=ax4b.axhline(0, xmin=0, xmax=1, alpha=1, color="k",lw=1)
   ax4b.plot(np.arange(0,len(tracedes1y)), signal.savgol_filter(tracedes1y,11,3), linestyle='--',color='tab:green',label = "a",lw=2)
   ax4b.plot(np.arange(0,len(tracedes2y)), signal.savgol_filter(tracedes2y,11,3), linestyle='--', color='tab:red',label = "b",lw=2)
   ax4b.plot(np.arange(0,len(tracedes_desiredy)), signal.savgol_filter(tracedes_desiredy/ratio,11,3), linestyle=':',color='k', label = "a + b",lw=2)
   if(tracetype=='efield'):
     tmp=ax4b.set(ylabel='$S_Y$ [$\mu$V/m]',xlabel='Sample')
   else:
     tmp=ax4b.set(ylabel='$S_Y$ [$\mu$V]',xlabel='Sample')
   ax4b.legend(loc='best')
   tmp=ax4b.set_ylim(yminlim*1.4,ymaxlim*1.4)
   if(PLOTPAPER):
     tmp=ax4b.set_xlim(280,720)

   ax4b = fig4b.add_subplot(2,2,4)
   tmp=ax4b.axhline(0, xmin=0, xmax=1, alpha=1, color="k",lw=1)
   ax4b.plot(xnew_desiredy/1000, signal.savgol_filter(tracedes_desiredy/ratio,11,3), linestyle=':',color='k',lw=2, label = "a + b")
   ax4b.plot(xnew_desiredy/1000, signal.savgol_filter(tracedes_desiredy,11,3),color='k', label = "Synthesized",lw=2)
   ax4b.plot(txtdes.T[0]/1000, signal.savgol_filter(txtdes.T[2],11,3), label = "Simulation",color='g',lw=2)
   tmp=ax4b.set(xlabel='Time [$\mu$s]')
   ax4b.legend(loc='best')
   tmp=ax4b.set_ylim(yminlim*1.4,ymaxlim*1.4)
   if(PLOTPAPER):
     tmp=ax4b.set_xlim(0.25,0.46)





def do_interpolation_hdf5(Shower_parameters, Time, EfieldTraces, XmaxDistance, xmaxposition, GroundAltitude, PositionsPlane, desired, desiredtime, VoltageTraces, FilteredVoltageTraces, antennamin=0, antennamax=159, EventNumber=0, DISPLAY=False, usetrace='efield',overdistance=3000,FillOutliersWithZeros=True):
    '''
    Reads in arrays, looks for neighbours, calls the interpolation and saves the traces

    Parameters:
    ----------
    desired: str
        numpy array of desired antenna positions (x,y,z,t0 info) (in grand coordinates, so , above sea level)
    InputFilename: str
        path to HDF5 simulation file
        The script accepts starshape as well as grid arrays
    antennamin,antennamax:int
        the program is designed to run on the first 160 antennas. If your simulation has more, you can specify a range to be used...but it has to be tested
    EventNumber: int
        number of event in the file to use. you can process only one at a time
    DISPLAY: True/False
        enables printouts and plots
    usetrace: str (note that for now you can only do one at a time, and on different output files)
        efield
        voltage
        filteredvoltage
    overdistance: For compatibilitiy and historical reasons, you can add an additional distance behind xmax for the start of the cone.
    FillOutliersWithZeros: If an antenna is outside of the starshape, you set it to 0. If false, the antenna is skipped (and there is no output to the file)

    Returns:
    ----------
        --
    Saves traces via index infomation in same folder as desired antenna positions


    NOTE: The selection of the neigbours is sufficiently stable, but does not always pick the "best" neigbour, still looking for an idea
    '''
    #print(shower_core)
    DEVELOPMENT=True #only for developing, use when working on the starshape patern and trying to interpolate the random check antenas on the starshape (or it will crash)
                     #it disables removing antennas outside the pattern
    #DEVELOPMENT=False

    DISPLAY=0
    #0 dont plot
    #3 plot plot the interpolated traces and the errors in the interpolation (if in DEVELOPMENT)
    #4 plot starshape  (if in DEVELOPMENT)
    #5 plot quadrant selection in alpha,phi


    PLOTPAPER=False #shows only antenna 4 and fixes the axis to accomodate for the plot on the JINST paper

    if(usetrace=="all"):
      print("usetrace is all, looping over all trace types")
      usetracelist=["efield","voltage","filteredvoltage"]
    else:
      usetracelist=[str(usetrace)]

    #Getting Required Information from the InputEvent

    #CurrentEventNumber=EventNumber
    #CurrentRunInfo=hdf5io.GetRunInfo(InputFilename)
   # CurrentEventName=hdf5io.GetEventName(CurrentRunInfo,CurrentEventNumber)
    #CurrentEventInfo=hdf5io.GetEventInfo(InputFilename,CurrentEventName)
    #CurrentShowerSimInfo=hdf5io.GetShowerSimInfo(InputFilename,CurrentEventName)
    #CurrentSignalSimInfo=hdf5io.GetSignalSimInfo(InputFilename,CurrentEventName)

    #Zenith=hdf5io.GetEventZenith(CurrentRunInfo,CurrentEventNumber)
    Zenith = Shower_parameters[1]
    #Azimuth=hdf5io.GetEventAzimuth(CurrentRunInfo,CurrentEventNumber)
    Azimuth = Shower_parameters[2]
    #Energy=np.log10(hdf5io.GetEventEnergy(CurrentRunInfo,CurrentEventNumber)) #The energy is just for the plots
    Energy=np.log10(Shower_parameters[0])
    #XmaxDistance=hdf5io.GetEventXmaxDistance(CurrentRunInfo,CurrentEventNumber)
    #GroundAltitude=hdf5io.GetGroundAltitude(CurrentEventInfo)
    #Inclination = hdf5io.GetEventBFieldIncl(CurrentEventInfo)
    Inclination = Shower_parameters[3]
    #xmaxposition=hdf5io.GetXmaxPosition(CurrentEventInfo)
    #xXmax=xmaxposition[0]
    #yXmax=xmaxposition[1]
    #zXmax=xmaxposition[2]
    #xmaxposition = np.array([xXmax, yXmax, zXmax])
    # SIMULATION
    # Read in simulated position list
    #CurrentAntennaInfo=hdf5io.GetAntennaInfo(InputFilename,CurrentEventName)
    antennamax=antennamax+1
    #one way of putting the antenna information as the numpy array this script was designed to use:
    #xpoints=CurrentAntennaInfo['X'].data[antennamin:antennamax]
    #ypoints=CurrentAntennaInfo['Y'].data[antennamin:antennamax]
    #zpoints=CurrentAntennaInfo['Z'].data[antennamin:antennamax]
    #t0s=CurrentAntennaInfo['T0'].data[antennamin:antennamax]
    #positions_sims=np.column_stack((xpoints,ypoints,zpoints))
    positions_sims = PositionsPlane

    # Hand over a list file including the antenna positions you would like to have. This could be improved by including an ID.
    positions_des = desired
    #DesiredAntennaInfoMeta=hdf5io.CreatAntennaInfoMeta(split(InputFilename)[1],CurrentEventName,AntennaModel="Interpolated") #TODO: discuss that!  
    DesiredIds=np.arange(0, len(positions_des)) #this could be taken from the input file of desired antennas
    #print(DesiredIds)
    DesiredAntx=deepcopy(positions_des.T[0])
    DesiredAnty=deepcopy(positions_des.T[1])
    DesiredAntz=deepcopy(positions_des.T[2]) #this deepcopy bullshit is becouse position_des is later modified by the rotation, and transposition apparently creates a shallow copy (a reference)
    DesiredSlopeA=np.zeros(len(positions_des))
    DesiredSlopeB=np.zeros(len(positions_des))
    #DesiredT0=deepcopy(positions_des.T[3])
    DesiredT0=deepcopy(desiredtime.T)

    #print(DesiredT0)
    # Here the magic starts
    #now i come back to having only x,y,z on positions_des
    ##positions_des=desired[:,0:3]
    positions_des=desired
    #Compute the antenna positions in alpha, phi
    #, pos_sims_ground, pos_des_ground, pos_sims_plane, pos_des_plane 
    pos_sims_angles, pos_des_angles, distanceratio = GetAntennaAnglesSimon(Zenith,Azimuth,xmaxposition,positions_sims,positions_des, GroundAltitude, Inclination)
    pos_sims_angles2, pos_des_angles2, distanceratio2 = GetAntennaAnglesOnTheGround(Zenith,Azimuth,GroundAltitude,XmaxDistance,positions_sims,positions_des,overdistance=0)
    #print(distanceratio)
    #print(np.shape(pos_sims_angles), np.shape(pos_des_angles), np.shape(distanceratio))

    #for k in range(len(pos_sims_angles[:,1])):
        #print(pos_sims_angles[k][1], pos_sims_angles2[k][1])
        #pos_sims_angles[k][1] = pos_sims_angles2[k][1]
    #for k in range(len(pos_des_angles[:,1])):
        #pos_des_angles[k][1] = pos_des_angles2[k][1]
    #diff = np.zeros(160)

    #for i in range(len(pos_sims_angles)):
        #diff[i] = pos_sims_angles[i][2] - pos_sims_angles2[i][2]
        #print(pos_sims_angles[i][1], pos_sims_angles2[i])
        #print(distanceratio, distanceratio2)
    #plt.scatter(np.arange(0,160,1), diff)
    #plt.xlabel("antenna ID")
    #plt.ylabel("$\\alpha_{matias} - \\alpha_{simon}$ [Deg.]")
    #plt.tight_layout()
    #plt.savefig("alpha_difference_interpolation.pdf")
    #plt.show()
    
    #print(pos_sims_angles2, pos_des_angles2)
    #print(distanceratio, distanceratio2)

    #print(pos_sims_angles, np.shape(pos_sims_angles))
    #pos_sims_angles, pos_des_angles, distanceratio = GetAntennaAngles(Zenith,Azimuth,GroundAltitude,XmaxDistance,positions_sims,positions_des)

    #aca empiezo a loopear en las antennas
    remove_antenna=[]
    desired_traceAll = []
    for i in np.arange(0,len(pos_des_angles)):
        if(PLOTPAPER and DISPLAY>0 and i!=4):
          print("skipping "+str(i))
          continue #(this is here for the plot of the interpolation paper)
        if(DISPLAY>0):
          print("starting with antenna "+str(i))

        #select the four antennas for the inerpolation for this desired antenna
        Selected_I,Selected_II,Selected_III,Selected_IV = SelectAntennasForInterpolation(pos_sims_angles,pos_des_angles[i])

        if(DISPLAY>4):
          PlotQuadrantSeparation(Selected_I, Selected_II, Selected_III, Selected_IV, pos_sims_angles, pos_des_angles, i)
        #Selected_I = -1
        Skip=False
        for tracetype in usetracelist:
            print("computing for "+tracetype+" on desired antenna "+str(i))
            #If there was a problem selecting the antennas, dont interpolate
            Skip=False
            if(Selected_I==-1 or Selected_II==-1 or Selected_III == -1 or Selected_IV==-1):
                if(FillOutliersWithZeros==True):
                    print("antenna outside of the starshape, interpolation not performed, filled with 0",i)
                    #tbinsize=hdf5io.GetTimeBinSize(CurrentSignalSimInfo)
                    #tmin=hdf5io.GetTimeWindowMin(CurrentSignalSimInfo)
                    #tmax=hdf5io.GetTimeWindowMax(CurrentSignalSimInfo)
                    #ntbins=(tmax-tmin)/tbinsize
                    #ntbins=int(ntbins)
                    ntbins = len(EfieldTraces[0][:,0])
                    print(ntbins)
                    desired_trace=np.zeros((ntbins,4))
                else:
                    print("antenna outside of the starshape, interpolation not performed, antenna removed",i)
                    if(Skip==False): #to do it only once
                       Skip=True
                       remove_antenna.append(i)
            else:
                #Do the interpolation
                desired_trace=PerformInterpolation(EfieldTraces, Time, Selected_I, Selected_II, Selected_III, Selected_IV, distanceratio[0][i], pos_sims_angles, pos_des_angles[i],DesiredT0[i], tracetype,VoltageTraces, FilteredVoltageTraces, DEVELOPMENT=DEVELOPMENT,DISPLAY=DISPLAY, PLOTPAPER=PLOTPAPER, i=i)
                desired_traceAll.append(desired_trace)
                #desired_trace=PerformInterpolation(InputFilename, CurrentEventName, CurrentAntennaInfo, CurrentSignalSimInfo, Selected_I, Selected_II, Selected_III, Selected_IV, distanceratio[0][i],pos_sims_angles, pos_des_angles[i],pos_sims_ground, pos_des_ground[i],DesiredT0[i], tracetype,DEVELOPMENT=DEVELOPMENT,DISPLAY=DISPLAY, PLOTPAPER=PLOTPAPER, i=i)
            #if we are not skipping, put it on the fie
            #if(Skip==False):
                #Put it on the file
                #if(tracetype=='efield'):
                    #EfieldTable=hdf5io.CreateEfieldTable(desired_trace, CurrentEventName, CurrentEventNumber , DesiredIds[i], i, "Interpolated", info={})
                    #hdf5io.SaveEfieldTable(OutputFilename,CurrentEventName,str(DesiredIds[i]),EfieldTable)
                #elif(tracetype=='voltage'):
                    #VoltageTable=hdf5io.CreateVoltageTable(desired_trace, CurrentEventName, CurrentEventNumber , DesiredIds[i], i, "Interpolated", info={})
                    #hdf5io.SaveVoltageTable(OutputFilename,CurrentEventName,str(DesiredIds[i]),VoltageTable)
                #elif(tracetype=='filteredvoltage'):
                    #VoltageTable=hdf5io.CreateVoltageTable(desired_trace, CurrentEventName, CurrentEventNumber , DesiredIds[i], i, "Interpolated", info={})
                    #hdf5io.SaveFilteredVoltageTable(OutputFilename,CurrentEventName,str(DesiredIds[i]),VoltageTable)

                #if(DEVELOPMENT and DISPLAY>3):
                    #This is just to plot the starshape with colors
                   #p2p,peak,fluence= hdf5io.get_time_amplitudes_fluence_hdf5(InputFilename, antennamax=159 ,antennamin=0, usetrace=tracetype,  windowsize=200, DISPLAY=False)
                   #desp2p,despeak,desfluence= hdf5io.get_time_amplitudes_fluence_hdf5(InputFilename, antennamax=175 ,antennamin=160, usetrace=tracetype,  windowsize=200, DISPLAY=False)
                   #PlotStarshape(Selected_I, Selected_II, Selected_III, Selected_IV, positions_sims,pos_sims_ground, pos_des_ground,positions_des,i,Energy,Zenith,p2p,desp2p,tracetype)

            #plt.show()
            #End loop on trace types
        #delete after iterating over all the tracerypes
        del Selected_I, Selected_II, Selected_III, Selected_IV
        #end of the loop on the desired positions

    #now, lets remove the skiped antennas from the index
    DesiredIds=np.delete(DesiredIds,remove_antenna)
    DesiredAntx=np.delete(DesiredAntx,remove_antenna)
    DesiredAnty=np.delete(DesiredAnty,remove_antenna)
    DesiredAntz=np.delete(DesiredAntz,remove_antenna)
    DesiredSlopeA=np.delete(DesiredSlopeA,remove_antenna)
    DesiredSlopeB=np.delete(DesiredSlopeB,remove_antenna)
    DesiredT0=np.delete(DesiredT0,remove_antenna)
   
    
    #CreateAntennaInfo(IDs, antx, anty, antz, antt, slopeA, slopeB, AntennaInfoMeta, P2Pefield=None,P2Pvoltage=None,P2Pfiltered=None,HilbertPeak=None,HilbertPeakTime=None):
    #3DesiredAntennaInfo=hdf5io.CreateAntennaInfo(DesiredIds, DesiredAntx, DesiredAnty, DesiredAntz, DesiredT0, DesiredSlopeA, DesiredSlopeB, DesiredAntennaInfoMeta) # we save the traces in the new hdf5
    ##hdf5io.SaveAntennaInfo(OutputFilename,DesiredAntennaInfo,CurrentEventName)
    #not using them, but i put SignalSim And ShowerSim Info
    #For now, i save a copy. I could modify some fields to show this is an interpolation
    ##hdf5io.SaveRunInfo(OutputFilename,CurrentRunInfo)
    ##hdf5io.SaveEventInfo(OutputFilename,CurrentEventInfo,CurrentEventName)
    ##hdf5io.SaveShowerSimInfo(OutputFilename,CurrentShowerSimInfo,CurrentEventName)
    ##hdf5io.SaveSignalSimInfo(OutputFilename,CurrentSignalSimInfo,CurrentEventName)

    #now we are at the point where all the antennas where interpolated and saved to file. I will calulate the peak to peak and hilbert envelope peak and time
    #this is done after everything was computed.
    
   # =============================================================================
   #        optionnal part 9to rewrite without dependency to the hdf5 file)
   # =============================================================================
    
    #if(usetrace=="all"):
      #print("Computing P2P for "+str(OutputFilename))

      #OutAntennaInfo=hdf5io.GetAntennaInfo(OutputFilename,CurrentEventName)
      #OutIDs=hdf5io.GetAntIDFromAntennaInfo(OutAntennaInfo)

      #p2pE=hdf5io.get_p2p_hdf5(OutputFilename,usetrace='efield')
      #p2pV=hdf5io.get_p2p_hdf5(OutputFilename,usetrace='voltage')
      #p2pFV=hdf5io.get_p2p_hdf5(OutputFilename,usetrace='filteredvoltage')

      #peaktimeE, peakE=hdf5io.get_peak_time_hilbert_hdf5(OutputFilename,usetrace='efield')
      #peaktimeV, peakV=hdf5io.get_peak_time_hilbert_hdf5(OutputFilename,usetrace='voltage')
      #peaktimeFV, peakFV=hdf5io.get_peak_time_hilbert_hdf5(OutputFilename,usetrace='filteredvoltage')

      #AntennaP2PInfo=hdf5io.CreateAntennaP2PInfo(OutIDs, DesiredAntennaInfoMeta, P2Pefield=p2pE,P2Pvoltage=p2pV,P2Pfiltered=p2pFV,HilbertPeakE=peakE,HilbertPeakV=peakV,HilbertPeakFV=peakFV,HilbertPeakTimeE=peaktimeE,HilbertPeakTimeV=peaktimeV,HilbertPeakTimeFV=peaktimeFV)
      #hdf5io.SaveAntennaP2PInfo(OutputFilename,AntennaP2PInfo,CurrentEventName) 

    return desired_traceAll


#-------------------------------------------------------------------

def interpol_check_hdf5(InputFilename, positions, new_pos, p2pE, InterpolMethod,usetrace='efield', DISPLAY=False):
    '''
    Interpolates the signal peak-to-peak electric field at new antenna positions
    Check that the interpolation efficiency at 6 antenna positions available in each shower file

    Parameters:
    InputFilename: str
        HDF5File
    positions: numpy array
        x, y, z coordinates of the antennas in the simulation (not used in trace interpolation method)
    new_pos: numpy array
        x, y, z coordinates of the antennas in new layout (at 6 check points)
    p2pE: numpy array
        [p2p_Ex, p2p_Ey, p2p_Ez, p2p_total]: peak-to-peak electric fields along x, y, z, and norm

    InterpolMethod: str
        interpolation method
        'lin' = linear interpolation from scipy.interpolate
        'rbf' = radial interpolation from scipy.interpolate
        'trace' = interpolation of signal traces:
            generates new interpolated trace files in path/Test/ directory

    DISPLAY: boolean
        if TRUE: 2D maps of peak-to-peak electric field
            at original and interpolated antennas are displayed

    Output:
    interp_err: numpy arrays
        interpolation error at each antenna (interpolated - original)/original
    p2p_total_new: numpy array
        peak-to-peak electric field at new antenna positions

    '''


    # interpolate (check rbf)
    logging.debug('interpol_check:Interpolating...'+str(usetrace))
    #print('Interpolating...'+path)

    number_ant = 160
    icheck = np.mgrid[160:176:1]

    myx_pos = positions[0,0:number_ant-1]
    myy_pos = positions[1,0:number_ant-1]
    myz_pos = positions[2,0:number_ant-1]
    mypositions = np.stack((myx_pos, myy_pos, myz_pos), axis=0)
    myp2p_total = p2pE[3,0:number_ant-1]

    from trace_interpol_hdf5 import do_interpolation_hdf5
    OutputFilename = InputFilename + '.Interpolated.'+str(usetrace)+'.hdf5'

    #do_interpolation(AntPath,new_pos,mypositions,Zenith,Azimuth,phigeo=147.43, thetageo=0.72, shower_core=np.array([0,0,2900]), DISPLAY=False)
    do_interpolation_hdf5(new_pos, InputFilename, OutputFilename, antennamin=0, antennamax=159, EventNumber=0, DISPLAY=DISPLAY, usetrace=usetrace)

    #NewAntNum = size(new_pos)
    #NewAntNum, NewAntPos, NewAntID = get_antenna_pos_zhaires(NewAntPath)
    #NewP2pE = get_p2p(path+"/Test",NewAntNum)

    NewP2pE = hdf5io.get_p2p_hdf5(OutputFilename,antennamax=15,antennamin=0,usetrace=usetrace)

    p2p_total_new = NewP2pE[3,:]
    p2p_x_new = NewP2pE[0,:]
    p2p_y_new = NewP2pE[1,:]
    p2p_z_new = NewP2pE[2,:]

    # checking the interpolation efficiency
    interp_err = abs(p2p_total_new-p2pE[3,icheck])/p2pE[3,icheck]
    interp_errx = abs(p2p_x_new-p2pE[0,icheck])/p2pE[0,icheck]
    interp_erry = abs(p2p_y_new-p2pE[1,icheck])/p2pE[1,icheck]
    interp_errz = abs(p2p_z_new-p2pE[2,icheck])/p2pE[2,icheck]

    #print(np.shape(p2p_total_new))
    #print(np.shape(p2pE[3,icheck]))
    #print(p2pE[3,icheck])
    #print("interp_err = #{}".format(interp_err))


    if (DISPLAY and InterpolMethod!='trace'):
        logging.debug('interpol_check:Plotting...')

        ##### Plot 2d figures of total peak amplitude in positions along North-South and East-West
        fig1 = plt.figure(10,figsize=(5,7), dpi=100, facecolor='w', edgecolor='k')


        ax1=fig1.add_subplot(211)
        name = 'total'
        plt.title(name)
        ax1.set_xlabel('positions along NS (m)')
        ax1.set_ylabel('positions along EW (m)')
        col1=ax1.scatter(positions[0,:],positions[1,:], c=p2pE[3,:],  vmin=min(myp2p_total), vmax=max(myp2p_total),  marker='o', cmap=cm.gnuplot2_r)
        plt.xlim((min(mypositions[0,:]),max(mypositions[0,:])))
        plt.ylim((min(mypositions[1,:]),max(mypositions[1,:])))
        plt.colorbar(col1)
        plt.tight_layout()


        ax2=fig1.add_subplot(212)
        name = 'total interpolated'
        plt.title(name)
        ax2.set_xlabel('x (m)')
        ax2.set_ylabel('y (m)')
        col2=ax2.scatter(new_pos[0,:],new_pos[1,:], c=p2p_total_new,  vmin=np.min(myp2p_total), vmax=np.max(myp2p_total),  marker='o', cmap=cm.gnuplot2_r)
        plt.xlim((min(mypositions[0,:]),max(mypositions[0,:])))
        plt.ylim((min(mypositions[1,:]),max(mypositions[1,:])))
        plt.colorbar(col2)
        plt.tight_layout()


        plt.show(block=False)


        if (interp_err.min() < 1.e-9):
            fig2 = plt.figure(figsize=(5,7), dpi=100, facecolor='w', edgecolor='k')


            ax1=fig2.add_subplot(211)
            name = 'total'
            plt.title(name)
            ax1.set_xlabel('positions along NS (m)')
            ax1.set_ylabel('positions along EW (m)')
            col1=ax1.scatter(positions[0,:],positions[1,:], c=p2pE[3,:],  vmin=min(myp2p_total), vmax=max(myp2p_total),  marker='o', cmap=cm.gnuplot2_r)
            plt.xlim((min(mypositions[0,:]),max(mypositions[0,:])))
            plt.ylim((min(mypositions[1,:]),max(mypositions[1,:])))
            plt.colorbar(col1)
            plt.tight_layout()


            ax2=fig1.add_subplot(212)
            name = 'total interpolated'
            plt.title(name)
            ax2.set_xlabel('x (m)')
            ax2.set_ylabel('y (m)')
            col2=ax2.scatter(new_pos[0,:],new_pos[1,:], c=p2p_total_new,  vmin=np.min(myp2p_total), vmax=np.max(myp2p_total),  marker='o', cmap=cm.gnuplot2_r)
            plt.xlim((min(mypositions[0,:]),max(mypositions[0,:])))
            plt.ylim((min(mypositions[1,:]),max(mypositions[1,:])))
            plt.colorbar(col2)
            plt.tight_layout()


            plt.show(block=False)



    return interp_err, p2p_total_new, interp_errx, p2p_x_new, interp_erry, p2p_y_new, interp_errz, p2p_z_new





def main():
    if ( len(sys.argv)<1 ):
        print("""
            Example on how to do interpolate a signal
                -- read in list of desired poistion
                -- read in already simulated arrazs
                -- find neigbours and perform interpolation
                -- save interpolated trace

            Usage: python3 interpolate.py <path>
            Example: python3 interpolate.py <path>

            path: Filename and path of the input hdf5file
        """)
        sys.exit(0)

    # path to list of desied antenna positions, traces will be stored in that corresponding folder
    #desired  = sys.argv[1]
    #desired=np.array([[ 100., 0., 2900.],[ 0., 100., 2900.]])
    #desired=np.loadtxt("/home/mjtueros/GRAND/GP300/GridShape/Stshp_XmaxLibrary_0.1995_85.22_0_Iron_23/Test/new_antpos.dat",usecols=(2,3,4))

    InputFilename=sys.argv[1]
    print("Input",InputFilename,InputFilename)
    CurrentRunInfo=hdf5io.GetRunInfo(InputFilename)
    CurrentEventName=hdf5io.GetEventName(CurrentRunInfo,0) #using the first event of each file (there is only one for now)
    CurrentAntennaInfo=hdf5io.GetAntennaInfo(InputFilename,CurrentEventName)
    #.
    #one way of putting the antenna information as the numpy array this script was designed to use:
    antennamin=160
    antennamax=176 # WE ARE GETTING THE RANDOM antennas!
    AntID=CurrentAntennaInfo['ID'].data[antennamin:antennamax]
    xpoints=CurrentAntennaInfo['X'].data[antennamin:antennamax]
    ypoints=CurrentAntennaInfo['Y'].data[antennamin:antennamax]
    zpoints=CurrentAntennaInfo['Z'].data[antennamin:antennamax]
    t0points=CurrentAntennaInfo['T0'].data[antennamin:antennamax]
    #
    desired=np.stack((xpoints,ypoints,zpoints,t0points), axis=1)


    OutputFilename=InputFilename+".InterpolatedAntennas.hdf5"
    print("Output",OutputFilename)
    # call the interpolation: Angles of magnetic field and shower core information needed, but set to default values
    #do_interpolation(desired,hdf5file, zenith, azimuth, phigeo=147.43, thetageo=0.72, shower_core=np.array([0,0,2900]), DISPLAY=False)
    do_interpolation_hdf5(desired, InputFilename, OutputFilename, antennamin=0, antennamax=159, EventNumber=0, DISPLAY=True,usetrace='voltage',)

if __name__== "__main__":
  main()

