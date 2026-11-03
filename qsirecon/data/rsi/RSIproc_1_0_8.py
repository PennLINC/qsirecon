#!/usr/bin/env python
# coding: utf-8

# ## Restriction Spectrum Imaging
# Richard Watts, Yale University, August 2020-April 2023
# 
# Based on
# <UL>
#    <LI>White et al, Probing Tissue Microstructure With Restriction Spectrum Imaging: Histological and Theoretical  Validation. HBM 2013 34:327-346
#   <LI>Hagler et al, Image processing and analysis methods for the Adolescent Brain Cognitive Development Study. Neuroimage 2019 116091
#   <LI>Descoteaux et al, Regularized, Fast, and Robust Analytical Q-Ball Imaging. MRM 2008 58:497-510
# </UL>

# In[1]:


import numpy as np
import nibabel as nib
import os, sys
from scipy.special import sph_harm

from icosahedron import make_icosahedron

np.set_printoptions(threshold=np.inf)


if len(sys.argv) < 5:
    print ('{0} data_ecc.nii.gz b0_brain_mask.nii.gz bval bvec'.format(sys.argv[0]))
    sys.exit()

dataFile = sys.argv[1]
maskFile = sys.argv[2]
bvalFile = sys.argv[3]
bvecFile = sys.argv[4]


RSI_lambda = 0.1
uniformRegularization = True

RSI_ADC_long  = np.array([ 1.0e-3, 1.0e-3, 3.0e-3])
RSI_ADC_trans = np.array([1.0e-10, 0.9e-3, 3.0e-3])
RSI_SH_order  = np.array([      4,      4,     0])



normalize = False # Normalize to L2norm?
removeFirst = False
debug = False


RSI_n_comp = RSI_ADC_long.size
nSphHarm = (RSI_SH_order+2)*(RSI_SH_order+1)//2
nFit = np.sum(nSphHarm)

if debug:
    print(nSphHarm)


def Reval(bval, bvec, x, DL, DT):
    cos_alpha = np.dot(bvec, x)
    t = np.exp(-bval*( (DL-DT)*(cos_alpha**2) + DT ))
    return t


def YLeval(L, ML, x):
    phi = np.arccos(x[2])
    
    theta = np.arctan2(x[1], x[0])
    theta = theta + (theta<0)*2.0*np.pi

    if ML<0:
        YL = np.sqrt(2.0)*sph_harm(ML,L,theta,phi).real
    elif ML == 0:
        YL = np.real(sph_harm(0,L,theta,phi))
    else:
        YL = np.sqrt(2.0)*sph_harm(ML,L,theta,phi).imag
    #print(theta)
    return YL


def calculateA(bval, bvec):
    x = make_icosahedron(3)
    M = x.shape[0]

    if debug:
        print(M)

    maxOrder = np.max(RSI_SH_order)

    nSphHarmMax = (maxOrder+2)*(maxOrder+1)//2
    YL = np.zeros(shape=(M, nSphHarmMax))
    reg = np.zeros(shape=(nSphHarmMax))

    for L in range(0,maxOrder+2,2):
        for ML in range(-L, L+1):
            p = (L*L + L)//2 + ML
        
            # Laplace-Beltrami Regularization factors
            reg[p] = L*L*(L+1)*(L+1)
            if debug:
                print(L, ML, p)
            for m in range(M):
                YL[m,p] = YLeval(L, ML, x[m,:])

    R = np.zeros(shape=(RSI_n_comp, Nmeas,M))
    for indexDT in range(RSI_n_comp):
        DL = RSI_ADC_long[indexDT]
        DT = RSI_ADC_trans[indexDT]
        if debug:
            print(DL, DT)

        for n in range(Nmeas):
            for m in range(M):
                R[indexDT,n,m] = Reval(bval[n], bvec[n,:], x[m,:], DL, DT)

    for indexDT in range(RSI_n_comp):
        if indexDT==0:
            # Updated 1/31/2024
            # A = np.matmul(R[indexDT,:,:], YL[:,0:nSphHarm[indexDT]]) # Incorrect
            A = np.matmul(R[indexDT,:,:], np.linalg.pinv(YL[:,0:nSphHarm[indexDT]]).T)
            alpha = reg[0:nSphHarm[indexDT]]
        else:
            # Updated 1/31/2024
            # A = np.hstack((A, np.matmul(R[indexDT,:,:], YL[:,0:nSphHarm[indexDT]]))) # Incorrect
            A = np.hstack((A, np.matmul(R[indexDT,:,:], np.linalg.pinv(YL[:,0:nSphHarm[indexDT]]).T)))
            alpha = np.hstack((alpha, reg[0:nSphHarm[indexDT]]))
    # Update 1/31/2024
    return A*YL[0,0], alpha


def calculateW(A):
    AtA = np.matmul(A.T, A)
    scaleF = np.mean(np.diag(AtA))

    if debug:
        print(scaleF)

    if uniformRegularization:
        W = np.matmul(np.linalg.inv(AtA + RSI_lambda*scaleF*np.identity(nFit)), A.T)
    else:
        W = np.matmul(np.linalg.inv(AtA + RSI_lambda*scaleF*np.diag(alpha)), A.T)

    if debug:
        print(A.shape)
        print(W.shape)
        print(np.matmul(W, A))

    np.savetxt('A.txt', A)
    np.savetxt('W.txt', W)
         
    #U, S, Vh = np.linalg.svd(A, full_matrices=True)
    #print(f'Singular values of A\n{S}\n')
    #print(f'{S[0]/S[-1]}')  

    #U, S, Vh = np.linalg.svd(At, full_matrices=True)
    #print(f'Singular values of At\n{S}\n')
    #print(f'{S[0]/S[-1]}')     
    return W


            


# Load data from file
bval = np.genfromtxt(bvalFile)
bvec = np.genfromtxt(bvecFile).T
data = nib.load(dataFile).get_fdata()
img = nib.load(maskFile)
mask = img.get_fdata()

if removeFirst:
    bval = bval[1:]
    bvec = bvec[1:,:]
    data = data[:,:,:,1:]

datab0 = data[:,:,:,bval<=10].mean(axis=3)
maxb = np.max(bval)

if debug:
    print(data.shape)
xres, yres, zres, Nmeas = data.shape

#print(np.linalg.norm(bvec, axis=1))
#bval = np.round(bval,-2)
#print(bval)

A, alpha = calculateA(bval, bvec)
W = calculateW(A)


n0 = np.zeros(shape=(RSI_n_comp, xres,yres,zres))
nd = np.zeros(shape=(RSI_n_comp, xres,yres,zres))
nt = np.zeros(shape=(RSI_n_comp, xres,yres,zres))
n0d = np.zeros(shape=(RSI_n_comp, xres,yres,zres))
sphharm = np.zeros(shape=(xres,yres,zres,nFit))
L2norm = np.zeros(shape=(xres,yres,zres))


# "Each of these measures is defined as the Euclidean norm (square root of the sum of squares) of the corresponding model coefficients divided by the norm of all model coefficients. These normalized RSI measures are unitless and range from 0 to 1. The square of each of these measures is equivalent to the signal fraction for their respective model components."

for iz in range(zres):
    if ((iz % 10)==0) and (debug):
        print('Processing slice {0}'.format(iz))
    for iy in range(yres):
        for ix in range(xres):
            if (mask[ix,iy,iz]>0.0) and (datab0[ix,iy,iz]>0.0):
                s = data[ix,iy,iz, :] / datab0[ix,iy,iz]
                beta = np.matmul(W, s)
                L2norm[ix,iy,iz] = np.linalg.norm(beta)

                if normalize:
                    norm = L2norm[ix,iy,iz]
                else:
                    norm = 1.0
                    
                indexCount = 0
                for c in range(RSI_n_comp):
                    # Isotropic
                    n0[c, ix,iy,iz] = beta[indexCount]/norm
                    if RSI_SH_order[c]>0:
                        # Anisotropic
                        nd[c, ix,iy,iz] = (np.linalg.norm(beta[indexCount+1:indexCount+nSphHarm[c]]))/norm

                    # Total
                    nt[c, ix,iy,iz] = (np.linalg.norm(beta[indexCount:indexCount+nSphHarm[c]]))/norm
                    indexCount = indexCount + nSphHarm[c]
                                
                sphharm[ix,iy,iz,:] = beta

if not normalize: # Get rid of really large impossible values isotropic components, should be 0-1
    n0[n0>2.0] = 2.0 # 
    n0[n0<-1.0] = -1.0


img.header.set_data_dtype(np.float32)

for c in range(RSI_n_comp):
    newimg = nib.Nifti1Image(np.squeeze(n0[c,:,:,:]), img.affine, img.header)
    nib.save(newimg, 'n0s{0:0d}.nii.gz'.format(c+1))
    if RSI_SH_order[c]>0:
        newimg = nib.Nifti1Image(np.squeeze(nd[c,:,:,:]), img.affine, img.header)
        nib.save(newimg, 'nds{0:0d}.nii.gz'.format(c+1))

    newimg = nib.Nifti1Image(np.squeeze(nt[c,:,:,:]), img.affine, img.header)
    nib.save(newimg, 'nts{0:0d}.nii.gz'.format(c+1))

newimg = nib.Nifti1Image(L2norm, img.affine, img.header)
nib.save(newimg, 'L2norm.nii.gz'.format(c+1))





