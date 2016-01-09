# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:l

"""Connectivity pattern analysis pipeline. """

import os
import nibabel as nib
import numpy as np
from scipy import stats


class CPAexception(Exception):
    """CPA exception.
    """
    def __init__(self, str):
        Exception.__init__(self)
        self._str = str



class DataSet(object):
    def __init__(self, targ_img_file, mask_img_file, cond_file = None):

        """
        Parameters
        ----------
        targ_img_file: target image file
        mask_img_file: mask image file
        cond_file: condtion file

        Returns
        -------

        """
        self.ftarg_img = targ_img_file
        self.fmask_img = mask_img_file
        self.fcond = cond_file
        self.tc = []
        self.label = []
        self.header = []


    def load(self, level = 'voxel'):
        """

        Parameters
        ----------
        level: node level, i.e., voxel or level

        Returns
        -------

        """
        # load target image
        targ_img = nib.load(self.ftarg_img)
        if len(targ_img.get_shape()) != 4:
            raise CPAexception('targ_img is not a Nifti image about 4D volume!')
        targ = targ_img.get_data()
        self.header = targ_img.header

        # load  mask image
        mask_img = nib.load(self.fmask_img)
        if len(mask_img.get_shape()) != 3:
            raise CPAexception('mask_img is not a Nifti image about 3D volume!')
        mask = mask_img.get_data()


        self.level = level
        if self.level == 'voxel':
            self.tc = targ[mask.astype(np.bool), :]
            # label for each non-zeros voxel
            self.label = mask[mask.astype(np.bool), :]

        elif self.level == 'roi':
            label  = np.unique(mask)
            # label for each ROI
            self.label = label[1:]
            # compute ROI time course
            tc = np.zeros(targ_img.get_shape()[3], len(self.label))
            for i in range(0,len(self.label)):
                tc[:,i] =  np.mean(targ[mask[mask.astype(np.bool)] == i,:])
            self.tc = tc
        else:
            print 'wrong level'


        #Read design matrix from design.mat file
        if self.fcond is not None:
            self.cond = np.loadtxt(self.fcond,skiprows=5)

            #Choose first, third and fifth column as input weights
            # dm = dm[:,[0,2,4]]




        # read cond file and assigh cond array to ds.cond
        self.cond = 'cond'



    def set_tc(self, tc):
        self.tc = tc

    def set_cond(self,cond):
        self.cond = cond

    def  set_label(self, label):
        self.label = label

    def set_header(self,header):
        self.header = header



class Connectivity(object):
    def __init__(self, metric = 'pearson', tm = False):
        self.metric = metric
        self.tm = tm
        self.mat = []

    def compute(self,ds):
        if not self.tm:
            if self.metric == 'pearson':
                self.mat =  stats.pearsonr(ds.tc,'correlation')

            elif self.metric == 'wavelet':
                self.mat = 'wavelet'
        else:
            cond_num = ds.cond.shape[1]
            # calculate weighted correlation for each condition
            for c in range(0,cond_num):
                self.mat = 'weighted_corr(ds.tc,ds.cond[:,c])'


 class Measure(object):
     def __init__(self, metric = 'sum', ntype = 'weighted'):
        """

        Parameters
        ----------
        metric: metric to measure the connectivity pattern,str
        thr: threshold, scalar
        module: module assignment, 1-D array
        type: network type, str

        Returns
        -------

        """
        self.metric = metric
        if self.metric == 'sum':
            self.cpu = np.nansum
        elif self.metric == 'std':
            self.cpu = np.std
        elif self.metric == 'skewness':
            self.cpu = stats.skew
        elif self.metric == 'kurtosis':
            self.cpu= stats.kurtosis

        self.ntype = ntype
        self.thr = []
        self.module = []


     def compute(self,conn, thr = None, module = None):

         self.thr = thr
         if self.thr is None:
             mat = conn.mat
         else:
             mat = conn.mat > thr

         if self.ntype == 'weighted':
             mat = np.multiply(mat,conn.mat)

         # module ID for each node
         if module is None:
             module = np.ones(conn.mat.shape[0])
         self.module = module

         M = np.unique(self.module)
         self.value = []
         for i in M:
             I = np.asarray(self.module == i).T
             for j in M:
                 J = np.asarray(self.module == j)
                 sub_mat = mat[I].reshape(I.shape[0],-1)[:,J].reshape(I.shape[0],-1)

                 self.value.append(self.cpu(sub_mat))


class CPA(object):
    """
    Connectivity pattern analysis(cpa) class

    Attributes
    ----------
    ds : DataSet object
    conn : Connectivity object
    meas: Measure object

    """
    def __init__(self, dataset, conn, measure):

        """
        Parameters
        ----------
        dataset: a dataset object

        Returns
        -------

        """

        self.ds = dataset
        self.conn = conn
        self.meas = measure


    def comp_conn(self):
        # compute connectivity pattern
        self.conn.compute(self.ds)



    def meas_conn(self, thr = None, module = None):
        """
        measure connectivity pattern

        Parameters
        ----------
        thr: threshold to remove moise edge, scalar
        module: module assignment

        Returns
        -------

        """
        self.meas.compute(self.conn, thr, module)

    def save(self, outdir):
        """
        Save measure value to hard disk
        Parameters
        ----------
        outdir: dir to save the connectivity measures

        Returns
        -------

        """
        if self.ds.level == 'roi':
        # save meas.value as txt

        elif self.ds.level == 'voxel':
            # reshape meas.value and save it as nii
            nib.save(self.meas,os.path.join(outdir,self.meas.metric+'.nii.gz'))



    # def meas_anat(self):
    #     print 'anat measure'
