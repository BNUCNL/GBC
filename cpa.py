import os
import nibabel as nib
import numpy as np
from scipy import stats
import neighbor as nb
from scipy.spatial import distance


class UserDefinedException(Exception):
    """
    Exception defined by user
    """

    def __init__(self, str):
        """

        Parameters
        ----------
        str : a string to indicate the exception

        """
        Exception.__init__(self)
        self._str = str


def pearson_correlation(D, w=None):
    """

    Parameters
    ----------
    D : A 2-D array containing multiple variables and observations.
        Each row of `D` represents a variable, and each column a single
        observation of all those variables.

    w : 1-D array of observation vector weights.

    Returns
    -------
    R : 2-D array, the corrcoef matrix of the variables.


    """
    # unweighted correlation
    if w is None:
        R = 1 - distance.cdist(D,D,'correlation')

        # weighted correlation
    else:
        c = np.cov(D, aweights=w)
        d = np.diag(c)
        R = c / np.sqrt(np.outer(d, d))
    
    # fisher-z transformation
    R = np.arctanh(R)
    
    return R


def interquartile_range(D, axis):
    q = np.nanpercentile(D, (25, 75), axis=1)
    iqr = q[:, 1] - q[:, 0]

    return iqr


def load_img(fimg):
    """
    Load Nifti1Image

    Parameters
    ----------
    fimg : a file or a Nifti1Image object

    Returns
    -------
    img : a Nifti1Image object
    """
    if isinstance(fimg, nib.Nifti1Image):
        img = fimg

    # load nifti image with nibabel
    elif os.path.isfile(fimg):
        img = nib.load(fimg)
    else:
        raise UserDefinedException('Wrong Image!')

    return img


class DataSet(object):
    def __init__(self, ftarg_img, fnode_img, flabel_img=None, cond_file=None, level='voxel'):
        """

        Parameters
        ----------
        ftarg_img : target image file(str) or a NifitiImage object
        fnode_img : node image file(str) or a NifitiImage object
        flabel_img : label image file(str) or a NifitiImage object
        cond_file : condition file, str
        level : level of interest, str, voxel or roi

        """
        # load target image
        targ_img = load_img(ftarg_img)
        if len(targ_img.shape) == 4:
            targ = targ_img.get_data()
            self.header = targ_img.header
        else:
            raise UserDefinedException('target image is not a 4D Nifti volume!')

        # load node image
        node_img = load_img(fnode_img)
        if (len(node_img.shape) == 3) and (node_img.shape == targ_img.shape[:3]):
            node = node_img.get_data()

            # node mask
            nmas = (node != 0)
        else:
            raise UserDefinedException('Node image and target image are not match!')

        self.level = level

        # extract info for voxel
        if self.level == 'voxel':
            # time course
            self.tc = targ[nmas, :]
            # node id
            self.nid = node[nmas]
            # node coordinates
            self.ncoords = np.transpose(np.nonzero(node))

        # extract info for roi
        elif self.level == 'roi':
            nid = np.unique(node)
            self.nid = nid[nid != 0]
            tc = np.zeros((self.nid.shape[0], targ.shape[3]))
            for i in range(self.nid.shape[0]):
                tc[i, :] = np.nanmean(targ[node == self.nid[i], :], axis=0)
            self.tc = tc
            self.ncoords = []

        else:
            self.tc = []
            raise UserDefinedException('Wrong level! it should be voxel or roi.')

        # prep label info
        if flabel_img is None:
            self.nlabel = np.ones(self.nid.shape[0])

        else:
            label_img = load_img(flabel_img)
            label = label_img.get_data()

            lmas = (label != 0)

            # compute label for each node
            if (node_img.shape == label_img.shape) and (nmas == lmas).all():
                if self.level == 'voxel':
                    self.nlabel = label[lmas]

                else:
                    self.nlabel = np.zeros((self.nid.shape[0]))
                    # compute ROI label as its mean
                    for i in range(self.nid.shape[0]):
                        self.nlabel[i] = np.mean(label[node == self.nid[i]])

            else:
                raise UserDefinedException('Label image and Node image are not match!')

        # Read design matrix from design file
        if cond_file is None:
            self.cond = None

        # load design info from cond file
        else:
            cond = np.loadtxt(cond_file, skiprows=5)
            self.cond = cond[:, np.arange(0, cond.shape[1] - 6, 2)]

        # node neighbor(nnb) will be assigned in self.compute_nb()
        self.nnb = []

    def compute_nb(self, radius):
        sph = nb.sphere(3, radius, self.header.get_zooms()).compute_offsets().T
        for v in range(0, self.nid.shape[0]):
            idx = nb.in2d(self.ncoords, self.ncoords[v, :] + sph)
            self.nnb.append(np.nonzero(idx)[0])

        return self.nnb

    def set_label(self, label):
        self.nlabel = label

    def set_tc(self, tc):
        self.tc = tc

    def set_cond(self, cond):
        self.cond = cond

    def set_nid(self, nid):
        self.nid = nid


class Connectivity(object):
    def __init__(self, ds, metric='pearson', tm=False):
        """

        Parameters
        ----------
        ds : DataSet object
        metric : metric to compute the connectivity
        tm : is task modulated?

        """

        self.metric = metric
        self.tm = tm
        self.ds = ds
        # mat will be assigned in self.compute()
        self.mat = []

    def compute(self):
        """

        Returns
        -------
        self : A Connectivity object

        """
        ds = self.ds
        if self.metric == 'pearson':
            if not self.tm:
                self.mat = pearson_correlation(ds.tc)

            else:
                self.mat = np.zeros((ds.tc.shape[0], ds.tc.shape[0], ds.cond.shape[1]))

                # standardize  weights to [0,1]
                W = ds.cond
                wmax, wmin = W.max(), W.min()
                W = (W - wmin) / (wmax - wmin)

                # compute connectivity matrix for each condition
                for c in range(W.shape[1]):
                    self.mat[:, :, c] = pearson_correlation(ds.tc, W[:, c])

        elif self.metric == 'wavelet':
            print 'Wavelet metric does not work now.'

        # set main diagnoal to nan
        np.fill_diagonal(self.mat, np.nan)
        
        return self

    def save(self, outdir='.'):
        """

        Parameters
        ----------
        outdir : dir to save the connectivity matrix


        """
        if self.ds.level == 'roi':
            np.savetxt(os.path.join(outdir, self.metric + '.conn'), self.mat, fmt='%.3f')
        else:
            raise UserDefinedException('Voxel level connectivity matrix is too large to save!')

    def set_ds(self, ds):
        self.ds = ds


class Measure(object):
    def __init__(self, conn, metric='sum', ntype='weighted'):
        """

        Parameters
        ----------
        metric: metric to measure the connectivity pattern,str
        thr: threshold, scalar
        ntype: network type, str

        """
        self.metric = metric
        if self.metric == 'sum':
            self.cpu = np.nansum

        elif self.metric == 'mean':
            self.cpu = np.nanmean

        elif self.metric == 'std':
            self.cpu = np.nanstd

        elif self.metric == 'iqr':
            self.cpu = interquartile_range

        elif self.metric == 'skewness':
            self.cpu = stats.skew

        elif self.metric == 'kurtosis':
            self.cpu = stats.kurtosis

        self.conn = conn
        self.ntype = ntype
        self.mtype = 'global'  # measure type

        # these variables will be assigned in self.compute()
        self.partition = []
        self.thr = []
        self.value = []

    def compute(self, thr=None, partition=None):
        """

        Parameters
        ----------
        thr : threshod to remove non-interest edge, scalar
        partition : node partition, 1-D array


        Returns
        -------
        self : A Measure object

        """

        self.thr = thr
        if self.thr is None and (self.ntype == 'binary'):
            raise UserDefinedException('Thr is necessary for binary network!')

        # thresholding the connectivity matrix
        if self.thr is None:
            mat = self.conn.mat
        else:
            mat = self.conn.mat > thr
            if self.ntype == 'weighted':
                mat = mat * self.conn.mat

        if partition is None:
            self.partition = self.conn.ds.nlabel
        else:
            self.partition = partition

        # compute connectivity between and across modules
        P = np.unique(self.partition).tolist()
        for i in P:
            I = np.where(self.partition == i)
            for j in P:
                J = np.where(self.partition == j)
                sub_mat = mat[np.ix_(I[0], J[0])]
                self.value.append(self.cpu(sub_mat, axis=1))

        return self

    def set_conn(self, conn):
        self.conn = conn

    def save(self, outdir='.'):
        """

        Parameters
        ----------
        outdir : dir to save the measures


        """

        P = np.unique(self.partition).tolist() # node partition
        NP = len(P)    # number of node
        R = range(NP)  # node range
        ds = self.conn.ds
        if ds.level == 'roi':
            # convert self.value to 2D array, each column corresponds a seed module
            value = np.zeros((ds.nid.shape[0], NP * NP))
            for i in R:
                I = (self.partition == P[i])
                for j in R:
                    J = int(i * NP + j)
                    value[I, J] = self.value[J]

            np.savetxt(os.path.join(outdir, '_'.join((self.ntype, 'global', self.metric)) + '.cpa'), value, fmt='%.3f')

        elif ds.level == 'voxel':
            dim = ds.header.get_data_shape()[:3]
            value = np.zeros((dim[0], dim[1], dim[2], NP * NP))

            # convert self.value to 4D volume, each 3D volume corresponds a seed module
            for i in R:
                I = ds.ncoords[self.partition == P[i], :]
                for j in R:
                    J = int(i * NP + j)
                    value[I[:, 0], I[:, 1], I[:, 2], J] = self.value[J]

            # remove nan
            value[np.isnan(value)] = 0

            # save voxelwise inter-module measure in 4D volume
            header = ds.header
            header['cal_max'] = value.max()
            header['cal_min'] = value.min()
            img = nib.Nifti1Image(value, None, header)
            nib.save(img, os.path.join(outdir, '_'.join((self.ntype, self.mtype, self.metric)) + '.nii.gz'))


class LocalMeasure(object):
    def __init__(self, conn, radius=6, metric='sum', ntype='weighted'):
        """

        Parameters
        ----------
        metric: metric to measure the connectivity pattern,str
        thr: threshold, scalar
        ntype: network type, str
        radius: radius for local neighbor(sphere), scalar,unit is mm
        """
        self.metric = metric
        if self.metric == 'sum':
            self.cpu = np.nansum
        elif self.metric == 'mean':
            self.cpu = np.nanmean
        elif self.metric == 'std':
            self.cpu = np.nanstd
        elif self.metric == 'skewness':
            self.cpu = stats.skew
        elif self.metric == 'kurtosis':
            self.cpu = stats.kurtosis

        self.conn = conn
        self.ntype = ntype
        self.radius = radius
        self.mtype = 'local'  # measure type

        # Variables will be assigned in self.compute()
        self.thr = []
        self.value = []


    def compute(self, thr=None):
        """

        Parameters
        ----------
        thr : threshod to remove non-interest edge, scalar

        Returns
        -------
        self : A LocalMeasure object

        """

        self.thr = thr
        if self.thr is None and (self.ntype == 'binary'):
            raise UserDefinedException('Thr is necessary for binary network!')

        # thresholding the connectivity mat
        if self.thr is None:
            mat = self.conn.mat
        else:
            mat = self.conn.mat > thr
            if self.ntype == 'weighted':
                mat = mat * self.conn.mat

        # compute local neighbor for each node
        nnb = self.conn.ds.compute_nb(self.radius)

        # compute local measure for each voxel
        self.value = np.zeros(mat.shape[0])
        for i in range(mat.shape[0]):
            self.value[i] = self.cpu(mat[i, nnb[i]])

        return self

    def set_conn(self, conn):
        self.conn = conn

    def save(self, outdir='.'):
        """

        Parameters
        ----------
        outdir : dir to save the measures


        """
        ds = self.conn.ds
        # convert self.value to 3D array
        dim = ds.header.get_data_shape()
        value = np.zeros((dim[0], dim[1], dim[2]))

        value[ds.ncoords[:, 0], ds.ncoords[:, 1], ds.ncoords[:, 2]] = self.value

        # save voxel-wise inter-module measure in 4D volume
        header = ds.header
        header['cal_max'] = value.max()
        header['cal_min'] = value.min()
        img = nib.Nifti1Image(value, None, header)
        nib.save(img, os.path.join(outdir, '_'.join((self.ntype, self.mtype, self.metric)) + '.nii.gz'))


class SpatialMeasure(Measure):
    def __init__(self, conn, metric='sum', ntype='weighted'):
        """

        Parameters
        ----------
        metric: metric to measure the connectivity pattern,str
        thr: threshold, scalar
        ntype: network type, str
        radius: radius for local neighbor(sphere), scalar,unit is mm
        """

        super(SpatialMeasure, self).__init__(conn, metric, ntype)
        self.mtype = 'spatial'

    def compute(self, thr=None, partition=None):
        """

        Parameters
        ----------
        thr : threshod for conn to remove non-interest edge, scalar
        partition : node partition, 1-D array


        Returns
        -------
        self : A LocalMeasure object

        """

        if thr is None:
            self.thr = 0
        else:
            self.thr = thr

        # thresholding the funcitonal connectivity mat
        mat = self.conn.mat < self.thr

        # compute the spatial distance mat
        ncoords = self.conn.ds.ncoords
        dist = distance.pdist(ncoords, 'euclidean')
        dist = distance.squareform(dist)
        dist[mat] = np.NaN

        if partition is None:
            self.partition = self.conn.ds.nlabel
        else:
            self.partition = partition

        # compute spatial distance between and across modules
        P = np.unique(self.partition).tolist()
        for i in P:
            I = np.where(self.partition == i)
            for j in P:
                J = np.where(self.partition == j)
                sub_mat = dist[np.ix_(I[0], J[0])]
                self.value.append(self.cpu(sub_mat, axis=1))

        return self

    def set_conn(self, conn):
        self.conn = conn
