import numpy as np
import geopandas as gpd
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm
import os
from scipy import signal


class readTransformerData:
    def __init__(self, dataprepinargs):
        self.dataparam = dataprepinargs

    def readfiles(self):
        timedir = self.dataparam["time"]
        covardir = self.dataparam["covar"]
        responsedir = self.dataparam["response"]
        self.time = np.swapaxes(np.load(timedir), 1, 2)
        self.covar = np.load(covardir)
        if self.dataparam["addPGV"]:
            PGV = np.max(self.time, axis=1)
            PGV = np.power(PGV.prod(axis=1), 1.0 / 3.0)
            self.covar = np.hstack([self.covar, np.expand_dims(PGV, axis=-1)])
        if self.dataparam["onlyPGV"]:
            PGV = np.max(self.time, axis=1)
            self.covar = np.expand_dims(np.power(PGV.prod(axis=1), 1.0 / 3.0), axis=-1)
        self.response = np.load(responsedir)[:, 0]

    def readfilesUSGS(self):
        usgsmetric = self.dataparam["usgsmetric"]
        covardir = self.dataparam["covar"]
        responsedir = self.dataparam["response"]
        self.usgsgm = np.load(usgsmetric)
        self.covar = np.load(covardir)
        # print(self.usgsgm.shape)
        # print(self.covar.shape)
        if self.dataparam["addPGV"]:
            self.covar = np.hstack([self.covar, self.usgsgm])
        if self.dataparam["onlyPGV"]:
            self.covar = self.usgsgm
        self.time = self.usgsgm
        self.response = np.load(responsedir)[:, 0]

    def readfilesFilter(self):
        timedir = self.dataparam["time"]
        covardir = self.dataparam["covar"]
        responsedir = self.dataparam["response"]
        flt_freq = self.dataparam["fltfreq"]
        dt = self.dataparam["dt"]
        self.time = np.swapaxes(np.load(timedir), 1, 2)
        print(self.time.shape)
        self.covar = np.load(covardir)
        max_freq = 20 * flt_freq * dt
        b, a = signal.butter(4, max_freq, btype="lowpass")
        self.time = signal.filtfilt(b, a, self.time, axis=1)
        print(self.time.shape)
        if self.dataparam["addPGV"]:
            PGV = np.max(self.time, axis=1)
            PGV = np.power(PGV.prod(axis=1), 1.0 / 3.0)
            self.covar = np.hstack([self.covar, np.expand_dims(PGV, axis=-1)])
        if self.dataparam["onlyPGV"]:
            PGV = np.max(self.time, axis=1)
            self.covar = np.expand_dims(np.power(PGV.prod(axis=1), 1.0 / 3.0), axis=-1)
        self.response = np.load(responsedir)[:, 0]

    def readfilesArias(self):
        ariasdir = self.dataparam["ariasmetric"]
        covardir = self.dataparam["covar"]
        responsedir = self.dataparam["response"]
        self.ariasintensity = np.load(ariasdir)
        self.covar = np.load(covardir)
        if self.dataparam["addPGV"]:
            self.covar = np.hstack([self.covar, self.ariasintensity])
        if self.dataparam["onlyPGV"]:
            self.covar = self.ariasintensity
        self.response = np.load(responsedir)[:, 0]
        self.time = self.ariasintensity

    def readfilesTerrain(self):
        covardir = self.dataparam["covar"]
        responsedir = self.dataparam["response"]
        self.covar = np.load(covardir)
        self.response = np.load(responsedir)[:, 0]
        self.time = self.response

    def preparedata(self):
        usgs = self.dataparam.get("USGS", False)
        arias = self.dataparam.get("Arias", False)
        filter = self.dataparam.get("Filter", False)
        terrain = self.dataparam.get("terrain", False)
        if usgs:
            self.readfilesUSGS()
        elif arias:
            self.readfilesArias()
        elif filter:
            self.readfilesFilter()
        elif terrain:
            self.readfilesTerrain()
        else:
            self.readfiles()

        if self.dataparam["removezeros"]:
            idx = np.where(self.response > 0)[0]
            self.time = self.time[idx]
            self.covar = self.covar[idx]
        if self.dataparam["inference"]:
            self.Xt = self.time
            self.Xc = self.covar
            self.Y = self.response
        else:
            (
                self.Xt_train,
                self.Xt_test,
                self.Xc_train,
                self.Xc_test,
                self.Y_train,
                self.Y_test,
            ) = train_test_split(
                self.time,
                self.covar,
                self.response,
                test_size=self.dataparam["testsize"],
                random_state=420,
            )
        self.covars = None
        self.time = None
        self.response = None
