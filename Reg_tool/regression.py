import ROOT as r
import numpy as np

def label_output_per(
    label, output, labelbin, labelmin, labelmax, outputbin, outputmin, outputmax,
    h2label="h2", h2name="h2"):
    entry = np.zeros((labelbin, outputbin+2))
    lbin_width = (labelmax-labelmin)/labelbin
    obin_width = (outputmax-outputmin)/outputbin
    for l, o in zip(label, output):
        lbin = int((l-labelmin)//lbin_width)
        obin = int((o-outputmin)//obin_width)
        if obin < 0:
            obin = 0
        elif obin >= outputbin:
            obin = outputbin+1
        else:
            obin = obin+1
        entry[int(lbin)][int(obin)] += 1
    # create label entry
    lentry = entry.sum(axis=1)
    # create TH2D and fill
    h2 = r.TH2D(h2label, h2name, labelbin, 0, labelbin, outputbin+2, 0, outputbin+2)
    for l in range(labelbin):
        for o in range(outputbin+2):
            if lentry[l]!=0:
                h2.Fill(l, o, entry[l, o]/lentry[l])
    h2.SetAxisRange(0, 1, "Z")
    for i in range(1, 13):
        if i==1:
            h2.GetYaxis().SetBinLabel(i, f"Under {outputmin:.1f}")
        elif i==12:
            h2.GetYaxis().SetBinLabel(i, f"Over {outputmax:.1f}")
        else:
            h2.GetXaxis().SetBinLabel(i-1, f"{(labelmin+lbin_width*(i-2)):.1f}~{(labelmin+lbin_width*(i-1)):.1f}")
            h2.GetYaxis().SetBinLabel(i, f"{(outputmin+obin_width*(i-2)):.1f}~{(outputmin+obin_width*(i-1)):.1f}")
    h2.GetXaxis().SetLabelSize(0.04)
    h2.SetTitleOffset(1.5, "X")
    h2.GetYaxis().SetLabelSize(0.04)
    h2.GetXaxis().SetTitle("Trueth Energy(GeV)")
    h2.GetYaxis().SetTitle("Regression Energy(GeV)")
    return h2

def label_par_add(
    h2positive, h2negative,
    labelbin, labelmin, labelmax, outputbin, outputmin, outputmax,
    h2label="h2", h2name="h2"):
    h2 = r.TH2D(h2label, h2name, labelbin, 0, labelbin, outputbin+2, 0, outputbin+2)
    h2.Add(h2positive, h2negative, 1, -1)
    lbin_width = (labelmax-labelmin)/labelbin
    obin_width = (outputmax-outputmin)/outputbin
    for i in range(1, 13):
        if i==1:
            h2.GetYaxis().SetBinLabel(i, f"Under {outputmin:.1f}")
        elif i==12:
            h2.GetYaxis().SetBinLabel(i, f"Over {outputmax:.1f}")
        else:
            h2.GetXaxis().SetBinLabel(i-1, f"{(labelmin+lbin_width*(i-2)):.1f}~{(labelmin+lbin_width*(i-1)):.1f}")
            h2.GetYaxis().SetBinLabel(i, f"{(outputmin+obin_width*(i-2)):.1f}~{(outputmin+obin_width*(i-1)):.1f}")
    h2.GetXaxis().SetLabelSize(0.04)
    h2.SetTitleOffset(1.5, "X")
    h2.GetYaxis().SetLabelSize(0.04)
    h2.GetXaxis().SetTitle("Trueth Energy(GeV)")
    h2.GetYaxis().SetTitle("Regression Energy(GeV)")
    return h2

def energy_error(
    label, output, labelbin, labelmin, labelmax, errorbin, maxerror,
    h2label="h2", h2name="h2"):
    h2 = r.TH2D(h2label, h2name, labelbin, labelmin, labelmax, errorbin, -maxerror, maxerror)
    h2.FillN(len(label), label.astype(np.double), (label-output).astype(np.double), np.ones(len(label)).astype(np.double))
    h2.GetXaxis().SetTitle("Trueth Energy(MeV)")
    h2.GetYaxis().SetTitle("Energy Error(MeV)")
    return h2

class hist_fit_list():
    def __init__(self, min, max, fit_col, nofpara):
        self.h1_label = []
        self.f1_label = []
        self.h1 = []
        self.f1 = []
        self.max = max
        self.min = min
        self.fit_col = fit_col
        self.nofpara = nofpara
        for i in range(nofpara):
            exec(f"self.paralist{i} = []")
            exec(f"self.errorlist{i} = []")

    def add_rdf(self, rdf, h1_label, h1_name, f1_label, columnname="Egap"):
        h1_tmp = rdf.Define("EgapGeV", columnname + "/1000").Histo1D((h1_label, h1_name, 1000, self.min, self.max), "EgapGeV")
        h1_tmp.GetXaxis().SetTitle("Energy(GeV)")
        h1_tmp.GetYaxis().SetTitle("Entry")
        f1_tmp = r.TF1(f1_label, self.fit_col, self.min, self.max)
        h1_tmp.Fit(f1_tmp)
        self.h1.append(h1_tmp)
        self.f1.append(f1_tmp)
        self.h1_label.append(h1_label)
        self.f1_label.append(f1_label)
        for i in range(self.nofpara):
            exec(f"self.paralist{i}.append(f1_tmp.GetParameter({i+1}))")
            exec(f"self.errorlist{i}.append(f1_tmp.GetParError({i+1}))")

    def add_ndarray(self, ndarray, h1_label, h1_name, f1_label):
        h1_tmp = r.TH1D(h1_label, h1_name, 1000, self.min, self.max)
        h1_tmp.FillN(len(ndarray), ndarray.astype(np.double), np.ones(len(ndarray)).astype(np.double))
        h1_tmp.GetXaxis().SetTitle("Energy(GeV)")
        h1_tmp.GetYaxis().SetTitle("Entry")
        f1_tmp = r.TF1(f1_label, self.fit_col, self.min, self.max)
        h1_tmp.Fit(f1_tmp)
        h1_tmp.SetDirectory(0)
        self.h1.append(h1_tmp)
        self.f1.append(f1_tmp)
        self.h1_label.append(h1_label)
        self.f1_label.append(f1_label)
        for i in range(self.nofpara):
            exec(f"self.paralist{i}.append(f1_tmp.GetParameter({i+1}))")
            exec(f"self.errorlist{i}.append(f1_tmp.GetParError({i+1}))")

    def get_h1(self):
        list = self.h1
        return list

    def get_f1(self):
        return self.f1

    def get_fit_par(self):
        parameter = []
        for i in range(self.nofpara):
            exec(f"parameter.append(np.array(self.paralist{i}))")
            exec(f"parameter.append(np.array(self.errorlist{i}))")
        return np.array(parameter)