import ROOT as r
import numpy as np

def sb_histgram(y_pred, y, nhs="hs", ntitle="backgraund", ptitle="signal"):
    positive = y_pred[y==1]
    negative = y_pred[y==0]
    hs = r.THStack(nhs, "CNN output; CNN output; Event")
    h1p = r.TH1D("signal histo", "signal", 100, 0, 1)
    h1n = r.TH1D("backgraund histo", "backgraund", 100, 0, 1)
    leg = r.TLegend(0.78, 0.7, 0.98, 0.90)
    for i in range(len(positive)):
        h1p.Fill(positive[i])
    for i in range(len(negative)):
        h1n.Fill(negative[i])
    h1p.SetLineWidth(2)
    h1n.SetLineWidth(2)
    h1p.SetLineColor(2)
    h1n.SetLineColor(4)
    leg.AddEntry(h1n, ntitle, "L")
    leg.AddEntry(h1p, ptitle, "L")
    hs.Add(h1p)
    hs.Add(h1n)
    return hs, leg

def plot_loss(tloss, vloss, ntmg="tmg"):
    tmg = r.TMultiGraph(ntmg, "Loss; epoch; Loss")
    tgt = r.TGraph(len(tloss), np.arange(len(tloss)).astype(np.float32), tloss.astype(np.float32))
    tgv = r.TGraph(len(vloss), np.arange(len(vloss)).astype(np.float32), vloss.astype(np.float32))
    leg = r.TLegend(0.78, 0.7, 0.98, 0.90)
    tgt.SetMarkerSize(1)
    tgt.SetMarkerColor(2)
    tgt.SetMarkerStyle(20)
    tgt.SetTitle("Training")
    tgv.SetMarkerSize(1)
    tgv.SetMarkerColor(4)
    tgv.SetMarkerStyle(20)
    tgv.SetTitle("Validation")
    leg.AddEntry(tgt,"Trainig", "P")
    leg.AddEntry(tgv, "Validation", "P")
    tmg.Add(tgt)
    tmg.Add(tgv)
    return tmg, leg

def plot_accuracy(tloss, vloss, ntmg="tmg"):
    tmg = r.TMultiGraph(ntmg, "Accuracy; epoch; Accuracy")
    tgt = r.TGraph(len(tloss), np.arange(len(tloss)).astype(np.float32), tloss.astype(np.float32))
    tgv = r.TGraph(len(vloss), np.arange(len(vloss)).astype(np.float32), vloss.astype(np.float32))
    leg = r.TLegend(0.78, 0.7, 0.98, 0.90)
    tgt.SetMarkerSize(1)
    tgt.SetMarkerColor(2)
    tgt.SetMarkerStyle(20)
    tgt.SetTitle("Training")
    tgv.SetMarkerSize(1)
    tgv.SetMarkerColor(4)
    tgv.SetMarkerStyle(20)
    tgv.SetTitle("Validation")
    leg.AddEntry(tgt,"Trainig", "P")
    leg.AddEntry(tgv, "Validation", "P")
    tmg.Add(tgt)
    tmg.Add(tgv)
    return tmg, leg

def plot_roc(y_pred, y):
    threshold = np.linspace(0, 1, 1000)
    tpr = []
    fpr = []
    for i in range(1000):
        anser = (y_pred >= threshold[i]).astype(int)
        data = np.stack([y, anser], 0)
        signal = data[:, data[0]==1]
        fn = (signal[:, signal[1]==0].shape)[1]
        tp = (signal[:, signal[1]==1].shape)[1]
        backgraund = data[:, data[0]==0]
        fp = (backgraund[:, backgraund[1]==1].shape)[1]
        tn = (backgraund[:, backgraund[1]==0].shape)[1]
        tpr.append(tp/(tp+fn))
        fpr.append(fp/(fp+tn))
    tpr = np.array(tpr).astype(np.float32)
    fpr = np.array(fpr).astype(np.float32)

    tg = r.TGraph(1000, tpr, 1-fpr)
    tg.SetTitle("ROC curve")
    tg.GetXaxis().SetTitle("True Positive Rate(TPR)")
    tg.GetYaxis().SetTitle("False Positive Rate(FPR)")
    tg.SetLineWidth(2)
    tg.SetLineColor(4)
    tg.GetXaxis().SetLimits(0, 1.1)
    tg.GetYaxis().SetLimits(0, 1.1)
    return tg

def test_accuracy(y_out, y_true, threshold=0.5):
    nd_threshold = np.full(y_out.shape, threshold)
    result = y_out >= nd_threshold
    result = result.astype(np.int32)
    nd_correct = result == y_true
    correct = np.sum(nd_correct.astype(np.int32))
    correct /= y_out.size
    return correct
