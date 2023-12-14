import evaluate

def compute_metrics(labels, preds, binary=False):
    # just normal token classification metrics, since the model is a token classifier

    acc = evaluate.load("accuracy")
    prec = evaluate.load("precision")
    rec = evaluate.load("recall")
    f1 = evaluate.load("f1")

    accuracy = []
    precision = []
    recall = []
    F1 = []

    if binary == False:
        for l, p in list(zip(labels, preds)):
            accuracy.append(acc.compute(predictions=p, references=l))
            precision.append(prec.compute(predictions=p, references=l, average="macro"))
            recall.append(rec.compute(predictions=p, references=l, average="macro"))
            F1.append(f1.compute(predictions=p, references=l, average="macro"))
    else:
        for l, p in list(zip(labels, preds)):
            accuracy.append(acc.compute(predictions=p, references=l))
            precision.append(prec.compute(predictions=p, references=l, average="binary"))
            recall.append(rec.compute(predictions=p, references=l, average="binary"))
            F1.append(f1.compute(predictions=p, references=l, average="binary"))

    return {
        "Accuracy" : sum([x["accuracy"]for x in accuracy])/len(labels),
        "Precision" : sum([x["precision"]for x in precision])/len(labels),
        "Recall" : sum([x["recall"]for x in recall])/len(labels),
        "F1" : sum([x["f1"]for x in F1])/len(labels)
    }
