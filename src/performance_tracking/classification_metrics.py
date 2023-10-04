import evaluate

def compute_metrics(labels, preds):
    # just normal token class metrics, since this is just a token classifier for now 

    acc = evaluate.load("accuracy")
    prec = evaluate.load("precision")
    rec = evaluate.load("recall")
    f1 = evaluate.load("f1")

    accuracy = []
    precision = []
    recall = []
    F1 = []

    for l, p in list(zip(labels, preds)):
        accuracy.append(acc.compute(predictions=p, references=l))
        precision.append(prec.compute(predictions=p, references=l, average="macro", labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], zero_division=0.0))
        recall.append(rec.compute(predictions=p, references=l, average="macro", labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], zero_division=0.0))
        F1.append(f1.compute(predictions=p, references=l, average="macro", labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
    
    return {
        "Accuracy" : sum([x["accuracy"]for x in accuracy])/len(labels),
        "Precision" : sum([x["precision"]for x in precision])/len(labels),
        "Recall" : sum([x["recall"]for x in recall])/len(labels),
        "F1" : sum([x["f1"]for x in F1])/len(labels)
    }
