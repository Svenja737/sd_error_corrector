import evaluate

def compute_metrics(labels, preds):
    # just normal token class metrics, since this is just a token classifier for now 

    metric = evaluate.load("seqeval")
    all_metrics = metric.compute(predictions=preds, references=labels)
    return {
        "Accuracy" : all_metrics["accuracy"],
        "Precision" : all_metrics["precision"],
        "Recall" : all_metrics["recall"],
        "F1" : all_metrics["f1"]
    }
