import evaluate

def compute_metrics(labels, preds):
    # just normal token class metrics, since this is just a token classifier for now 

    metric = evaluate.load("seqeval")
    all_metrics = metric.compute(predictions=preds, references=labels)
    
    return {
        "precision": all_metrics["overall_precision"],
        "recall": all_metrics["overall_recall"],
        "f1": all_metrics["overall_f1"],
        "accuracy": all_metrics["overall_accuracy"],
    }
