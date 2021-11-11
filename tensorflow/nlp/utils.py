from comet_ml import config
import tensorflow as tf

exp = config.experiment

def finalize_model(model, test_examples, test_labels, experiment):
    
    
    # Log text
    for i in range(20):
        experiment.log_text(test_examples[0], metadata={"label": test_labels[0].item()})
        
    # log confusion matrix
    
    preds = model.predict(test_examples)
    
    def onehot(val):
        retval = [0, 0]
        tmp = (val[0] + 1) / 2
        tmp = int(round(tmp))
        tmp = max(min(1, tmp), 0)
        retval[tmp] = 1
        return retval
    
    new_preds = [onehot(v) for v in preds]
    
    def index_to_example(index):
        text = test_examples[index]
        # data = experiment.log_text(text)
        return {"sample": text.decode(), 
                "assetId": None,
                "type": "string"}
    
    experiment.log_confusion_matrix(new_preds, 
                                test_labels, 
                                index_to_example_function=index_to_example,
                                file_name="movie-reviews")
    
                
    # Log Model
    model.save('models/movie-reviews-nn.h5')
    experiment.log_model('movie-reviews-nn', 'models/movie-reviews-nn.h5')
 

