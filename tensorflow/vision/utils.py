from comet_ml import config
import tensorflow as tf

exp = config.experiment

def finalize_model(model, x_train, y_train, x_test, y_test, exp):

    def test_index_to_example(index):
        img = x_test[index].reshape(28,28)
        # log the data to Comet, whether it's log_image, log_text, log_audio, ... 
        data = exp.log_image(img, name="test_%d.png" % index)
        return {"sample": str(index), "assetId": data["imageId"]}
    
    # Add tags
    exp.add_tag('mnist')
    exp.add_tag('keras')
    
    # Confusion Matrix
    preds = model.predict(x_test)

    exp.log_confusion_matrix(y_test, 
                             preds,
                             index_to_example_function=test_index_to_example)
    

    

    # Log Histograms
    for layer in model.layers:
        if layer.get_weights() != []:
            x = layer.get_weights()
            for _, lst in enumerate(x):
                exp.log_histogram_3d(lst, name=layer.name, step = _)
                
    # Log Model
    model.save('models/mnist-nn.h5')
    exp.log_model('mnist-neural-net', 'models/mnist-nn.h5')
 

