from comet_ml import Artifact, config
import tensorflow as tf

experiment = config.experiment

def finalize_model(model, x_train, y_train, x_test, y_test, experiment):

    artifact = Artifact(
        name="mnist-model",
        artifact_type="model",
        aliases=["BASELINE"],
        metadata={
            "data": "model_asset"  
        }
    )
    def test_index_to_example(index):
        img = x_test[index].reshape(28,28)
        # log the data to Comet, whether it's log_image, log_text, log_audio, ... 
        data = experiment.log_image(img, name="test_%d.png" % index)

        if data is None:
            return None

        return {"sample": str(index), "assetId": data["imageId"]}
    
    # Add tags
    experiment.add_tag('mnist')
    experiment.add_tag('keras')
    
    # Confusion Matrix
    preds = model.predict(x_test)

    experiment.log_confusion_matrix(y_test, 
                             preds,
                             index_to_example_function=test_index_to_example)
    

    

    # Log Histograms
    for layer in model.layers:
        if layer.get_weights() != []:
            x = layer.get_weights()
            for _, lst in enumerate(x):
                experiment.log_histogram_3d(lst, name=layer.name, step = _)
                
    # Log model as asset and to artifact store
    model.save('models/mnist-nn.h5')
    experiment.log_model('mnist-neural-net', 'models/mnist-nn.h5')
    # artifact.add('models/mnist-nn.h5')
    # experiment.log_artifact(artifact)