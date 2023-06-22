# Using Comet with Sagemaker

Sagemaker is Amazon's end-to-end machine learning service that targets a large swath of data science and machine learning practitioners. With Sagemaker, data scientists and developers can build and train machine learning models, and directly deploy them into a production-ready hosted environment. Sagemaker's product offerings span the needs of data/business analysts, data scientists, and machine learning engineers.

Comet is an excellent complement to Sagemaker, enhancing the developer experience by allowing users to easily track experiments, collaborate with team members, and visualize results in an intuitive and easy-to-understand way while using the frameworks and tools that they are most comfortable with. Additionally, the platform provides a wide range of customization options, including the ability to create custom visualizations and dashboards, so that users can tailor their experience to meet their specific needs.

By using Comet, users can streamline their workflows while benefiting from Sagemaker's powerful infrastructure orchestration and model deployment capabilities.

## Logging Custom Scripts with Comet and Sagemaker
Comet requires minimal changes to your existing Sagemaker workflow in order to get up and running. Let’s take a look at a simple example that uses the Sagemaker SDK and Notebook instances to run a custom script.

```
├── src
│   ├── train.py
│   └── requirements.txt
└── launch.ipynb
```

Your `src` directory would contain the model specific code needed to execute your training run, while `launch.ipynb` would run in your Notebook instance, and contain code related to configuring and launching your job with the Sagemaker SDK.

To enable Comet logging in this workflow, simply

1. Add `comet_ml` as a dependency in your `requirement.txt` file
2. Import the `comet_ml` library at the top of the `train.py` script
3. Create a Comet `Experiment` object within the training script
4. Pass in your Comet Credentials to the Sagemaker Estimator using the environment argument.
5. Launch your training job in Sagemaker using `estimator.fit`

### Examples

- [Image Classification with Pytorch](/integrations/model-training/sagemaker/log_custom_scripts/pytorch-mnist)
- [Image Classification with Tensorflow](/integrations/model-training/sagemaker/log_custom_scripts/tensorflow-mnist)
- [Text Classification with HuggingFace](/integrations/model-training/sagemaker/log_custom_scripts/huggingface-text-classification)



