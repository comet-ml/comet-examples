### AudioCompare

The `AudioCompare` panel is used to examine audio waveforms and spectrograms
in a single experiment or across experiments. See also the built-in Audio Panel.


<table>
<tr>
<td>
<img src="https://raw.githubusercontent.com/comet-ml/comet-examples/refs/heads/master/panels/AudioCompare/audio-compare.png" 
     style="max-width: 300px; max-height: 300px;">
</img>
</td>

<td>
<img src="https://raw.githubusercontent.com/comet-ml/comet-examples/refs/heads/master/panels/AudioCompare/built-in-audio-panel.png" 
     style="max-width: 300px; max-height: 300px;">
</img>
</td>

</tr>
</table>

#### Python Panel

To include this panel from the github repo, use this code in a Custom Python Panel:

```
%include https://raw.githubusercontent.com/comet-ml/comet-examples/refs/heads/master/panels/AudioCompare/AudioCompare.py
```

Or, you can simply [copy the code](https://raw.githubusercontent.com/comet-ml/comet-examples/refs/heads/master/panels/AudioCompare/AudioCompare.py) into a custom Python Panel.

#### Resources

* Example Comet Project: [www.comet.com/examples/comet-example-audio-compare](https://www.comet.com/examples/comet-example-audio-compare/view/pV46hu7kzY8kOsC77ZWMDJwic/panels)
* [Logging Audio](https://www.comet.com/docs/v2/api-and-sdk/python-sdk/reference/Experiment/#comet_ml.Experiment.log_audio)
* [UI Audio Tab](https://www.comet.com/docs/v2/guides/comet-ui/experiment-management/single-experiment-page/#audio-tab)
* [Get audio assets programmatically](https://www.comet.com/docs/v2/api-and-sdk/python-sdk/reference/APIExperiment/#comet_ml.APIExperiment.get_asset_list)