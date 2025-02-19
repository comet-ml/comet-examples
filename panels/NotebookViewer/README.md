### NotebookViewer

The `NotebookViewer` panel is used to render logged Notebooks, either from
[colab.research.google.com](https://colab.research.google.com/) or
any [Jupyter Notebook](https://jupyter.org/).

Comet will automatically log your Colab notebooks, both as a full
history of commenads as `Code.ipynb', but also as a completed notebook
with images and output. For Jupyter, you can use our
[`cometx config --auto-log-notebook yes`](https://github.com/comet-ml/cometx/blob/main/README.md#cometx-config)


<table>
<tr>
<td>
<img src="https://raw.githubusercontent.com/comet-ml/comet-examples/refs/heads/master/panels/NotebookViewer/notebookviewer.png" 
     style="max-width: 300px; max-height: 300px;">
</img>
</td>
</tr>
</table>

#### Resources

* Example Comet Project: [www.comet.com/examples/foodchatbot-eval](https://www.comet.com/examples/foodchatbot-eval/efa8e134778a456dac2e1a85e1604e13)
* Enable auto-logging of your notebooks in Jupyter:
  * `cometx config --auto-log-notebook yes`
  * [Documentation](https://github.com/comet-ml/cometx/blob/main/README.md#cometx-config)
* Colab Notebooks are logged automatically
* Additional Documentation:
  * [Using Comet in a Notebook](https://dev.comet.com/docs/v2/guides/experiment-management/jupyter-notebook/)
