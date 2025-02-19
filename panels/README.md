### NotebookViewer

The `NotebookViewer` panel is used to render logged Notebooks, either from
[colab.research.google.com](https://colab.research.google.com/) or
any [Jupyter Notebook](https://jupyter.org/).

Comet will automatically log your Colab notebooks, both as a full
history of commenads as `Code.ipynb', but also as a completed notebook
with images and output. For Jupyter, you can use our
[cometx config --auto-log-notebook yes](https://github.com/comet-ml/cometx/blob/main/README.md#cometx-config)


<table>
<tr>
<td>
<img src="https://raw.githubusercontent.com/comet-ml/comet-examples/refs/heads/master/panels/NotebookViewer/notebookviewer.png" 
     style="max-width: 300px; max-height: 300px;">
</img>
</td>
</tr>
</table>


For more information, see the panel <a href="https://github.com/comet-ml/comet-examples/blob/master/panels/NotebookViewer/README.md">README.md</a>
### TotalFidelityMetricPlot

The `TotalFidelityMetricPlot` panel is used to plot Total Fidelity Metrics --- metrics that are not sampled in any way.

You can have your Comet Adminstrator turn on "Store metrics without sampling" in the `Admin Dashboard` => `Organization settings`.

<table>
<tr>
<td>
<img src="https://raw.githubusercontent.com/comet-ml/comet-examples/refs/heads/master/panels/TotalFidelityMetricPlot/totalfidelity.png" 
     style="max-width: 300px; max-height: 300px;">
</img>
</td>
<td>
<img src="https://raw.githubusercontent.com/comet-ml/comet-examples/refs/heads/master/panels/TotalFidelityMetricPlot/organization-settings.png" 
     style="max-width: 300px; max-height: 300px;">
</img>
</td>
</tr>
</table>


For more information, see the panel <a href="https://github.com/comet-ml/comet-examples/blob/master/panels/TotalFidelityMetricPlot/README.md">README.md</a>
### DataGridViewer

The `DataGridViewer` panel is used to visualize Comet `DataGrids` which
can contain Images, text, and numeric data.

The UX is a sophisticated approach to grouping data to see (and select)
images and other data in a tabular format, with a search feature that
allows fast querying of the data (including metadata) using Python syntax.

<table>
<tr>
<td>
<img src="https://raw.githubusercontent.com/comet-ml/comet-examples/refs/heads/master/panels/DataGridViewer/tabular-view.png" 
     style="max-width: 300px; max-height: 300px;">
</img>
</td>
<td>
<img src="https://raw.githubusercontent.com/comet-ml/comet-examples/refs/heads/master/panels/DataGridViewer/group-by.png" 
     style="max-width: 300px; max-height: 300px;">
</img>
</td>
<td>
<img src="https://raw.githubusercontent.com/comet-ml/comet-examples/refs/heads/master/panels/DataGridViewer/image-dialog.png" 
     style="max-width: 300px; max-height: 300px;">
</img>
</td>
</tr>
</table>


For more information, see the panel <a href="https://github.com/comet-ml/comet-examples/blob/master/panels/DataGridViewer/README.md">README.md</a>
