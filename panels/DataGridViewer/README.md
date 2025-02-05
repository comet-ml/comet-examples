### DataGridViewer

The `DataGridViewer` panel is used to visualize Comet `DataGrids` which
can contain Images, text, and numeric data.

The UX is a sophisticated approach to grouping data to see (and select)
images and other data in a tabular format, with a search feature that
allows fast querying of the data (including metadata) using Python syntax.

#### Snapshots

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

#### Example Code

```
%pip install comet_ml datagrid

import comet_ml
from datagrid import DataGrid, Image
import requests

experiment = comet_ml.start(
    project_name="demo-datagrids"
)
dg = DataGrid(
    columns=["Image", "Score", "Category"],
    name="Demo"
)
url = f"https://picsum.photos/300/200"
for i in range(50):
    im = PImage.open(requests.get(url, stream=True).raw)
    category = random.choice(categories)
    score = random.random()
    label = random.choice(items)
    image = Image(
        im,
        metadata={"category": category, "score": score},
    )
    dg.append([image, score, category])

dg.log(experiment)
experiment.end()
```

#### Resources

* Copy panel to workspace:
  * `cometx log YOUR-WORKSPACE DataGridViewer --type panel`
  * Via code (see notebook below)
* Example notebook: [DataGridViewer.ipynb](https://github.com/comet-ml/comet-examples/blob/master/panels/DataGridViewer/DataGridViewer.ipynb)
  * [Run in colab](https://colab.research.google.com/github/comet-ml/comet-examples/blob/master/panels/DataGridViewer/DataGridViewer.ipynb)
  * [Open in NBViewer](https://nbviewer.org/github/comet-ml/comet-examples/blob/master/panels/DataGridViewer/DataGridViewer.ipynb)
* Example Comet Project: [www.comet.com/examples/comet-example-datagrid](https://www.comet.com/examples/comet-example-datagrid/view/dVz9h6RFURYwHVQcgXvJ3RWqU/panels)
* Documentation: 
  * [DataGrid](https://github.com/dsblank/datagrid)
  * [Search syntax](https://github.com/dsblank/datagrid/blob/main/Search.md)
