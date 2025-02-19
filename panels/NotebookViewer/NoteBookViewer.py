%pip install nbconvert

from comet_ml import API
import json
from nbconvert import HTMLExporter
from nbformat import read

st.set_page_config(layout="wide")


api = API()

columns = st.columns(2)

experiments = api.get_panel_experiments()

if len(experiments) == 0:
    print("No available experiments")
elif len(experiments) == 1:
    experiment = experiments[0]
    columns[0].markdown("Experiment:\\\n**%s**"  % (experiment.name or experiment.id))
else:
    experiment = columns[0].selectbox(
        "Select an experiment:", 
        experiments, 
        format_func=lambda experiment: (experiment.name or experiment.id)
    )

if experiment:
    assets = experiment.get_asset_list("notebook")
    if len(assets) == 0:
        notebook = None
    elif len(assets) == 1:
        notebook = assets[0]
        columns[1].markdown("Notebook:\\\n**%s**"  % notebook["fileName"])
    else:
        notebook = columns[1].selectbox(
            "Select a notebook:", 
            assets, 
            format_func=lambda asset: asset["fileName"]
        )
    if notebook:
        bytes = experiment.get_asset(
            notebook["assetId"], 
            return_type="binary"
        )
        with open("notebook.ipynb", "wb") as fp:
            fp.write(bytes)

        notebook_json = json.load(open("notebook.ipynb"))
        if len(notebook_json["cells"]) == 0:
            print("Notebook is empty")
            st.stop()
        if "metadata" in notebook_json and "widgets" in notebook_json["metadata"]:
            del notebook_json["metadata"]["widgets"]
            json.dump(notebook_json, open("notebook.ipynb", "w"))

        with open("notebook.ipynb", "r", encoding="utf-8") as f:
            nb = read(f, as_version=4)
        exporter = HTMLExporter()
        (output, resources) = exporter.from_notebook_node(nb)
        with open("fixed.html", "w", encoding="utf-8") as f:
            f.write(output)
        st.html("fixed.html")
    else:
        print("No notebooks available")
else:
    print("No experiment available")
        
st.markdown("""
<style>
.jp-InputPrompt {
    width: 10%;
}
.jp-OutputPrompt {
    width: 10%;
}
.stHtml {
  border: 1px solid;
  padding: 10px;
  box-shadow: 5px 10px lightgray;
}
</style>
""",
    unsafe_allow_html=True
)
