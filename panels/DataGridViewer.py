%pip install datagrid st-click-detector st-tools

# Visualize tabluar data containing images, and data:

# >>> from datagrid import DataGrid, Image

# >>> dg = DataGrid(
# >>>     columns=["Image", "Score", "Category"],
# >>>     name="Demo"
# >>> )

# Loop:
# >>>     image = Image(im, metadata={"category": category, "score": score})
# >>>     dg.append([image, score, category])

# >>> dg.log(experiment)

st.set_page_config(layout="wide", initial_sidebar_state="collapsed") 

from comet_ml import API
import math
from st_click_detector import click_detector
from st_tools import SessionStateManager
import pandas as pd
import datagrid
from datagrid.app import (
    build_table,
    render_image_dialog,
    render_text_dialog,
    render_float_dialog,
    render_integer_dialog,
    render_boolean_dialog,
    render_json_dialog,
    render_download_dialog,
)
from datagrid._datatypes.utils import (
    get_color,
)
from datagrid.server.queries import (
    select_query_page,
    select_query_count,
    select_category,
    select_histogram,
    generate_chart_image,
    get_completions,
    verify_where,
)
import base64
import io
import os
import zipfile
import json

api = API()

DATAGRID = ""
BASEURL = api._client.server_url

def get_pagesize(group_by):
    if group_by:
        return GROUP_PAGESIZE
    else:
        return PAGESIZE

def set_offset(offset):
    st.session_state["offset"] = offset
    config.save()


def reset_where_for_histogram(key, column, group_by, group_by_value, bins):
    column_name = column.replace(" ", "__").lower()
    column_where = '(%s == "%s")' % (
        group_by.replace(" ", "__").lower(),
        group_by_value,
    )
    st.session_state["offset"] = 0
    st.session_state["sort_order"] = "Ascending"
    st.session_state["sort_by"] = column
    st.session_state["group_by"] = ""
    ranges = []
    for item in st.session_state[key]["selection"]["points"]:
        start = item["point_index"]
        stop = item["point_index"] + 1
        ranges.append("""%s <= %s < %s""" % (bins[start], column_name, bins[stop]))
    if st.session_state.where:
        if not st.session_state.where.startswith("("):
            st.session_state.where = "(" + st.session_state.where + ")"
        if column_where not in st.session_state.where:
            st.session_state.where += " and " + column_where
        st.session_state.where += " and (" + (" or ".join(ranges)) + ")"
    else:
        st.session_state.where = column_where + " and (" + (" or ".join(ranges)) + ")"
    config.save()


def reset_where_for_category(key, column, group_by, group_by_value):
    column_name = column.replace(" ", "__").lower()
    column_where = '(%s == "%s")' % (
        group_by.replace(" ", "__").lower(),
        group_by_value,
    )
    st.session_state["offset"] = 0
    st.session_state["sort_order"] = "Ascending"
    st.session_state["sort_by"] = column
    st.session_state["group_by"] = ""
    values = []
    for point in st.session_state[key]["selection"]["points"]:
        values.append('"%s"' % point["y"])
    if st.session_state.where:
        if not st.session_state.where.startswith("("):
            st.session_state.where = "(" + st.session_state.where + ")"
        if column_where not in st.session_state.where:
            st.session_state.where += " and " + column_where
        st.session_state.where += """ and (%s in [%s])""" % (
            column_name,
            (",".join(values)),
        )
    else:
        st.session_state.where = """%s and (%s in [%s])""" % (
            column_where,
            column_name,
            (",".join(values)),
        )
    config.save()


def reset_all():
    if "datagrid" in st.session_state:
        st.session_state["datagrid"] = DATAGRID
        st.session_state["offset"] = 0
        st.session_state["sort_by"] = "row-id"
        st.session_state["sort_order"] = "Ascending"
        st.session_state["group_by"] = ""
        st.session_state["where"] = ""
    config.save()


def reset_offset():
    st.session_state["offset"] = 0
    config.save()


@st.cache_data(persist="disk", show_spinner="Loading getting datagrid asset...")
def experiment_get_asset(_experiment, experiment_id, asset_id, return_type):
    return _experiment.get_asset(asset_id, return_type=return_type)


# @st.cache_data(persist="disk", show_spinner="Loading datagrids...")
def get_datagrids(_experiments, experiment_keys):
    datagrids = [("", None, None)]
    for experiment in _experiments:
        name = experiment.get_name()
        assets = experiment.get_asset_list(asset_type="datagrid")
        for asset in assets:
            datagrids.append(
                (
                    "%s: %s" % (name, os.path.basename(asset["fileName"])),
                    experiment,
                    asset["assetId"],
                )
            )
    return datagrids


# @st.cache_data(persist="disk")
def get_cached_completions(datagrid_name):
    return get_completions(datagrid_name, None)


experiments = api.get_panel_experiments()
datagrids = {
    name: (name, exp, asset_id)
    for name, exp, asset_id in get_datagrids(
        experiments, [experiment.id for experiment in experiments]
    )
}


config = SessionStateManager(
    instance_id=api.get_panel_options().get("instanceId"),
    main_key=("selected_datagrid", "", datagrids.keys()),
    keys=["pagesize", "group_pagesize", "decimal_precision",
          "integer_separator", "group_by", "sort_by", "sort_order",
          "where", "datagrid", "offset", "table_id"],
)

# Controls:
columns = st.columns([0.75, 0.4, 0.4, 0.4, 1])

selected_datagrid = columns[0].selectbox(
    "DataGrid:",
    sorted(datagrids.keys()),
    # index=1 if len(datagrids) == 2 else 0,
    # format_func=lambda item: item[0],
    key="selected_datagrid",
    on_change=reset_all,
)

if not selected_datagrid:
    columns[1].selectbox(
        "Group by:",
        [],
        disabled=True,
    )
    columns[2].selectbox(
        "Sort by:",
        [],
        disabled=True,
    )
    columns[3].selectbox(
        "Sort order:",
        [],
        disabled=True,
    )
    columns[4].text_input(
        "[ℹ️](https://github.com/comet-ml/kangas/wiki/Filter-Expressions) Search: ",
        placeholder='column_name > 0.5 or column name.json_field == "value"',
        disabled=True,
    )
else:
    name, experiment, asset_id = datagrids[selected_datagrid]

    # name is Experiment: Datagrid-10.json.zip
    dg_name = name.split(": ")[1].rsplit(".", 2)[0]
    zip_name = experiment.id + "/" + dg_name + ".json.zip"
    json_name = experiment.id + "/" + dg_name + ".json"
    DATAGRID = experiment.id + "/" + dg_name + ".datagrid"

    config.initialize(
        {
            "pagesize": 8,
            "group_pagesize": 4,
            "decimal_precision": 5,
            "integer_separator": True,
            "group_by": "",
            "sort_by": None,
            "where": "",
            "datagrid": DATAGRID,
            "sort_order": "Ascending",
            "offset": 0,
            "table_id": 1,
        },
    )

    with st.sidebar:
        st.title("DataGrid Viewer Settings")
        PAGESIZE = st.number_input(
            "Rows per page:", min_value=1, max_value=100, key="pagesize", on_change=config.save
        )
        GROUP_PAGESIZE = st.number_input(
            "Grouped rows per page:", min_value=1, max_value=20, key="group_pagesize", on_change=config.save
        )
        DECIMAL_PRECISION = st.selectbox(
            "Decimal precision:", [None, 0, 1, 2, 3, 4, 5, 6], key="decimal_precision", on_change=config.save
        )
        INTEGER_SEPARATOR = st.checkbox(
            "Use thousands separator for integers", key="integer_separator", on_change=config.save
        )

    if not os.path.exists(DATAGRID):
        bar = st.progress(0, "Downloading datagrid...")
        binary = experiment_get_asset(
            experiment, experiment.id, asset_id, return_type="binary"
        )
        bar.progress(50, "Writing datagrid...")
        os.makedirs(experiment.id, exist_ok=True)
        with open(zip_name, "wb") as fp:
            fp.write(binary)
        bar.progress(75, "Unzipping datagrid")
        with zipfile.ZipFile(zip_name, "r") as zip_ref:
            zip_ref.extractall(experiment.id)
        with open(json_name, "r") as fp:
            datagrid_json = json.load(fp)
        new_columns = datagrid_json["columns"]
        dg = datagrid._DataGrid(
            datagrid_json["rows"], columns=new_columns, verify=False
        )
        dg.save(DATAGRID)
        bar.empty()

    # Get datagrid, and show it:
    dg = datagrid.read_datagrid(DATAGRID)
    schema = dg.get_schema()
    # completions = get_cached_completions(DATAGRID)
    # {key: list} where key matches last len(key) chars

    column_names_for_sort = [
        column
        for column in dg.get_columns()
        if schema[column]["type"] not in ["IMAGE-ASSET"]
    ]
    column_names_for_group = [
        column
        for column in dg.get_columns()
        if schema[column]["type"] not in ["IMAGE-ASSET"]
    ]

    group_by = columns[1].selectbox(
        "Group by:",
        [""] + column_names_for_group,
        key="group_by",
        on_change=reset_offset,
    )
    sort_by = columns[2].selectbox(
        "Sort by:",
        ["row-id"] + column_names_for_sort,
        key="sort_by",
        disabled=True if group_by else False,
        on_change=config.save
    )
    columns[3].selectbox(
        "Sort order:",
        ["Ascending", "Descending"],
        key="sort_order",
        on_change=config.save
    )
    where = columns[4].text_input(
        "[ℹ️](https://github.com/comet-ml/kangas/wiki/Filter-Expressions) Search: ",
        placeholder='column_name > 0.5 or column name.json_field == "value"',
        key="where",
        on_change=reset_offset,
        autocomplete="off",
    )
    if where != "":
        verify_results = verify_where(DATAGRID, None, where)
        if not verify_results["valid"]:
            st.error(f'Invalid Python search expression: {verify_results["message"]}')

    # ------------------------------ Group
    if group_by:
        data = select_query_page(
            dgid=DATAGRID,
            offset=st.session_state["offset"],
            group_by=group_by,
            sort_by=group_by,
            sort_desc=st.session_state["sort_order"] == "Descending",
            where=None,
            limit=GROUP_PAGESIZE,
            select_columns=dg.get_columns(),
            computed_columns=None,
            where_expr=where,
            debug=False,
            timestamp=None,
        )["rows"]
        count = select_query_count(
            dgid=DATAGRID,
            group_by=group_by,
            computed_columns=None,
            where_expr=where,
        )
    else:
        # Non-grouped view
        data = dg.select(
            where=where,
            sort_by=sort_by,
            select_columns=["row-id"] + dg.get_columns(),
            sort_desc=st.session_state["sort_order"] == "Descending",
            to_dicts=True,
            limit=PAGESIZE,
            offset=st.session_state["offset"],
        )
        count = select_query_count(
            dgid=DATAGRID,
            group_by=None,
            computed_columns=None,
            where_expr=where,
        )

    if data:
        table, width = build_table(
            DATAGRID,
            group_by,
            where,
            data,
            schema,
            experiment,
            st.session_state["table_id"],
            config={
                "decimal_precision": DECIMAL_PRECISION,
                "integer_separator": INTEGER_SEPARATOR,
            },
        )
        col_row = click_detector(table)
        if col_row:
            col, row = [int(v) for v in col_row.split(",")]
            column_name = list(data[row].keys())[col]
            value = list(data[row].values())[col]
            cell_type = schema[column_name]["type"]

            # "INTEGER", "FLOAT", "BOOLEAN", "TEXT", "JSON"
            # "IMAGE-ASSET", "VIDEO-ASSET", "CURVE-ASSET", "ASSET-ASSET", "AUDIO-ASSET"
            if cell_type == "IMAGE-ASSET":
                render_image_dialog(BASEURL, group_by, value, schema, experiment, config)
            elif cell_type == "TEXT":
                render_text_dialog(
                    BASEURL,
                    group_by,
                    value,
                    schema,
                    experiment,
                    reset_where_for_category,
                    config,
                )
            elif cell_type == "FLOAT":
                render_float_dialog(
                    BASEURL,
                    group_by,
                    value,
                    schema,
                    experiment,
                    reset_where_for_histogram,
                    config,
                )
            elif cell_type == "INTEGER":
                render_integer_dialog(
                    BASEURL,
                    group_by,
                    value,
                    schema,
                    experiment,
                    reset_where_for_category,
                    config,
                )
            elif cell_type == "BOOLEAN":
                render_boolean_dialog(BASEURL, group_by, value, schema, experiment, config)
            elif cell_type == "JSON":
                render_json_dialog(BASEURL, group_by, value, schema, experiment, config)
            else:
                print("Unsupported expanded render type: %r" % cell_type)

        first_row = st.session_state["offset"] + 1
        total_pages = math.ceil(count / get_pagesize(group_by))
        max_row = min(
            st.session_state["offset"] + get_pagesize(group_by), count
        )
        current_page = (
            math.floor(st.session_state["offset"] / get_pagesize(group_by))
            + 1
        )

        columns = st.columns([2, 6, 1, 1, 1, 1, 1])
        columns[0].markdown(
            f"""<div style="text-align: center; padding-top: 5px;">Showing {first_row} - {max_row} of {count} rows</div>""",
            unsafe_allow_html=True,
        )
        if columns[1].button("Download Selected"):
            render_download_dialog(BASEURL, dg, schema, where, experiment, config)
        if columns[2].button(
            "| <", use_container_width=True, disabled=current_page == 1
        ):
            set_offset(0)
            st.rerun()
        if columns[3].button("<", use_container_width=True, disabled=current_page == 1):
            set_offset(st.session_state["offset"] - get_pagesize(group_by))
            st.rerun()
        columns[4].markdown(
            '<div style="text-align: center; padding-top: 5px;">Page %d/%d</div>'
            % (current_page, total_pages),
            unsafe_allow_html=True,
        )
        if columns[5].button(
            "&gt;", use_container_width=True, disabled=current_page == total_pages
        ):
            set_offset(st.session_state["offset"] + get_pagesize(group_by))
            st.rerun()
        if columns[6].button(
            "&gt; |", use_container_width=True, disabled=current_page == total_pages
        ):
            set_offset((total_pages - 1) * get_pagesize(group_by))
            st.rerun()
    else:
        st.warning("No data matching the filter criteria")

styles = f"""
<style>
  div[data-testid="stMainBlockContainer"] {{
    padding: 58px;
  }}
</style>
"""
st.markdown(styles, unsafe_allow_html=True)
