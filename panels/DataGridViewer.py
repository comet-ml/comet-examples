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

HELP_SEARCH = """
DataGrid Search uses Python syntax for selecting matching rows. To use the value of a column in the expression, just use the name of the column like this `score`, or if it has a space in it, like this `column__1` (two underscores).

Some examples:

To select all of the rows that have a fitness score less than 0.1, given that you have a column named "Fitness":

```
fitness < 0.1
```

Note that the column name is case-insensitive (i.e., you can use any case of letters). Column values can be used any where in the filter expression. For example, if you wanted to select all of the rows where column "Score 1" was greater than or equal to "Score 2", you would write:

```
score__1 >= score__2
```

You can use any of Python's operators, including:

* `<` - less than
* `<=` - less than or equal
* `>` - greater than
* `>=` - greater than or equal
* `==` - equal to
* `!=` - not equal to
* `is` - is the same (e.g., is None)
* `is not` - is not the same (e.g. is not None)
* `+` - addition
* `-` - subtraction
* `*` - multiplication
* `/` - division
* `//` - integer division
* `**` - raise to a power
* `not` - flip the boolean value
* `in` - is value in a list of values

Also you can use Python's comparison operator chaining. That is, any of the above operators can be used in a shorthand way as follows:

```
column_a < column_b < column_c
```

which is shorthand for:

```
column_a < column_b and column_b < column c
```

Note: the single underscore matches a literal underscore, where a double underscore matches a space, in the column name or attribute name.

Use `is None` and `is not None` for selecting rows that have null values. Otherwise, null values are ignored in most other uses.

You can combine comparisons using Python's and and or (use parentheses to force evaluation order different from Python's defaults):

```
((loss - base__line) < 0.5) or (loss > 0.95)
```

## JSON attribute access

For JSON and metadata columns, you can use the dot operator to access values:

```
image.extension == "jpg"
```

or nested values:

```
image.labels.dog > 2
```

The dotted-name path following a column of JSON data deviates from Python semantics. In addition, the path can only be use for nested dictionaries. If the JSON has a list, then you will not be able to use the dotted-name syntax. However, you can use Python's list comprehension, like these examples:

* `any([x["label"] == 'dog' for x in image.overlays])` - images with dogs
* `all([x["label"] == 'person' for x in image.overlays])` - images with only people (no other labels)
* `any([x["label"] in ["car", "bicycle"] for x in image.overlays])` - images with cars or bicycles
* `any([x["label"] == "cat" or x["label"] == "dog" for x in image.overlays])` - images with dogs or cats
* `any([x["label"] == "cat" for x in image.overlays]) and any([x["label"] == "dog" for x in image.overlays])` - images with dogs and cats
* `any([x["score"] > 0.999 for x in image.overlays])` - images with an annotation score greater than 0.999

List comprehension also deviates slightly from standard Python semantics:

* `[item for item in LIST]` - same as Python (item is each element in the list)
* `[item for item in DICT]` - item is the DICT

Note that you will need to wrap the list comprehension in either `any()` or `all()`. You can also use `flatten()` around nested list comprehensions.

See below for more information on other string and JSON methods.

Note that any mention of a column that contains an asset type (e.g., an image) will automatically reference its metadata.

## Python String and JSON methods

These are mostly used with column values:

* `STRING in Column__Name`
* `Column__Name.endswith(STRING)`
* `Column__Name.startswith(STRING)`
* `Column__Name.strip()`
* `Column__Name.lstrip()`
* `Column__Name.rstrip()`
* `Column__Name.upper()`
* `Column__Name.lower()`
* `Column__Name.split(DELIM[, MAXSPLITS])`

## Python if/else expression

You can use Python's if/else expression to return one value or another.

```
("low" if loss < 0.5 else "high") == "low"
```

## Python Builtin Functions

You can use any of these Python builtin functions:

* `abs()` - absolute value
* `round()` - rounds to int
* `max(v1, v2, ...)` or `max([v1, v2, ...])` - maximum of list of values
* `min(v1, v2, ...)` or `min([v1, v2, ...])` - minimum of list of values
* `len()` - length of item

## Aggregate Functions

You can use the following aggregate functions. Note that these are more expensive to compute, as they require computing on all rows.

* `AVG(Column__Name)`
* `MAX(Column__Name)`
* `MIN(Column__Name)`
* `SUM(Column__Name)`
* `TOTAL(Column__Name)`
* `COUNT(Column__Name)`
* `STDEV(Column__Name)`

Examples:

Find all rows that have a loss value less than the average:

```
loss < AVG(loss)
```

## Python Library Functions and Values

Functions from Python's random library:

* `random.random()`
* `random.randint()`

Note that random values are regenerated on each call, and thus only have limited use. That means that every time the DataGrid is accessed, the values will change. This would result in bizarre results if you grouped or sorted by a random value.

Functions and values from Python's math library:

* `math.pi`
* `math.sqrt()`
* `math.acos()`
* `math.acosh()`
* `math.asin()`
* `math.asinh()`
* `math.atan()`
* `math.atan2()`
* `math.atanh()`
* `math.ceil()`
* `math.cos()`
* `math.cosh()`
* `math.degrees()`
* `math.exp()`
* `math.floor()`
* `math.log()`
* `math.log10()`
* `math.log2()`
* `math.radians()`
* `math.sin()`
* `math.sinh()`
* `math.tan()`
* `math.tanh()`
* `math.trunc()`

### Functions and values from Python's datetime library:

* `datetime.date(YEAR, MONTH, DAY)`
* `datetime.datetime(YEAR, MONTH, DAY[, HOUR, MINUTE, SECOND])`

"""
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
    keys=[
        "pagesize",
        "group_pagesize",
        "decimal_precision",
        "integer_separator",
        "group_by",
        "sort_by",
        "sort_order",
        "where",
        "datagrid",
        "offset",
        "table_id",
    ],
)

# Controls:
columns = st.columns([0.75, 0.4, 0.4, 0.4, 1])

selected_datagrid = columns[0].selectbox(
    "DataGrid",
    sorted(datagrids.keys()),
    # index=1 if len(datagrids) == 2 else 0,
    # format_func=lambda item: item[0],
    key="selected_datagrid",
    on_change=reset_all,
)

if not selected_datagrid:
    columns[1].selectbox(
        "Group by",
        [],
        placeholder="Select a column",
        disabled=True,
    )
    columns[2].selectbox(
        "Sort by",
        [],
        placeholder="Select a column",
        disabled=True,
    )
    columns[3].selectbox(
        "Sort order",
        [],
        placeholder="Select a column",
        disabled=True,
    )
    columns[4].text_input(
        "Search",
        placeholder='column_name > 0.5 or column name.json_field == "value"',
        help=HELP_SEARCH,
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
            "Rows per page",
            min_value=1,
            max_value=100,
            key="pagesize",
            on_change=config.save,
        )
        GROUP_PAGESIZE = st.number_input(
            "Grouped rows per page",
            min_value=1,
            max_value=20,
            key="group_pagesize",
            on_change=config.save,
        )
        DECIMAL_PRECISION = st.selectbox(
            "Decimal precision",
            [None, 0, 1, 2, 3, 4, 5, 6],
            key="decimal_precision",
            on_change=config.save,
        )
        INTEGER_SEPARATOR = st.checkbox(
            "Use thousands separator for integers",
            key="integer_separator",
            on_change=config.save,
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
        "Group by",
        [""] + column_names_for_group,
        placeholder="Select a column",
        key="group_by",
        on_change=reset_offset,
    )
    sort_by = columns[2].selectbox(
        "Sort by",
        ["row-id"] + column_names_for_sort,
        key="sort_by",
        placeholder="Select a column",
        disabled=True if group_by else False,
        on_change=config.save,
    )
    columns[3].selectbox(
        "Sort order",
        ["Ascending", "Descending"],
        key="sort_order",
        on_change=config.save,
    )
    where = columns[4].text_input(
        "Search",
        placeholder='column_name > 0.5 or column name.json_field == "value"',
        key="where",
        help=HELP_SEARCH,
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
                render_image_dialog(
                    BASEURL, group_by, value, schema, experiment, config
                )
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
                render_boolean_dialog(
                    BASEURL, group_by, value, schema, experiment, config
                )
            elif cell_type == "JSON":
                render_json_dialog(BASEURL, group_by, value, schema, experiment, config)
            else:
                print("Unsupported expanded render type: %r" % cell_type)

        first_row = st.session_state["offset"] + 1
        total_pages = math.ceil(count / get_pagesize(group_by))
        max_row = min(st.session_state["offset"] + get_pagesize(group_by), count)
        current_page = (
            math.floor(st.session_state["offset"] / get_pagesize(group_by)) + 1
        )

        columns = st.columns([1, 0.5, 1])
        left_side = columns[0].columns([1.1, 1])
        left_side[0].markdown(
            f"""<div style="text-align: left; padding-top: 5px; white-space: nowrap;">Rows {first_row} - {max_row} of {count}</div>""",
            unsafe_allow_html=True,
        )
        if left_side[1].button("Download..."):
            render_download_dialog(BASEURL, dg, schema, where, experiment, config)

        right_side = columns[2].columns([1, 1, 3.5, 1, 1])
        if right_side[0].button(
            "", icon=":material/first_page:", disabled=current_page == 1
        ):
            set_offset(0)
            st.rerun()
        if right_side[1].button(
            "", icon=":material/chevron_left:", disabled=current_page == 1
        ):
            set_offset(st.session_state["offset"] - get_pagesize(group_by))
            st.rerun()
        right_side[2].markdown(
            '<div style="text-align: center; padding-top: 5px; white-space: nowrap;">Page %d/%d</div>'
            % (current_page, total_pages),
            unsafe_allow_html=True,
        )
        #right_side[2].button(
        #    'Page %d/%d' % (current_page, total_pages),
        #    use_container_width=True,
        #)
        if right_side[3].button(
            "", icon=":material/chevron_right:", disabled=current_page == total_pages
        ):
            set_offset(st.session_state["offset"] + get_pagesize(group_by))
            st.rerun()
        if right_side[4].button(
            "", icon=":material/last_page:", disabled=current_page == total_pages
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
