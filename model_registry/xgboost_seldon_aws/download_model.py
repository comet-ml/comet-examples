#!/usr/bin/env python
# -*- coding: utf-8 -*-
# *******************************************************
#   ____                     _               _
#  / ___|___  _ __ ___   ___| |_   _ __ ___ | |
# | |   / _ \| '_ ` _ \ / _ \ __| | '_ ` _ \| |
# | |__| (_) | | | | | |  __/ |_ _| | | | | | |
#  \____\___/|_| |_| |_|\___|\__(_)_| |_| |_|_|
#
#  Sign up for free at http://www.comet.ml
#  Copyright (C) 2015-2020 Comet ML INC
#  This file can not be copied and/or distributed without
#  the express permission of Comet ML Inc.
# *******************************************************

"""
Explore an offline experiment archive.

Display summaries of an offline experiments:

$ comet offline *.zip

Display CSV (Comma-Separated Value) format. Shows an
experiment's data in a row format:

Workspace, Project, Experiment, Level, Section, Name, Value

where:

* level: detail, maximum, or minimum
* section: metric, param, log_other, etc.
* name: name of metric, param, etc.

$ comet offline --csv *.zip

Use --level, --section, or --name to filter the rows.
"""

import sys
import argparse
from comet_ml import API

parser = argparse.ArgumentParser(description="Set seldon-core version")
parser.add_argument("--workspace", required=True)
parser.add_argument("--registry_name", required=True)
parser.add_argument("--model_version", required=True)

opts = parser.parse_args()

api = API()

api.download_registry_model(
    opts.workspace, opts.registry_name, opts.model_version, output_path="model", expand=True
)
