"""
Copyright (C) 2020 Intel Corporation

SPDX-License-Identifier: BSD-3-Clause
"""

from shapely.geometry import LineString, box, Polygon

def get_polygon(point_list):
    return Polygon(point_list)


def get_box(points_tuple):
    return box(*points_tuple)


def get_line(data):
    return LineString(data)

