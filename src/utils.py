import yaml
from collections import namedtuple


# class YamlStruct:
#     def __init__(self, **kwargs):
#         for key, value in kwargs.items():
#             if isinstance(value, dict):
#                 self.__dict__[key] = YamlStruct(**value)
#             else:
#                 self.__dict__[key] = value
#
#
# def open_yaml(filepath):
#     with open(filepath, 'r') as f:
#         content_dict = yaml.safe_load(f)
#
#     return YamlStruct(**content_dict)


def open_yaml_namedtuple(filepath):
    with open(filepath, 'r') as f:
        content_dict = yaml.safe_load(f)

    return dict_to_namedtuple('Config', content_dict)


def dict_to_namedtuple(name, content):
    result = {}
    for key, value in content.items():
        if isinstance(value, dict):
            result[key] = dict_to_namedtuple(key.capitalize(), value)
        else:
            result[key] = value

    return namedtuple(name, result.keys())(**result)
