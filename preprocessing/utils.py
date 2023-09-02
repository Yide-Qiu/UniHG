
import requests
import pdb
import os
import time

def entity_or_attribute(ss):
    if ss[0] == ('P' or 'Q') and ss[1:].isdigit() == True:
        return True
    return False


def is_attribute(ss):
    if len(ss) < 2:
        return False
    if ss[0] == 'P' and ss[1:].isdigit() == True:
        return True
    return False

def is_entity(ss):
    if len(ss) < 2:
        return False
    if ss[0] == 'Q' and ss[1:].isdigit() == True:
        return True
    return False

def Bad_Attribute(ss):
    if ss.find('ID') >= 0:
        return True
    
    return False

def entity_select(js):
    # True means be select
    if 'en' not in js['labels'].keys() or 'en' not in js['descriptions']:
        return False
    if 'P31' not in js['claims'].keys():
        return False
    if 'datavalue' not in js['claims']['P31'][0]['mainsnak'].keys():
        return False
    return True

def label_select(js):
    if 'en' not in js['labels'].keys() or 'en' not in js['descriptions']:
        return False
    return True






