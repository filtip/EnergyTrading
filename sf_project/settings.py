from dataclasses import dataclass
import datetime as dt
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


#Otevírání obchodů + zavírání obchodů (časové omezení)
open_until = dt.time(10,00)
close_time_limit = dt.time(11,00)

BEST_PREDICTORS = {
    "GAS_M_1": {
        "predictor_cols": ["M_1", "QY_1", "EM_M_1", "GAS_M_2"],
        "optimal_threshold": [0.05, 0.075, 0.1, 0.125, 0.15]},

    "GAS_M_2": {
        "predictor_cols": ["M_1", "QY_1", "GAS_M_1"],
        "optimal_threshold": [0.05, 0.075, 0.1, 0.125, 0.15]},

    "EM_M_1": {
        "predictor_cols": ["M_1", "QY_1", "Y_1", "GAS_M_1"],
        "optimal_threshold": [0.2, 0.3, 0.4, 0.5, 0.6]},

    "QY_2" : {
        "predictor_cols": ["M_1", "QY_1", "Y_1"],
        "optimal_threshold": [0.2, 0.3, 0.4, 0.5, 0.6]},

    "QY_3": {
        "predictor_cols": ["M_1", "QY_1", "Y_1"],
        "optimal_threshold": [0.2, 0.3, 0.4, 0.5, 0.6]},

    "QY_4": {
        "predictor_cols": ["QY_1","Y_1","EM_M_1"],
        "optimal_threshold": [0.2, 0.3, 0.4, 0.5, 0.6]},

    "QY_5": {
        "predictor_cols": ["QY_1","Y_1","EM_M_1","GAS_M_1"],
        "optimal_threshold": [0.2, 0.3, 0.4, 0.5, 0.6]},

    "Y_2": {
        "predictor_cols": ["M_1", "QY_1", "Y_1", "GAS_M_1"],
        "optimal_threshold": [0.2, 0.3, 0.4, 0.5, 0.6]},

    "M_2": {
        "predictor_cols": ["M_1", "QY_1", "EM_M_1","GAS_M_1"],
        "optimal_threshold": [0.2, 0.3, 0.4, 0.5, 0.6]},

    "M_3": {
        "predictor_cols": ["M_1", "QY_1", "EM_M_1","GAS_M_1"],
        "optimal_threshold": [0.2, 0.3, 0.4, 0.5, 0.6]},

    "M_4": {
        "predictor_cols": ["M_1", "QY_1", "Y_1", "GAS_M_1"],
        "optimal_threshold": [0.2, 0.3, 0.4, 0.5, 0.6]},

    "M_5": {
        "predictor_cols": ["QY_1", "Y_1", "EM_M_1", "GAS_M_1"],
        "optimal_threshold": [0.35, 0.525, 0.7, 0.875, 1.05]},

    "IT_M_1": {
        "predictor_cols": ["M_1","QY_1", "Y_1", "EM_M_1"],
        "optimal_threshold": [0.35, 0.525, 0.7, 0.875, 1.05]},

    "IT_QY_1": {
        "predictor_cols": ["M_1", "QY_1", "Y_1", "EM_M_1","GAS_M_1"],
        "optimal_threshold": [0.35, 0.525, 0.7, 0.875, 1.05]}}
