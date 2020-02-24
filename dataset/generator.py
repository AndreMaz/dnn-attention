import datetime
import pandas as pd
# import tensorflow as tf

MONTH_NAMES_FULL = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December"
]


def first3Letters(month: list) -> list:
    MONTH_NAMES_3LETTER = []
    for index, value in enumerate(month):
        MONTH_NAMES_3LETTER.append(value[:3])

    return MONTH_NAMES_3LETTER


def generate(minYear: int, maxYear: int) -> list:
    daterange = pd.date_range(minYear, maxYear)
    dates = []
    for single_date in daterange:
        date: list = single_date.strftime("%Y-%m-%d").split('-')

        for index, value in enumerate(date):
            date[index] = int(date[index])

        dates.append(date)

    return dates

