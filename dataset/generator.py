import pandas as pd
import numpy as np
# import tensorflow as tf

from dataset.date_format import INPUT_FNS, encodeInputDateStrings, encodeOutputDateStrings, dateTupleToYYYYDashMMDashDD

def generateOrderedDates(minYear: str, maxYear: str) -> list:
    daterange = pd.date_range(minYear, maxYear)
    dates = []
    for single_date in daterange:
        date: list = single_date.strftime("%Y-%m-%d").split('-')

        for index, value in enumerate(date):
            date[index] = int(date[index])

        dates.append(date)

    return dates

def dateTuplesToTensor(dateTuples):
    inputsStrings = []

    for _, fn in enumerate(INPUT_FNS):
        for _, dateTuple in enumerate(dateTuples):
            formatedDate = fn(dateTuple)
            inputsStrings.append(encodeInputDateStrings(formatedDate))
    
    return inputsStrings

def generateDataSet(minYear="1950-01-01", maxYear="1950-02-01", trainSplit=0.25, validationSplit=0.15):
    dateTuples = generateOrderedDates(minYear, maxYear)

    np.random.shuffle(dateTuples)

    return dateTuplesToTensor(dateTuples)