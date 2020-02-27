import datetime
import pandas as pd
# import tensorflow as tf
import numpy as np

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

INPUT_LENGTH = 12 # Maximum length of all input formats.

def first3Letters(month: list) -> list:
    result = []
    for index, value in enumerate(month):
        result.append(value[:3].upper())

    return result

MONTH_NAMES_3LETTER = first3Letters(MONTH_NAMES_FULL)

def uniqueMonthLetters(monthList: list) -> str:
    letters = "".join(monthList)
    letters = sorted(set(letters))
    return "".join(letters)

INPUT_VOCAB = "\n0123456789/-., " + uniqueMonthLetters(first3Letters(MONTH_NAMES_FULL))

# OUTPUT_VOCAB includes an start-of-sequence (SOS) token, represented as
# '\t'. Note that the date strings are represented in terms of their
# constituent characters, not words or anything else.
OUTPUT_LENGTH = 10; # Length of 'YYYY-MM-DD'.
OUTPUT_VOCAB = "\n\t0123456789-"
START_CODE = 1


def toTwoDigitString(num: int) -> str:
   if (num < 10): return '0'+str(num)
    
   return str(num)

def dateTupleToDDMMMYYYY(dateTuple: list) -> str:
    # Date format such as 31DEC2050
    monthStr = MONTH_NAMES_3LETTER[dateTuple[1] - 1]
    dayStr = toTwoDigitString(dateTuple[2])

    return f"{dayStr}{monthStr}{dateTuple[0]}"

def dateTupleToMMSlashDDSlashYYYY(dateTuple: list) -> str:
    # Date format such as 01/20/2019
    monthStr = toTwoDigitString(dateTuple[1])
    dayStr = toTwoDigitString(dateTuple[2])

    return f"{monthStr}/{dayStr}/{dateTuple[0]}"

def dateTupleToMSlashDSlashYYYY(dateTuple: list) -> str:
    # Date format such as 1/20/2019
    return f"{dateTuple[1]}/{dateTuple[2]}/{dateTuple[0]}"

def dateTupleToMMSlashDDSlashYY(dateTuple: list) -> str:
    # Date format such as 01/20/19
    monthStr = toTwoDigitString(dateTuple[1])
    dayStr = toTwoDigitString(dateTuple[2])
    yearStr = f"{dateTuple[0]}"[2:]

    return f"{monthStr}/{dayStr}/{yearStr}"

def dateTupleToMSlashDSlashYY(dateTuple: list) -> str:
    # Date format such as 1/20/19
    yearStr = f"{dateTuple[0]}"[:2]
    return f"{dateTuple[1]}/{dateTuple[2]}/{yearStr}"

def dateTupleToMMDDYY(dateTuple: list) -> str:
    # Date format such as 012019
    monthStr = toTwoDigitString(dateTuple[1])
    dayStr = toTwoDigitString(dateTuple[2])
    yearStr = f"{dateTuple[0]}"[2:]

    return f"{monthStr}{dayStr}{yearStr}"

def dateTupleToMMMSpaceDDSpaceYY(dateTuple: list) -> str:
    # Date format such as JAN 20 19
    monthStr = MONTH_NAMES_3LETTER[dateTuple[1]-1]
    dayStr = toTwoDigitString(dateTuple[2])
    yearStr = f"{dateTuple[0]}"[2:]

    return f"{monthStr} {dayStr} {yearStr}"

def dateTupleToMMMSpaceDDSpaceYYYY(dateTuple: list) -> str:
    # Date format such as JAN 20 2019
    monthStr = MONTH_NAMES_3LETTER[dateTuple[1]-1]
    dayStr = toTwoDigitString(dateTuple[2])
    
    return f"{monthStr} {dayStr} {dateTuple[0]}"

def dateTupleToMMMSpaceDDCommaSpaceYY(dateTuple: list) -> str:
    # Date format such as JAN 20, 19
    monthStr = MONTH_NAMES_3LETTER[dateTuple[1]-1]
    dayStr = toTwoDigitString(dateTuple[2])
    yearStr = f"{dateTuple[0]}"[2:]

    return f"{monthStr} {dayStr}, {yearStr}"

def dateTupleToMMMSpaceDDCommaSpaceYYYY(dateTuple: list) -> str:
    # Date format such as JAN 20, 2019
    monthStr = MONTH_NAMES_3LETTER[dateTuple[1]-1]
    dayStr = toTwoDigitString(dateTuple[2])
    
    return f"{monthStr} {dayStr}, {dateTuple[0]}"

def dateTupleToDDDashMMDashYYYY(dateTuple: list) -> str:
    # Date format such as 20-01-2019
    monthStr = toTwoDigitString(dateTuple[1])
    dayStr = toTwoDigitString(dateTuple[2])
    return f"{dayStr}-{monthStr}-{dateTuple[0]}"

def dateTupleToDDashMDashYYYY(dateTuple: list) -> str:
    # Date format such as 20-1-2019
    return f"{dateTuple[2]}-{dateTuple[1]}-{dateTuple[0]}"

def dateTupleToDDDotMMDotYYYY(dateTuple: list) -> str:
    # Date format such as 20.01.2019
    monthStr = toTwoDigitString(dateTuple[1])
    dayStr = toTwoDigitString(dateTuple[2])
    return f"{dayStr}.{monthStr}.{dateTuple[0]}"

def dateTupleToDDotMDotYYYY(dateTuple: list) -> str:
    # Date format such as 20.1.2019
    return f"{dateTuple[2]}.{dateTuple[1]}.{dateTuple[0]}"

def dateTupleToYYYYDotMMDotDD(dateTuple: list) -> str:
    # Date format such as 2019.01.20
    monthStr = toTwoDigitString(dateTuple[1])
    dayStr = toTwoDigitString(dateTuple[2])
    return f"{dateTuple[0]}.{monthStr}.{dayStr}"

def dateTupleToYYYYDotMDotD(dateTuple: list) -> str:
    # Date format such as 2019.1.20
    return f"{dateTuple[0]}.{dateTuple[1]}.{dateTuple[2]}"


def dateTupleToYYYYMMDD(dateTuple: list) -> str:
    # Date format such as 20190120
    monthStr = toTwoDigitString(dateTuple[1])
    dayStr = toTwoDigitString(dateTuple[2])
    return f"{dateTuple[0]}{monthStr}{dayStr}"

def dateTupleToYYYYDashMDashD(dateTuple: list) -> str:
    # Date format such as 2019-1-20
    monthStr = toTwoDigitString(dateTuple[1])
    dayStr = toTwoDigitString(dateTuple[2])
    return f"{dateTuple[0]}{monthStr}{dayStr}"

def dateTupleToDSpaceMMMSpaceYYYY(dateTuple: list) -> str:
    # Date format such as 20 JAN 2019
    monthStr = MONTH_NAMES_3LETTER[dateTuple[1]-1]
    return f"{dateTuple[2]} {monthStr} {dateTuple[0]}"

def dateTupleToYYYYDashMMDashDD(dateTuple: list) -> str:
    # Date format (ISO) such as 2019-01-20
    monthStr = toTwoDigitString(dateTuple[1])
    dayStr = toTwoDigitString(dateTuple[2])
    return f"{dateTuple[0]}-{monthStr}-{dayStr}"

INPUT_FNS = [
  dateTupleToDDMMMYYYY,
  dateTupleToMMDDYY,
  dateTupleToMMSlashDDSlashYY,
  dateTupleToMMSlashDDSlashYYYY,
  dateTupleToMSlashDSlashYYYY,
  dateTupleToDDDashMMDashYYYY,
  dateTupleToDDashMDashYYYY,
  dateTupleToMMMSpaceDDSpaceYY,
  dateTupleToMSlashDSlashYY,
  dateTupleToMMMSpaceDDSpaceYYYY,
  dateTupleToMMMSpaceDDCommaSpaceYY,
  dateTupleToMMMSpaceDDCommaSpaceYYYY,
  dateTupleToDDDotMMDotYYYY,
  dateTupleToDDotMDotYYYY,
  dateTupleToYYYYDotMMDotDD,
  dateTupleToYYYYDotMDotD,
  dateTupleToYYYYMMDD,
  dateTupleToYYYYDashMDashD,
  dateTupleToDSpaceMMMSpaceYYYY,
  dateTupleToYYYYDashMMDashDD
]

def encodeInputDateStrings(dateStrings: list):
    numStrings = len(dateStrings)
    x = np.zeros((numStrings, INPUT_LENGTH), dtype="float32")

    for i, value in enumerate(dateStrings):
        for j in range(INPUT_LENGTH):
            if (j < len(dateStrings[i])):
                char = dateStrings[i][j]
                index = INPUT_VOCAB.index(char)
                x[i,j] = index

    return x

def encodeOutputDateStrings(dateStrings: list):
    numStrings = len(dateStrings)
    x = np.zeros((numStrings, OUTPUT_LENGTH), dtype='int32')

    for i, value in enumerate(dateStrings):
        for j in range(OUTPUT_LENGTH):
            char = dateStrings[i][j]
            index = OUTPUT_VOCAB.index(char)
            x[i,j] = index

    return x

