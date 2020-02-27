import datetime

def gen_dates():
    start = datetime.datetime.strptime("21-06-2010", "%d-%m-%Y")
    end = datetime.datetime.strptime("07-07-2014", "%d-%m-%Y")
    date_generated = [
        start + datetime.timedelta(days=x) for x in range(0, (end-start).days)
    ]

    for date in date_generated:
        print(date)

if __name__ == "__main__":
    gen_dates()
