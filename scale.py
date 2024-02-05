def scale_into_number(v):
    s = []

    if v[0] == '10-19':
        s.append(1)
    elif v[0] == '20-29':
        s.append(2)
    elif v[0] == '30-39':
        s.append(3)
    elif v[0] == '40-49':
        s.append(4)
    elif v[0] == '50-59':
        s.append(5)
    elif v[0] == '60-69':
        s.append(6)
    elif v[0] == '70-79':
        s.append(7)
    elif v[0] == '80-89':
        s.append(8)
    elif v[0] == '90-99':
        s.append(9)
    else:
        s.append(v[0])

    if v[1] == 'lt40':
        s.append(1)
    elif v[1] == 'ge40':
        s.append(2)
    elif v[1] == 'premeno':
        s.append(3)
    else:
        s.append(v[1])

    if v[2] == '0-4':
        s.append(1)
    elif v[2] == '5-9':
        s.append(2)
    elif v[2] == '10-14':
        s.append(3)
    elif v[2] == '15-19':
        s.append(4)
    elif v[2] == '20-24':
        s.append(5)
    elif v[2] == '25-29':
        s.append(6)
    elif v[2] == '30-34':
        s.append(7)
    elif v[2] == '35-39':
        s.append(8)
    elif v[2] == '40-44':
        s.append(9)
    elif v[2] == '45-49':
        s.append(10)
    elif v[2] == '50-54':
        s.append(11)
    elif v[2] == '55-59':
        s.append(12)
    else:
        s.append(v[2])

    if v[3] == '0-2':
        s.append(1)
    elif v[3] == '3-5':
        s.append(2)
    elif v[3] == '6-8':
        s.append(3)
    elif v[3] == '9-11':
        s.append(4)
    elif v[3] == '12-14':
        s.append(5)
    elif v[3] == '15-17':
        s.append(6)
    elif v[3] == '18-20':
        s.append(7)
    elif v[3] == '21-23':
        s.append(8)
    elif v[3] == '24-26':
        s.append(9)
    elif v[3] == '27-29':
        s.append(10)
    elif v[3] == '30-32':
        s.append(11)
    elif v[3] == '33-35':
        s.append(12)
    elif v[3] == '36-39':
        s.append(13)
    else:
        s.append(v[3])

    if v[4] == 'yes':
        s.append(1)
    elif v[4] == 'no':
        s.append(2)
    else:
        s.append(v[4])

    if v[5] == '1':
        s.append(1)
    elif v[5] == '2':
        s.append(2)
    elif v[5] == '3':
        s.append(3)
    else:
        s.append(v[5])

    if v[6] == 'left':
        s.append(1)
    elif v[6] == 'right':
        s.append(2)
    else:
        s.append(v[6])

    a = v[7].split('_')
    if v[7] == 'central':
        s.append(5)
    elif a[0] == 'left' and a[1] == 'up':
        s.append(1)
    elif a[0] == 'left' and a[1] == 'low':
        s.append(2)
    elif a[0] == 'right' and a[1] == 'up':
        s.append(3)
    elif a[0] == 'right' and a[1] == 'low':
        s.append(4)
    else:
        s.append(v[7])

    if v[8] == 'yes':
        s.append(1)
    elif v[8] == 'no':
        s.append(2)
    else:
        s.append(v[8])

    if v[9] == 'no-recurrence-events':
        s.append(-1)
    elif v[9] == 'recurrence-events':
        s.append(1)
    else:
        s.append(v[9])
    return s
