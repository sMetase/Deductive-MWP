import csv

id = '143'
pred = '0.25'

data = []
data.append(['id', 'pred'])
data.append([f'{id}', f'{pred}'])

filename = 'data/test.csv'

with open(filename, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(data)